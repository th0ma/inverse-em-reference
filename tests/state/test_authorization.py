import pytest

from inverse_em.errors import SchemaValidationError, StateTransitionError
from inverse_em.scientific.authorization import AuthorizationReceipt


ZERO = "0" * 64


def valid_receipt():
    return {
        "schema_version": "authorization-receipt/1.0", "study_id": "fixture",
        "predecessor_state": "FROZEN", "requested_state": "SEALED_CLEAN",
        "configuration_sha256": ZERO,
        "predecessor_receipt_sha256": ZERO, "issuer_label": "synthetic-test",
        "issued_at": "2026-01-01T00:00:00+00:00",
        "best_checkpoint": {"status": "PRESENT", "value": ZERO, "reason": None},
        "population": {"status": "PRESENT", "value": ZERO, "reason": None},
        "normalizer": {"status": "PRESENT", "value": ZERO, "reason": None},
    }


def test_receipt_validates_and_hashes():
    receipt = AuthorizationReceipt.from_mapping(valid_receipt())
    assert len(receipt.sha256) == 64


def test_receipt_rejects_illegal_transition():
    data = valid_receipt(); data["requested_state"] = "ROBUSTNESS"
    with pytest.raises(StateTransitionError):
        AuthorizationReceipt.from_mapping(data)


def test_receipt_rejects_bad_digest():
    data = valid_receipt(); data["configuration_sha256"] = "bad"
    with pytest.raises(SchemaValidationError):
        AuthorizationReceipt.from_mapping(data)


def test_receipt_rejects_naive_timestamp():
    data = valid_receipt(); data["issued_at"] = "2026-01-01T00:00:00"
    with pytest.raises(SchemaValidationError):
        AuthorizationReceipt.from_mapping(data)


@pytest.mark.parametrize("version", ["authorization-receipt/2.0", "", 1, None])
def test_receipt_rejects_unsupported_schema(version):
    data = valid_receipt(); data["schema_version"] = version
    with pytest.raises(SchemaValidationError):
        AuthorizationReceipt.from_mapping(data)


def test_receipt_rejects_missing_schema():
    data = valid_receipt(); del data["schema_version"]
    with pytest.raises(SchemaValidationError):
        AuthorizationReceipt.from_mapping(data)


PROTECTED_TRANSITIONS = [
    ("TRAINING", "FROZEN"),
    ("FROZEN", "SEALED_CLEAN"),
    ("SEALED_CLEAN", "ROBUSTNESS"),
]


@pytest.mark.parametrize("before,after", PROTECTED_TRANSITIONS)
@pytest.mark.parametrize("field", ["best_checkpoint", "population", "normalizer"])
@pytest.mark.parametrize("status", ["MISSING", "UNRESOLVED", "NOT_APPLICABLE"])
def test_protected_transition_rejects_every_nonpresent_identity(before, after, field, status):
    data = valid_receipt()
    data["predecessor_state"], data["requested_state"] = before, after
    data[field] = {"status": status, "value": None, "reason": "Synthetic unresolved identity"}
    with pytest.raises(SchemaValidationError, match=f"PRESENT {field}"):
        AuthorizationReceipt.from_mapping(data)


@pytest.mark.parametrize("before,after", PROTECTED_TRANSITIONS)
def test_protected_transition_accepts_all_three_present(before, after):
    data = valid_receipt()
    data["predecessor_state"], data["requested_state"] = before, after
    assert AuthorizationReceipt.from_mapping(data).requested_state.value == after


@pytest.mark.parametrize("before,after", PROTECTED_TRANSITIONS)
@pytest.mark.parametrize("invalid_field", ["best_checkpoint", "population", "normalizer"])
def test_one_present_identity_cannot_hide_another_invalid_identity(before, after, invalid_field):
    data = valid_receipt()
    data["predecessor_state"], data["requested_state"] = before, after
    data[invalid_field] = {"status": "NOT_APPLICABLE", "value": None, "reason": "Attempted bypass"}
    with pytest.raises(SchemaValidationError):
        AuthorizationReceipt.from_mapping(data)


def test_nonprotected_transition_still_requires_explicit_statuses():
    data = valid_receipt()
    data["predecessor_state"], data["requested_state"] = "DEVELOPMENT", "TRAINING"
    for field in ("best_checkpoint", "population", "normalizer"):
        data[field] = {"status": "UNRESOLVED", "value": None, "reason": "Later-phase applicability"}
    assert AuthorizationReceipt.from_mapping(data).best_checkpoint.status.value == "UNRESOLVED"
