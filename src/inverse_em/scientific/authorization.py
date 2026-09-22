from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from inverse_em.errors import SchemaValidationError
from inverse_em.provenance.canonical import canonical_sha256
from inverse_em.provenance.manifest import QualifiedValue, ValueStatus, qualified_from_mapping
from .state import ScientificState, validate_transition

AUTHORIZATION_RECEIPT_SCHEMA_VERSION = "authorization-receipt/1.0"
SUPPORTED_AUTHORIZATION_RECEIPT_VERSIONS = frozenset({AUTHORIZATION_RECEIPT_SCHEMA_VERSION})


def _digest(value: Any, name: str, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise SchemaValidationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _timestamp(value: str) -> str:
    if type(value) is not str:
        raise SchemaValidationError("issued_at must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaValidationError("issued_at must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise SchemaValidationError("issued_at must include a timezone")
    return value


@dataclass(frozen=True)
class AuthorizationReceipt:
    schema_version: str
    study_id: str
    predecessor_state: ScientificState
    requested_state: ScientificState
    configuration_sha256: str
    predecessor_receipt_sha256: str
    issuer_label: str
    issued_at: str
    best_checkpoint: QualifiedValue[str]
    population: QualifiedValue[str]
    normalizer: QualifiedValue[str]

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_AUTHORIZATION_RECEIPT_VERSIONS:
            raise SchemaValidationError(f"Unsupported AuthorizationReceipt schema version: {self.schema_version!r}")
        if type(self.study_id) is not str or type(self.issuer_label) is not str or not self.study_id.strip() or not self.issuer_label.strip():
            raise SchemaValidationError("Receipt identity fields must be non-empty")
        validate_transition(self.predecessor_state, self.requested_state)
        _digest(self.configuration_sha256, "configuration_sha256", True)
        _digest(self.predecessor_receipt_sha256, "predecessor_receipt_sha256", True)
        for name in ("best_checkpoint", "population", "normalizer"):
            qualified = getattr(self, name)
            if qualified.status is ValueStatus.PRESENT:
                _digest(qualified.value, name, True)
        if (self.predecessor_state, self.requested_state) in {
            (ScientificState.TRAINING, ScientificState.FROZEN),
            (ScientificState.FROZEN, ScientificState.SEALED_CLEAN),
            (ScientificState.SEALED_CLEAN, ScientificState.ROBUSTNESS),
        }:
            for name in ("best_checkpoint", "population", "normalizer"):
                if getattr(self, name).status is not ValueStatus.PRESENT:
                    raise SchemaValidationError(f"Protected transition requires PRESENT {name} identity")
        _timestamp(self.issued_at)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AuthorizationReceipt":
        allowed = {field.name for field in __import__("dataclasses").fields(cls)}
        unknown = set(data) - allowed
        missing = {"schema_version", "study_id", "predecessor_state", "requested_state", "configuration_sha256", "predecessor_receipt_sha256", "issuer_label", "issued_at", "best_checkpoint", "population", "normalizer"} - set(data)
        if unknown or missing:
            raise SchemaValidationError(f"Receipt fields invalid; unknown={sorted(unknown)}, missing={sorted(missing)}")
        if type(data["schema_version"]) is not str or data["schema_version"] not in SUPPORTED_AUTHORIZATION_RECEIPT_VERSIONS:
            raise SchemaValidationError(f"Unsupported AuthorizationReceipt schema version: {data['schema_version']!r}")
        values = dict(data)
        values["predecessor_state"] = ScientificState(values["predecessor_state"])
        values["requested_state"] = ScientificState(values["requested_state"])
        digest_parser = lambda value: _digest(value, "qualified identity", True)
        for name in ("best_checkpoint", "population", "normalizer"):
            values[name] = qualified_from_mapping(values[name], digest_parser)
        return cls(**values)
