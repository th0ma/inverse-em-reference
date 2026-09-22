import math
import hashlib
import pytest

from inverse_em.errors import SchemaValidationError
from inverse_em.provenance.canonical import canonical_bytes, canonical_sha256


def test_mapping_order_does_not_change_identity():
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})


def test_scientifically_different_values_change_identity():
    assert canonical_sha256({"value": 1}) != canonical_sha256({"value": 1.0})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_values_rejected(value):
    with pytest.raises(SchemaValidationError):
        canonical_bytes({"value": value})


def test_non_string_mapping_key_rejected():
    with pytest.raises(SchemaValidationError):
        canonical_bytes({1: "value"})


def test_typed_scalars_remain_distinct():
    assert len({canonical_sha256(1), canonical_sha256(1.0), canonical_sha256(True)}) == 3


def test_signed_zero_is_normalized():
    assert canonical_bytes(0.0) == canonical_bytes(-0.0)
    assert canonical_sha256(0.0) == canonical_sha256(-0.0)


@pytest.mark.parametrize("value,expected", [
    (1.0, b'{"canonicalization":"canonical-json-v1","payload":1.0}'),
    (0.1, b'{"canonicalization":"canonical-json-v1","payload":0.10000000000000001}'),
    (1e20, b'{"canonicalization":"canonical-json-v1","payload":1e20}'),
    (1e-7, b'{"canonicalization":"canonical-json-v1","payload":9.9999999999999995e-8}'),
])
def test_fixed_float_vectors(value, expected):
    assert canonical_bytes(value) == expected


def test_unicode_nfc_normalization_and_collision_rejection():
    composed, decomposed = "é", "e\u0301"
    assert canonical_bytes(composed) == canonical_bytes(decomposed)
    with pytest.raises(SchemaValidationError, match="collide"):
        canonical_bytes({composed: 1, decomposed: 2})


@pytest.mark.parametrize("version", ["canonical-json-v2", "", 1, None])
def test_unsupported_canonicalization_version_rejected(version):
    with pytest.raises(SchemaValidationError):
        canonical_bytes({}, version)


def test_independent_sha256_known_vector():
    assert hashlib.sha256(b"abc").hexdigest() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
