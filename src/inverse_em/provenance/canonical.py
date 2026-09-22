from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping
import hashlib
import json
import math
import unicodedata

from inverse_em.errors import SchemaValidationError

CANONICALIZATION_VERSION = "canonical-json-v1"
SUPPORTED_CANONICALIZATION_VERSIONS = frozenset({CANONICALIZATION_VERSION})


def validate_canonicalization_version(value: Any) -> str:
    if type(value) is not str or value not in SUPPORTED_CANONICALIZATION_VERSIONS:
        raise SchemaValidationError(f"Unsupported canonicalization version: {value!r}")
    return value


def normalize_identity_text(value: str) -> str:
    if type(value) is not str:
        raise SchemaValidationError("Identity text must be a string")
    return unicodedata.normalize("NFC", value)


def validate_logical_path(value: Any) -> str:
    """Validate a portable repository-relative POSIX path."""
    path = normalize_identity_text(value)
    if not path or "\\" in path or PureWindowsPath(path).drive or PureWindowsPath(path).is_absolute() or PurePosixPath(path).is_absolute():
        raise SchemaValidationError(f"Logical path must be relative POSIX syntax: {value!r}")
    if any(part in {"", ".", ".."} for part in path.split("/")):
        raise SchemaValidationError(f"Logical path contains an invalid segment: {value!r}")
    return path


def _float_text(value: float) -> str:
    """Encode finite binary64 using 17 significant decimal digits."""
    if not math.isfinite(value):
        raise SchemaValidationError("Canonical data cannot contain NaN or Infinity")
    if value == 0.0:
        return "0.0"
    rendered = format(value, ".17g")
    if "e" in rendered or "E" in rendered:
        mantissa, exponent = rendered.lower().split("e")
        return f"{mantissa}e{int(exponent)}"
    return rendered if "." in rendered else rendered + ".0"


def _encode(value: Any) -> str:
    if is_dataclass(value):
        return _encode(asdict(value))
    if isinstance(value, Enum):
        return _encode(value.value)
    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if type(value) is float:
        return _float_text(value)
    if type(value) is str:
        return json.dumps(normalize_identity_text(value), ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise SchemaValidationError("Canonical mapping keys must be strings")
            normalized_key = normalize_identity_text(key)
            if normalized_key in normalized:
                raise SchemaValidationError("Canonical mapping keys collide after Unicode NFC normalization")
            normalized[normalized_key] = item
        return "{" + ",".join(f"{_encode(key)}:{_encode(normalized[key])}" for key in sorted(normalized)) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_encode(item) for item in value) + "]"
    raise SchemaValidationError(f"Unsupported canonical type: {type(value).__name__}")


def canonical_bytes(value: Any, version: str = CANONICALIZATION_VERSION) -> bytes:
    scheme = validate_canonicalization_version(version)
    return _encode({"canonicalization": scheme, "payload": value}).encode("utf-8")


def canonical_sha256(value: Any, version: str = CANONICALIZATION_VERSION) -> str:
    return hashlib.sha256(canonical_bytes(value, version)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
