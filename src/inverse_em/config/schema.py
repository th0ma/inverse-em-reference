from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Mapping
import json
import math

from inverse_em.errors import SchemaValidationError
from inverse_em.provenance.canonical import canonical_sha256, normalize_identity_text, validate_logical_path
from inverse_em.scientific.roles import ScientificRole

SCIENTIFIC_CONFIG_SCHEMA_VERSION = "scientific-config/1.0"
SUPPORTED_SCIENTIFIC_CONFIG_VERSIONS = frozenset({SCIENTIFIC_CONFIG_SCHEMA_VERSION})


def _strict_keys(data: Mapping[str, Any], allowed: set[str], required: set[str], context: str) -> None:
    unknown = set(data) - allowed
    missing = required - set(data)
    if unknown:
        raise SchemaValidationError(f"{context}: unknown fields: {sorted(unknown, key=str)}")
    if missing:
        raise SchemaValidationError(f"{context}: missing fields: {sorted(missing)}")


def _validate_schema_version(value: Any) -> str:
    if type(value) is not str or value not in SUPPORTED_SCIENTIFIC_CONFIG_VERSIONS:
        raise SchemaValidationError(f"Unsupported ScientificConfig schema version: {value!r}")
    return value


def _is_machine_absolute_path(value: str) -> bool:
    """Return whether a string is syntactically a local absolute path."""
    return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()


def _freeze(value: Any, key_context: str | None = None) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise SchemaValidationError("Configuration mapping keys must be strings at every level")
            normalized_key = normalize_identity_text(key)
            if normalized_key in frozen:
                raise SchemaValidationError("Configuration keys collide after Unicode NFC normalization")
            frozen[normalized_key] = _freeze(item, normalized_key)
        return MappingProxyType(frozen)
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze(v) for v in value)
    if type(value) is float and not math.isfinite(value):
        raise SchemaValidationError("Configuration cannot contain NaN or Infinity")
    if type(value) is str:
        normalized = normalize_identity_text(value)
        if _is_machine_absolute_path(normalized):
            raise SchemaValidationError("Portable configuration cannot contain machine-local absolute paths")
        if key_context == "path" or (key_context is not None and key_context.endswith("_path")):
            return validate_logical_path(normalized)
        return normalized
    if value is None or type(value) in {bool, int, float}:
        return value
    raise SchemaValidationError(f"Unsupported configuration value type: {type(value).__name__}")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


@dataclass(frozen=True)
class AcceptancePolicy:
    class_a: str = "exact_identity"
    class_b_rtol: float = 1e-12
    class_b_atol: float = 1e-14
    class_c: str = "unresolved"

    def __post_init__(self) -> None:
        if self.class_a != "exact_identity":
            raise SchemaValidationError("Class A must use exact identity")
        if type(self.class_b_rtol) is not float or type(self.class_b_atol) is not float:
            raise SchemaValidationError("Class B tolerances must be explicit floats")
        if not math.isfinite(self.class_b_rtol) or not math.isfinite(self.class_b_atol):
            raise SchemaValidationError("Class B tolerances must be finite")
        if self.class_b_rtol < 0 or self.class_b_atol < 0:
            raise SchemaValidationError("Tolerances must be non-negative")
        if self.class_c != "unresolved":
            raise SchemaValidationError("Class C cannot be frozen during Phase 0")


@dataclass(frozen=True)
class ScientificConfig:
    schema_version: str
    study_id: str
    scientific_role: ScientificRole
    sections: Mapping[str, Any] = field(default_factory=dict)
    acceptance: AcceptancePolicy = field(default_factory=AcceptancePolicy)

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        if type(self.study_id) is not str or not self.study_id.strip():
            raise SchemaValidationError("study_id must be a non-empty string")
        frozen = _freeze(self.sections)
        if any(value == "UNRESOLVED" for value in _walk(frozen)):
            raise SchemaValidationError("Required configuration contains unresolved scientific values")
        object.__setattr__(self, "study_id", normalize_identity_text(self.study_id))
        object.__setattr__(self, "sections", frozen)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ScientificConfig":
        allowed = {"schema_version", "study_id", "scientific_role", "sections", "acceptance"}
        _strict_keys(data, allowed, {"schema_version", "study_id", "scientific_role", "sections"}, "configuration")
        acceptance_data = data.get("acceptance", {})
        if not isinstance(acceptance_data, Mapping):
            raise SchemaValidationError("acceptance must be a mapping")
        _strict_keys(acceptance_data, {"class_a", "class_b_rtol", "class_b_atol", "class_c"}, set(), "acceptance")
        return cls(
            schema_version=_validate_schema_version(data["schema_version"]),
            study_id=_require_str(data["study_id"], "study_id"),
            scientific_role=_require_role(data["scientific_role"]),
            sections=_require_mapping(data["sections"], "sections"),
            acceptance=AcceptancePolicy(**acceptance_data),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "scientific_role": self.scientific_role.value,
            "sections": _plain(self.sections),
            "acceptance": {
                "class_a": self.acceptance.class_a,
                "class_b_rtol": self.acceptance.class_b_rtol,
                "class_b_atol": self.acceptance.class_b_atol,
                "class_c": self.acceptance.class_c,
            },
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_mapping())


def _walk(value: Any):
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def _require_str(value: Any, name: str) -> str:
    if type(value) is not str:
        raise SchemaValidationError(f"{name} must be a string")
    return value


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{name} must be a mapping")
    return value


def _require_role(value: Any) -> ScientificRole:
    raw = _require_str(value, "scientific_role")
    try:
        return ScientificRole(raw)
    except ValueError as exc:
        raise SchemaValidationError(f"scientific_role is invalid: {raw}") from exc


def load_config(path: str | Path) -> ScientificConfig:
    source = Path(path)
    suffix = source.suffix.lower()
    text = source.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(SchemaValidationError(f"Non-finite JSON number: {value}")))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise SchemaValidationError("YAML support requires the optional 'yaml' dependency") from exc
        data = yaml.safe_load(text)
    else:
        raise SchemaValidationError("Configuration must be JSON or YAML")
    if not isinstance(data, Mapping):
        raise SchemaValidationError("Configuration root must be a mapping")
    return ScientificConfig.from_mapping(data)
