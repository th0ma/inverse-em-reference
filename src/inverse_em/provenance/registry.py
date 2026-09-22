from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Mapping

from inverse_em.errors import SchemaValidationError
from inverse_em.scientific.roles import ScientificRole

ARTIFACT_REGISTRY_SCHEMA_VERSION = "artifact-registry/1.0"
SUPPORTED_ARTIFACT_REGISTRY_VERSIONS = frozenset({ARTIFACT_REGISTRY_SCHEMA_VERSION})


class StorageClass(str, Enum):
    GIT = "GIT"
    RELEASE = "RELEASE"
    GIT_LFS = "GIT_LFS"


class EvidenceClass(str, Enum):
    AUTHORITATIVE = "AUTHORITATIVE"
    REFERENCE = "REFERENCE"
    DEVELOPMENT = "DEVELOPMENT"


def _sha(value: str, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise SchemaValidationError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class ArtifactRecord:
    logical_id: str
    scientific_role: ScientificRole
    study_id: str
    sha256: str
    byte_size: int
    format: str
    storage_class: StorageClass
    evidence_class: EvidenceClass
    provenance_manifest_sha256: str
    schema_version: str | None = None
    release_id: str | None = None
    location: str | None = None

    def __post_init__(self) -> None:
        if any(type(value) is not str or not value.strip() for value in (self.logical_id, self.study_id, self.format)):
            raise SchemaValidationError("Artifact identity fields must be non-empty")
        for name in ("schema_version", "release_id", "location"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value.strip()):
                raise SchemaValidationError(f"{name} must be null or a non-empty string")
        _sha(self.sha256, "sha256")
        _sha(self.provenance_manifest_sha256, "provenance_manifest_sha256")
        if type(self.byte_size) is not int or self.byte_size < 0:
            raise SchemaValidationError("byte_size must be a non-negative integer")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ArtifactRecord":
        allowed = {item.name for item in fields(cls)}
        unknown = set(data) - allowed
        required = allowed - {"schema_version", "release_id", "location"}
        missing = required - set(data)
        if unknown or missing:
            raise SchemaValidationError(f"Artifact fields invalid; unknown={sorted(unknown)}, missing={sorted(missing)}")
        values = dict(data)
        values["scientific_role"] = ScientificRole(values["scientific_role"])
        values["storage_class"] = StorageClass(values["storage_class"])
        values["evidence_class"] = EvidenceClass(values["evidence_class"])
        return cls(**values)


@dataclass(frozen=True)
class ArtifactRegistry:
    schema_version: str
    artifacts: tuple[ArtifactRecord, ...]

    def __post_init__(self) -> None:
        ids = [record.logical_id for record in self.artifacts]
        if self.schema_version not in SUPPORTED_ARTIFACT_REGISTRY_VERSIONS:
            raise SchemaValidationError(f"Unsupported ArtifactRegistry schema version: {self.schema_version!r}")
        if len(ids) != len(set(ids)):
            raise SchemaValidationError("Registry logical IDs must be unique")
        by_digest: dict[str, tuple[int, str, str | None]] = {}
        for record in self.artifacts:
            metadata = (record.byte_size, record.format, record.schema_version)
            if record.sha256 in by_digest and by_digest[record.sha256] != metadata:
                raise SchemaValidationError(f"Conflicting byte-level metadata for digest {record.sha256}")
            by_digest[record.sha256] = metadata

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ArtifactRegistry":
        if set(data) != {"schema_version", "artifacts"}:
            raise SchemaValidationError("Registry requires exactly schema_version and artifacts")
        if not isinstance(data["artifacts"], list):
            raise SchemaValidationError("artifacts must be a list")
        if type(data["schema_version"]) is not str or data["schema_version"] not in SUPPORTED_ARTIFACT_REGISTRY_VERSIONS:
            raise SchemaValidationError(f"Unsupported ArtifactRegistry schema version: {data['schema_version']!r}")
        return cls(data["schema_version"], tuple(ArtifactRecord.from_mapping(item) for item in data["artifacts"]))
