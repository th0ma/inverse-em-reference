from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, Mapping, TypeVar

from inverse_em.errors import SchemaValidationError
from inverse_em.provenance.canonical import normalize_identity_text, validate_logical_path
from inverse_em.scientific.roles import ScientificRole
from inverse_em.scientific.state import ScientificState, validate_transition

RUN_MANIFEST_SCHEMA_VERSION = "scientific-run-manifest/1.0"
SUPPORTED_RUN_MANIFEST_VERSIONS = frozenset({RUN_MANIFEST_SCHEMA_VERSION})
T = TypeVar("T")


class ValueStatus(str, Enum):
    PRESENT = "PRESENT"
    MISSING = "MISSING"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNRESOLVED = "UNRESOLVED"


def _strict(data: Mapping[str, Any], cls: type, required: set[str] | None = None) -> None:
    allowed = {item.name for item in fields(cls)}
    required = allowed if required is None else required
    unknown, missing = set(data) - allowed, required - set(data)
    if unknown or missing:
        raise SchemaValidationError(f"{cls.__name__} fields invalid; unknown={sorted(unknown)}, missing={sorted(missing)}")


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise SchemaValidationError(f"{name} must be a non-empty string")
    return normalize_identity_text(value)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SchemaValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _sha(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise SchemaValidationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _timestamp(value: Any, name: str) -> str:
    raw = _text(value, name)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaValidationError(f"{name} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise SchemaValidationError(f"{name} must include a timezone")
    return raw


@dataclass(frozen=True)
class QualifiedValue(Generic[T]):
    status: ValueStatus
    value: T | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status is ValueStatus.PRESENT:
            if self.value is None or self.reason is not None:
                raise SchemaValidationError("PRESENT requires a value and no reason")
        elif self.value is not None or type(self.reason) is not str or not self.reason.strip():
            raise SchemaValidationError(f"{self.status.value} requires no value and a non-empty reason")

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self.value) if self.value is not None and hasattr(self.value, "__dataclass_fields__") else self.value
        if isinstance(value, tuple):
            value = [asdict(item) if hasattr(item, "__dataclass_fields__") else item for item in value]
        return {"status": self.status.value, "value": value, "reason": self.reason}


def qualified_from_mapping(data: Any, parser: Callable[[Any], T]) -> QualifiedValue[T]:
    if not isinstance(data, Mapping) or set(data) != {"status", "value", "reason"}:
        raise SchemaValidationError("QualifiedValue requires exactly status, value, and reason")
    try:
        status = ValueStatus(data["status"])
    except (ValueError, TypeError) as exc:
        raise SchemaValidationError(f"Invalid qualified status: {data['status']!r}") from exc
    value = parser(data["value"]) if status is ValueStatus.PRESENT else data["value"]
    return QualifiedValue(status, value, data["reason"])


@dataclass(frozen=True)
class RepositoryIdentity:
    commit: str
    dirty: bool
    git_status_sha256: str

    def __post_init__(self) -> None:
        _text(self.commit, "commit")
        if type(self.dirty) is not bool:
            raise SchemaValidationError("dirty must be boolean")
        _sha(self.git_status_sha256, "git_status_sha256")

    @classmethod
    def from_mapping(cls, data: Any) -> "RepositoryIdentity":
        if not isinstance(data, Mapping): raise SchemaValidationError("repository must be a mapping")
        _strict(data, cls); return cls(**data)


@dataclass(frozen=True)
class ConfigurationIdentity:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", validate_logical_path(self.path)); _sha(self.sha256, "configuration.sha256")

    @classmethod
    def from_mapping(cls, data: Any) -> "ConfigurationIdentity":
        if not isinstance(data, Mapping): raise SchemaValidationError("configuration must be a mapping")
        _strict(data, cls); return cls(**data)


@dataclass(frozen=True)
class PhysicsIdentity:
    revision: str
    sha256: str

    def __post_init__(self) -> None:
        _text(self.revision, "physics.revision"); _sha(self.sha256, "physics.sha256")

    @classmethod
    def from_mapping(cls, data: Any) -> "PhysicsIdentity":
        if not isinstance(data, Mapping): raise SchemaValidationError("physics value must be a mapping")
        _strict(data, cls); return cls(**data)


@dataclass(frozen=True)
class PopulationRecord:
    population_id: str
    scientific_role: ScientificRole
    sha256: str
    sample_count: int
    seed: int

    def __post_init__(self) -> None:
        _text(self.population_id, "population_id"); _sha(self.sha256, "population.sha256")
        _integer(self.sample_count, "sample_count", 1); _integer(self.seed, "seed")

    @classmethod
    def from_mapping(cls, data: Any) -> "PopulationRecord":
        if not isinstance(data, Mapping): raise SchemaValidationError("population must be a mapping")
        _strict(data, cls)
        try: role = ScientificRole(data["scientific_role"])
        except (ValueError, TypeError) as exc: raise SchemaValidationError("Invalid population role") from exc
        return cls(data["population_id"], role, data["sha256"], data["sample_count"], data["seed"])


@dataclass(frozen=True)
class NormalizerIdentity:
    sha256: str
    schema_version: str

    def __post_init__(self) -> None:
        _sha(self.sha256, "normalizer.sha256"); _text(self.schema_version, "normalizer.schema_version")

    @classmethod
    def from_mapping(cls, data: Any) -> "NormalizerIdentity":
        if not isinstance(data, Mapping): raise SchemaValidationError("normalizer value must be a mapping")
        _strict(data, cls); return cls(**data)


@dataclass(frozen=True)
class SeedRecord:
    role: str
    value: int

    def __post_init__(self) -> None:
        _text(self.role, "seed.role"); _integer(self.value, "seed.value")

    @classmethod
    def from_mapping(cls, data: Any) -> "SeedRecord":
        if not isinstance(data, Mapping): raise SchemaValidationError("seed must be a mapping")
        _strict(data, cls); return cls(**data)


@dataclass(frozen=True)
class ModelIdentity:
    class_name: str
    parameter_count: int
    dtype: str
    device: str

    def __post_init__(self) -> None:
        _text(self.class_name, "model.class_name"); _integer(self.parameter_count, "parameter_count")
        _text(self.dtype, "model.dtype"); _text(self.device, "model.device")

    @classmethod
    def from_mapping(cls, data: Any) -> "ModelIdentity":
        if not isinstance(data, Mapping): raise SchemaValidationError("model value must be a mapping")
        _strict(data, cls); return cls(**data)


@dataclass(frozen=True)
class CheckpointMetadata:
    epoch: int
    update: int
    sha256: str

    def __post_init__(self) -> None:
        _integer(self.epoch, "checkpoint.epoch"); _integer(self.update, "checkpoint.update"); _sha(self.sha256, "checkpoint.sha256")

    @classmethod
    def from_mapping(cls, data: Any) -> "CheckpointMetadata":
        if not isinstance(data, Mapping): raise SchemaValidationError("checkpoint must be a mapping")
        _strict(data, cls); return cls(**data)


@dataclass(frozen=True)
class TrainingMetadata:
    epochs_completed: int
    updates_completed: int
    checkpoint_selection_criterion: str
    best: QualifiedValue[CheckpointMetadata]
    terminal: QualifiedValue[CheckpointMetadata]

    def __post_init__(self) -> None:
        _integer(self.epochs_completed, "epochs_completed"); _integer(self.updates_completed, "updates_completed")
        _text(self.checkpoint_selection_criterion, "checkpoint_selection_criterion")

    @classmethod
    def from_mapping(cls, data: Any) -> "TrainingMetadata":
        if not isinstance(data, Mapping): raise SchemaValidationError("training value must be a mapping")
        _strict(data, cls)
        return cls(data["epochs_completed"], data["updates_completed"], data["checkpoint_selection_criterion"],
                   qualified_from_mapping(data["best"], CheckpointMetadata.from_mapping),
                   qualified_from_mapping(data["terminal"], CheckpointMetadata.from_mapping))


@dataclass(frozen=True)
class ArtifactIdentity:
    logical_id: str
    sha256: str

    def __post_init__(self) -> None:
        _text(self.logical_id, "artifact.logical_id"); _sha(self.sha256, "artifact.sha256")

    @classmethod
    def from_mapping(cls, data: Any) -> "ArtifactIdentity":
        if not isinstance(data, Mapping): raise SchemaValidationError("artifact must be a mapping")
        _strict(data, cls); return cls(**data)


@dataclass(frozen=True)
class PackageVersion:
    name: str
    version: str

    def __post_init__(self) -> None: _text(self.name, "package.name"); _text(self.version, "package.version")
    @classmethod
    def from_mapping(cls, data: Any) -> "PackageVersion":
        if not isinstance(data, Mapping): raise SchemaValidationError("package must be a mapping")
        _strict(data, cls); return cls(**data)


@dataclass(frozen=True)
class EnvironmentMetadata:
    python_version: str
    platform: str
    packages: tuple[PackageVersion, ...]

    def __post_init__(self) -> None: _text(self.python_version, "python_version"); _text(self.platform, "platform")
    @classmethod
    def from_mapping(cls, data: Any) -> "EnvironmentMetadata":
        if not isinstance(data, Mapping): raise SchemaValidationError("environment value must be a mapping")
        _strict(data, cls)
        if not isinstance(data["packages"], list): raise SchemaValidationError("packages must be a list")
        return cls(data["python_version"], data["platform"], tuple(PackageVersion.from_mapping(x) for x in data["packages"]))


@dataclass(frozen=True)
class TimestampMetadata:
    started_at: str
    completed_at: str

    def __post_init__(self) -> None:
        _timestamp(self.started_at, "started_at"); _timestamp(self.completed_at, "completed_at")
    @classmethod
    def from_mapping(cls, data: Any) -> "TimestampMetadata":
        if not isinstance(data, Mapping): raise SchemaValidationError("timestamps value must be a mapping")
        _strict(data, cls); return cls(**data)


def _tuple_parser(item_parser: Callable[[Any], T]) -> Callable[[Any], tuple[T, ...]]:
    def parse(value: Any) -> tuple[T, ...]:
        if not isinstance(value, list): raise SchemaValidationError("Qualified collection value must be a list")
        return tuple(item_parser(item) for item in value)
    return parse


@dataclass(frozen=True)
class RunManifest:
    schema_version: str
    study_id: str
    scientific_role: ScientificRole
    state_before: ScientificState
    state_after: ScientificState
    repository: RepositoryIdentity
    configuration: ConfigurationIdentity
    physics: QualifiedValue[PhysicsIdentity]
    populations: QualifiedValue[tuple[PopulationRecord, ...]]
    normalizer: QualifiedValue[NormalizerIdentity]
    seeds: QualifiedValue[tuple[SeedRecord, ...]]
    model: QualifiedValue[ModelIdentity]
    training: QualifiedValue[TrainingMetadata]
    artifacts: QualifiedValue[tuple[ArtifactIdentity, ...]]
    environment: QualifiedValue[EnvironmentMetadata]
    timestamps: QualifiedValue[TimestampMetadata]

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_RUN_MANIFEST_VERSIONS:
            raise SchemaValidationError(f"Unsupported RunManifest schema version: {self.schema_version!r}")
        _text(self.study_id, "study_id"); validate_transition(self.state_before, self.state_after)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RunManifest":
        if not isinstance(data, Mapping): raise SchemaValidationError("Manifest root must be a mapping")
        _strict(data, cls)
        if type(data["schema_version"]) is not str or data["schema_version"] not in SUPPORTED_RUN_MANIFEST_VERSIONS:
            raise SchemaValidationError(f"Unsupported RunManifest schema version: {data['schema_version']!r}")
        try:
            role, before, after = ScientificRole(data["scientific_role"]), ScientificState(data["state_before"]), ScientificState(data["state_after"])
        except (ValueError, TypeError) as exc: raise SchemaValidationError("Invalid manifest role or state") from exc
        return cls(data["schema_version"], data["study_id"], role, before, after,
                   RepositoryIdentity.from_mapping(data["repository"]), ConfigurationIdentity.from_mapping(data["configuration"]),
                   qualified_from_mapping(data["physics"], PhysicsIdentity.from_mapping),
                   qualified_from_mapping(data["populations"], _tuple_parser(PopulationRecord.from_mapping)),
                   qualified_from_mapping(data["normalizer"], NormalizerIdentity.from_mapping),
                   qualified_from_mapping(data["seeds"], _tuple_parser(SeedRecord.from_mapping)),
                   qualified_from_mapping(data["model"], ModelIdentity.from_mapping),
                   qualified_from_mapping(data["training"], TrainingMetadata.from_mapping),
                   qualified_from_mapping(data["artifacts"], _tuple_parser(ArtifactIdentity.from_mapping)),
                   qualified_from_mapping(data["environment"], EnvironmentMetadata.from_mapping),
                   qualified_from_mapping(data["timestamps"], TimestampMetadata.from_mapping))
