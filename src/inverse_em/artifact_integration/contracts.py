"""Immutable, use-specific contracts. No model construction or artifact I/O."""
from dataclasses import asdict, dataclass
from enum import Enum
import math
import re
import json

from inverse_em.provenance.canonical import canonical_sha256


class Use(str, Enum):
    INFERENCE = "INFERENCE"
    CONTINUATION = "CONTINUATION"
    PROVENANCE_ONLY = "PROVENANCE_ONLY"


class Outcome(str, Enum):
    DIRECTLY_COMPATIBLE = "DIRECTLY_COMPATIBLE"
    CONTROLLED_MIGRATION_REQUIRED = "CONTROLLED_MIGRATION_REQUIRED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    REGENERATION_REQUIRED = "REGENERATION_REQUIRED"
    RETRAINING_REQUIRED = "RETRAINING_REQUIRED"


class Domain(str, Enum):
    SOURCE = "historical_source_file"
    STATE = "tensor_state_content"
    CONFIG = "scientific_configuration"
    NATIVE = "repository_native_artifact"


class Family(str, Enum):
    E = "E"
    H = "H"
    CLASSIFIER = "CLASSIFIER"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    NORMALIZER = "NORMALIZER"


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Identity:
    domain: Domain
    schema: str
    digest: str

    def __post_init__(self):
        require(type(self.domain) is Domain, "Identity namespace required")
        require(type(self.schema) is str and bool(self.schema), "Identity schema required")
        require(type(self.digest) is str and re.fullmatch("[0-9a-f]{64}", self.digest), "Invalid digest")


def checked(value, domain):
    require(type(value) is Identity, "Typed identity required")
    value.__post_init__()
    require(value.domain is domain, "Identity domain substitution")
    return value


def identity(domain, schema, value):
    return Identity(domain, schema, canonical_sha256(value))


@dataclass(frozen=True)
class Transformation:
    name: str
    version: int
    semantics: tuple

    def __post_init__(self):
        require(type(self.name) is str and type(self.version) is int and type(self.semantics) is tuple
                and all(type(v) is str for v in self.semantics), "Immutable transformation descriptor")


@dataclass(frozen=True)
class Mapping:
    source: Identity
    destination: Identity
    evidence: Identity
    transformation: Transformation | None = None

    def __post_init__(self):
        require(type(self.source) is Identity and type(self.destination) is Identity, "Typed endpoints required")
        self.source.__post_init__()
        self.destination.__post_init__()
        require(self.source.domain is not self.destination.domain, "Cross-domain mapping required")
        checked(self.evidence, Domain.NATIVE)
        if self.transformation is not None:
            require(type(self.transformation) is Transformation, "Typed transformation")
            self.transformation.__post_init__()


class InsufficientEvidence(ValueError):
    """A descriptor is not a resolved scientific compatibility assertion."""


_TRANSFORMATIONS = {
    "explicit_statistics": (Domain.NATIVE, "explicit-synthetic-normalizer-source/1", Domain.STATE,
        "exact-normalizer-statistics/1", ("explicit_clean_training_statistics", "no_fitting", "exact_values")),
    "tensor_observation": (Domain.SOURCE, "synthetic-EH-source/1", Domain.STATE,
        "ordered-f64le-tensors/1", ("exact_ordered_tensor_bytes", "field_and_configuration_bound")),
    "validated_container": (Domain.SOURCE, "synthetic-source-bytes/1", Domain.NATIVE,
        "validated-synthetic-document/1", ("validated_document", "context_and_use_bound")),
    "identity_copy": (Domain.SOURCE, "synthetic-source-bytes/1", Domain.NATIVE,
        "synthetic-byte-copy/1", ("identity_no_op", "exact_bytes", "no_semantic_change")),
}


def _evidence_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


class EvidenceRegistry:
    """Evidence produced from observed tiny fixtures, not caller digest assertions.

    No public digest/record registration API. Source registration in SyntheticStore
    does not mint evidence. This is not an authentication sandbox for hostile code
    able to mutate private Python objects.
    """
    def __init__(self):
        self._records = {}

    def training_provenance(self, task, count, stages):
        """Establish a tiny explicit fixture descriptor, not generate observations."""
        require(task in ("CLASSIFIER", "S1", "S2", "S3") and type(count) is int and 0 < count <= 16,
                "Explicit tiny training fixture only")
        require(type(stages) is tuple and stages == (tuple(range(1, 9)) if task == "S3" else ()), "Training stage provenance")
        record = {"schema": "synthetic-clean-training-provenance/1", "role": "TRAINING", "task": task,
                  "observation_count": count, "stages": stages, "clean": True,
                  "origin": "explicit_synthetic_fixture", "configuration": asdict(configuration(Family.NORMALIZER))}
        result = identity(Domain.NATIVE, record["schema"], record)
        self._records[result.digest] = _evidence_bytes(record)
        return result

    def _issue(self, operation, source, destination, facts):
        spec = _TRANSFORMATIONS[operation]
        require((source.domain, source.schema, destination.domain, destination.schema) == spec[:4], "Mapping endpoints")
        transformation = Transformation(operation, 1, spec[4])
        record = {"schema": "resolved-synthetic-mapping-evidence/1", "source": asdict(source),
                  "destination": asdict(destination), "transformation": asdict(transformation), "facts": facts}
        proof = identity(Domain.NATIVE, record["schema"], record)
        self._records[proof.digest] = _evidence_bytes(record)
        return Mapping(source, destination, proof, transformation)

    def resolve(self, mapping, operation, facts):
        require(type(mapping) is Mapping, "Typed mapping required")
        mapping.__post_init__()
        spec = _TRANSFORMATIONS.get(operation)
        if (spec is None or mapping.transformation != Transformation(operation, 1, spec[4])
                or (mapping.source.domain, mapping.source.schema, mapping.destination.domain, mapping.destination.schema) != spec[:4]
                or mapping.evidence.schema != "resolved-synthetic-mapping-evidence/1"
                or mapping.evidence.digest == "0" * 64 or mapping.evidence in (mapping.source, mapping.destination)):
            raise InsufficientEvidence("Unsupported/unresolved transformation or evidence schema")
        record = {"schema": "resolved-synthetic-mapping-evidence/1", "source": asdict(mapping.source),
                  "destination": asdict(mapping.destination), "transformation": asdict(mapping.transformation), "facts": facts}
        if (mapping.evidence != identity(Domain.NATIVE, record["schema"], record)
                or self._records.get(mapping.evidence.digest) != _evidence_bytes(record)):
            raise InsufficientEvidence("Mapping evidence does not resolve to observed facts")
        return mapping.evidence

    def normalizer(self, task, mean, std, count, stages, training_identity):
        checked(training_identity, Domain.NATIVE)
        training_record = {"schema": "synthetic-clean-training-provenance/1", "role": "TRAINING", "task": task,
            "observation_count": count, "stages": stages, "clean": True,
            "origin": "explicit_synthetic_fixture", "configuration": asdict(configuration(Family.NORMALIZER))}
        require(training_identity == identity(Domain.NATIVE, training_record["schema"], training_record)
                and self._records.get(training_identity.digest) == _evidence_bytes(training_record),
                "Independently established supported training provenance required")
        stats = (task, mean, std, count, count * 30, stages,
                 ("Re(E_z)", "Im(E_z)", "Re(H_phi)", "Im(H_phi)"), "float64", 0)
        state = identity(Domain.STATE, "exact-normalizer-statistics/1", stats)
        facts = {"origin": "explicit-synthetic-clean-training-statistics", "task": task,
                 "configuration": asdict(configuration(Family.NORMALIZER)), "statistics": stats,
                 "training_population": asdict(training_identity)}
        source = identity(Domain.NATIVE, "explicit-synthetic-normalizer-source/1", facts)
        # Validate all statistics before any evidence can be issued.
        provisional = Normalizer(task, mean, std, count, count * 30, stages, source,
                                 Mapping(source, state, source))
        mapping = self._issue("explicit_statistics", source, state, facts)
        return Normalizer(task, provisional.mean, provisional.std, count, count * 30, stages, source, mapping)

    def lineage(self, field, tensors):
        from .validation import validate_tensors
        require(field in (Family.E, Family.H), "E/H only")
        state = validate_tensors(tensors, tensor_schema(field))
        source = identity(Domain.SOURCE, "synthetic-EH-source/1", tensors)
        facts = {"field": field.value, "configuration": asdict(configuration(field)),
                 "representation": "relative_angle_B", "state": asdict(state)}
        return Lineage(field, source, state, configuration(field), self._issue("tensor_observation", source, state, facts))

    def _publication(self, source, destination, context_sha256, use):
        facts = {"context_sha256": context_sha256, "use": use.value}
        mapping = self._issue("validated_container", source, destination, facts)
        self.resolve(mapping, "validated_container", facts)
        return mapping

    def identity_copy(self, raw):
        import hashlib
        require(type(raw) is bytes and len(raw) <= 4096, "Tiny explicit no-op fixture only")
        digest = hashlib.sha256(raw).hexdigest()
        facts = {"size": len(raw), "sha256": digest}
        mapping = self._issue("identity_copy", Identity(Domain.SOURCE, "synthetic-source-bytes/1", digest),
                              Identity(Domain.NATIVE, "synthetic-byte-copy/1", digest), facts)
        return mapping, facts


@dataclass(frozen=True)
class Runtime:
    python: str
    numpy: str
    torch: str
    device: str = "cpu"
    dtype: str = "float64"

    def __post_init__(self):
        require(all(type(x) is str and x for x in (self.python, self.numpy, self.torch)), "Missing runtime")
        require((self.device, self.dtype) == ("cpu", "float64"), "Unsupported runtime policy")


def current_runtime():
    from importlib.metadata import version
    import platform
    return Runtime(platform.python_version(), version("numpy"), version("torch"))


def configuration(family):
    # Lazy imports of closed, pure configuration descriptors only.
    from inverse_em.config.surrogate import SurrogateScientificConfig
    from inverse_em.config.classifier import ClassifierScientificConfig
    from inverse_em.config.s1 import S1Config
    from inverse_em.config.s2 import S2Config
    from inverse_em.config.s3 import S3Config
    from inverse_em.config.phase2 import Phase2ScientificConfig
    cls = {Family.E: SurrogateScientificConfig, Family.H: SurrogateScientificConfig,
           Family.CLASSIFIER: ClassifierScientificConfig, Family.S1: S1Config,
           Family.S2: S2Config, Family.S3: S3Config,
           Family.NORMALIZER: Phase2ScientificConfig}[family]
    return Identity(Domain.CONFIG, "closed-scientific-configuration/1", cls().sha256)


def tensor_schema(family):
    """Literal state schemas checked against closed models by synthetic tests."""
    if family is Family.NORMALIZER:
        return (("mean", (4,)), ("std", (4,)))
    if family in (Family.E, Family.H):
        layers = tuple((f"net.{2*i}", (out, inp)) for i, (inp, out) in
                       enumerate(zip((3, 128, 128, 128, 128), (128, 128, 128, 128, 2))))
    elif family is Family.CLASSIFIER:
        layers = (("conv", (64, 4, 3)),) + tuple(
            (f"blocks.{i}.conv{j}", (64, 64, 3)) for i in range(4) for j in (1, 2)) + (("fc", (5, 64)),)
    else:
        require(family in (Family.S1, Family.S2, Family.S3), "Unknown family")
        layers = (("features.0", (32, 4, 5)), ("features.2", (64, 32, 5)),
                  ("features.4", (96, 64, 3)), ("dense.1", (128, 2880)),
                  ("radius_head", (3, 128)), ("angle_head", (6, 128)))
    return tuple(item for name, shape in layers for item in
                 ((name + ".weight", shape), (name + ".bias", (shape[0],))))


def architecture(family):
    return identity(Domain.CONFIG, "closed-state-schema/1", (family, configuration(family),
                    tensor_schema(family), "float64", "cpu", "no_buffers"))


def scientific_dependency(family, name):
    if name == "physics":
        from inverse_em.config.surrogate import SurrogateScientificConfig
        c = SurrogateScientificConfig()
        return identity(Domain.CONFIG, "physics-source-contract/1", (c.physics_revision, c.physics_source_sha256))
    require(name == "rng_contract", "Unknown scientific dependency")
    # Bind the closed configuration including its RNG convention and seed-role
    # descriptors. Merely describing seeds does not exercise any RNG stream.
    return identity(Domain.CONFIG, "closed-rng-contract/1", (family, configuration(family)))


def active_slots(family):
    return {Family.S1: (0,), Family.S2: (0, 1), Family.S3: (0, 1, 2)}.get(family, ())


def boundary(family, role):
    if family is Family.S3:
        return ("S3_V1_VALIDATION_COMPLETE_PRE_SCHEDULER" if role == "BEST" else
                "S3_V1_EPOCH_VALIDATION_BEST_HISTORY_SCHEDULER_COMPLETE")
    if family is Family.S2 and role == "BEST":
        return "VALIDATION_AND_ACCOUNTING_COMPLETE_PRE_SCHEDULER"
    if family in (Family.S1, Family.S2):
        return "EPOCH_COMPLETE_VALIDATION_AND_SCHEDULER_COMPLETE"
    return "SCHEMA_BOUND_SNAPSHOT"  # Phase-3/4 boundary is bound in the continuation descriptor.


@dataclass(frozen=True)
class Normalizer:
    task: str
    mean: tuple
    std: tuple
    observation_count: int
    scalar_count: int
    stages: tuple
    provenance: Identity
    mapping: Mapping
    channels: tuple = ("Re(E_z)", "Im(E_z)", "Re(H_phi)", "Im(H_phi)")
    dtype: str = "float64"
    ddof: int = 0

    @property
    def state(self):
        return identity(Domain.STATE, "exact-normalizer-statistics/1", (self.task, self.mean, self.std,
                        self.observation_count, self.scalar_count, self.stages, self.channels, self.dtype, self.ddof))

    def __post_init__(self):
        require(self.task in ("CLASSIFIER", "S1", "S2", "S3"), "Normalizer task")
        require(type(self.mean) is tuple and type(self.std) is tuple and len(self.mean) == len(self.std) == 4,
                "Normalizer shape")
        require(all(type(v) is float and math.isfinite(v) for v in self.mean + self.std), "Normalizer finite float64 statistics")
        require(all(v > 0 for v in self.std), "Normalizer positive std")
        require(type(self.observation_count) is int and 0 < self.observation_count <= 16,
                "Only tiny explicit synthetic statistics authorized")
        require(type(self.scalar_count) is int and self.scalar_count == 30 * self.observation_count, "Normalizer counts")
        require(type(self.stages) is tuple and all(type(v) is int for v in self.stages)
                and self.stages == (tuple(range(1, 9)) if self.task == "S3" else ()), "Normalizer stages")
        require(self.channels == ("Re(E_z)", "Im(E_z)", "Re(H_phi)", "Im(H_phi)") and type(self.channels) is tuple,
                "Normalizer channel order")
        require(self.dtype == "float64" and type(self.ddof) is int and self.ddof == 0, "Normalizer dtype/ddof")
        checked(self.provenance, Domain.NATIVE)
        require(type(self.mapping) is Mapping, "Normalizer mapping required")
        self.mapping.__post_init__()
        require(self.mapping.source == self.provenance and self.mapping.destination == self.state, "Normalizer identity mapping")


@dataclass(frozen=True)
class Lineage:
    field: Family
    source: Identity
    state: Identity
    configuration: Identity
    mapping: Mapping
    representation: str = "relative_angle_B"

    def __post_init__(self):
        require(self.field in (Family.E, Family.H) and type(self.field) is Family, "E/H lineage required")
        checked(self.source, Domain.SOURCE)
        checked(self.state, Domain.STATE)
        checked(self.configuration, Domain.CONFIG)
        require(self.configuration == configuration(self.field) and self.representation == "relative_angle_B", "Lineage science")
        require(type(self.mapping) is Mapping, "Lineage mapping required")
        self.mapping.__post_init__()
        require((self.mapping.source, self.mapping.destination) == (self.source, self.state), "Lineage endpoints")


@dataclass(frozen=True)
class Context:
    family: Family
    use: Use
    task: str
    role: str
    configuration: Identity
    architecture: Identity
    expected_state: Identity
    runtime: Runtime
    dependencies: tuple
    normalizer: Normalizer | None
    lineage: tuple
    boundary: str
    continuation_components: tuple = ()
    scope: str = "SYNTHETIC_ONLY"

    def __post_init__(self):
        require(type(self.family) is Family and type(self.use) is Use and self.scope == "SYNTHETIC_ONLY", "Context type/scope")
        checked(self.configuration, Domain.CONFIG)
        checked(self.architecture, Domain.CONFIG)
        checked(self.expected_state, Domain.STATE)
        require(self.configuration == configuration(self.family) and self.architecture == architecture(self.family), "Frozen configuration/schema")
        require(type(self.runtime) is Runtime, "Runtime context required")
        self.runtime.__post_init__()
        require(type(self.dependencies) is tuple and all(type(p) is tuple and len(p) == 2 for p in self.dependencies)
                and tuple(k for k, _ in self.dependencies) ==
                ("physics", "training_population", "validation_population", "rng_contract", "provenance"), "Incomplete dependencies")
        for key, value in self.dependencies:
            checked(value, Domain.CONFIG if key in ("physics", "rng_contract") else Domain.NATIVE)
            if key in ("physics", "rng_contract"):
                require(value == scientific_dependency(self.family, key), "Wrong physics/RNG contract")
        require(self.task == self.family.value or (self.family is Family.NORMALIZER and self.task in
                ("CLASSIFIER", "S1", "S2", "S3")), "Wrong task")
        if self.family in (Family.E, Family.H):
            require(self.normalizer is None, "Surrogate has no input normalizer")
        else:
            require(type(self.normalizer) is Normalizer, "Exact normalizer required")
            self.normalizer.__post_init__()
            require(self.normalizer.task == self.task, "Wrong normalizer task")
        require(type(self.lineage) is tuple, "Immutable lineage required")
        if self.family in (Family.S1, Family.S2, Family.S3):
            require(len(self.lineage) == 2 and tuple(x.field for x in self.lineage) == (Family.E, Family.H), "E/H pair required")
            for item in self.lineage:
                require(type(item) is Lineage, "Typed lineage required")
                item.__post_init__()
        else:
            require(not self.lineage, "Unexpected lineage")
        inference = "best_validation" if self.family is Family.CLASSIFIER else "NORMALIZER" if self.family is Family.NORMALIZER else "BEST"
        continuation = "resumable" if self.family in (Family.E, Family.H, Family.CLASSIFIER) else "CONTINUATION"
        roles = (inference,) if self.use is Use.INFERENCE else (continuation,) if self.use is Use.CONTINUATION else (inference, continuation, "TERMINAL")
        require(self.role in roles and self.boundary == boundary(self.family, self.role), "Role/boundary mismatch")
        require(type(self.continuation_components) is tuple and all(type(p) is tuple and len(p) == 2
                for p in self.continuation_components), "Immutable continuation bindings")
        if self.use is Use.CONTINUATION:
            require(self.family is not Family.NORMALIZER, "Normalizer has no continuation use")
            require(tuple(k for k, _ in self.continuation_components) == continuation_keys(self.family), "Incomplete continuation context")
            for _, value in self.continuation_components:
                checked(value, Domain.STATE)
        else:
            require(not self.continuation_components, "Unexpected continuation bindings")

    @property
    def sha256(self):
        self.__post_init__()
        return canonical_sha256(self)


def continuation_keys(family):
    common = ("optimizer", "scheduler", "rng", "history", "selection", "best_snapshot", "counters", "boundary_state")
    return common + (("kendall", "stage", "early_stop") if family in (Family.S1, Family.S2, Family.S3) else ("early_stop",))
