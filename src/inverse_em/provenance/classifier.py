from __future__ import annotations

from dataclasses import dataclass

from inverse_em.provenance.canonical import canonical_sha256


@dataclass(frozen=True)
class ClassifierExecutionReceipt:
    configuration_sha256: str
    training_population_identity: str
    validation_population_identity: str
    normalizer_sha256: str
    physics_identity: str
    repository_revision: str
    torch_version: str
    epoch_count: int
    update_count: int
    selection_rule: str = "strict clean-validation macro-F1 increase; exact tie retains earliest"

    @property
    def sha256(self): return canonical_sha256(self)


@dataclass(frozen=True)
class ClassifierCheckpointReceipt:
    logical_path: str
    artifact_type: str
    file_sha256: str
    epoch: int
    update_count: int
    model_state_sha256: str

    @property
    def sha256(self): return canonical_sha256(self)
