from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from inverse_em.errors import SchemaValidationError
from inverse_em.provenance.canonical import canonical_sha256


class InverseTask(str, Enum):
    CLASSIFIER = "classifier"
    S1 = "s1"
    S2 = "s2"
    S3 = "s3"


CHANNEL_ORDER = ("Re(E_z)", "Im(E_z)", "Re(H_phi)", "Im(H_phi)")
TRAINING_SNR_INTERVAL_DB = (30.0, 40.0)
ROBUSTNESS_SNR_DB = (40.0, 30.0, 20.0, 15.0, 10.0)
ROBUSTNESS_REALIZATIONS = 10
NORMALIZATION_DDOF = 0


@dataclass(frozen=True)
class TaskNoiseSeeds:
    task: InverseTask
    augmentation_seed: int
    robustness_seed: int

    def __post_init__(self):
        if type(self.augmentation_seed) is not int or type(self.robustness_seed) is not int:
            raise SchemaValidationError("Noise seeds must be integers")
        if self.augmentation_seed < 0 or self.robustness_seed < 0:
            raise SchemaValidationError("Noise seeds must be nonnegative")


TASK_NOISE_SEEDS = (
    TaskNoiseSeeds(InverseTask.CLASSIFIER, 20261905, 20261907),
    TaskNoiseSeeds(InverseTask.S1, 20261815, 20261817),
    TaskNoiseSeeds(InverseTask.S2, 20262005, 20262007),
    TaskNoiseSeeds(InverseTask.S3, 20262205, 20262207),
)


def noise_seeds_for(task: InverseTask) -> TaskNoiseSeeds:
    matches = tuple(item for item in TASK_NOISE_SEEDS if item.task is task)
    if len(matches) != 1:
        raise SchemaValidationError("Unknown inverse task")
    return matches[0]


@dataclass(frozen=True)
class Phase2ScientificConfig:
    specification_edition: str = "Phase 2 v1"
    noise_model_identity: str = "raw-complex-gaussian-v1"
    rng_algorithm: str = "NumPy PCG64DXSM"
    channel_order: tuple[str, ...] = CHANNEL_ORDER
    training_snr_interval_db: tuple[float, float] = TRAINING_SNR_INTERVAL_DB
    robustness_snr_db: tuple[float, ...] = ROBUSTNESS_SNR_DB
    robustness_realizations: int = ROBUSTNESS_REALIZATIONS
    paired_across_snr: bool = True
    validation_is_clean: bool = True
    normalization_ddof: int = NORMALIZATION_DDOF
    task_noise_seeds: tuple[TaskNoiseSeeds, ...] = TASK_NOISE_SEEDS

    def __post_init__(self):
        if self.channel_order != CHANNEL_ORDER:
            raise SchemaValidationError("Channel order differs from the frozen contract")
        if self.training_snr_interval_db != TRAINING_SNR_INTERVAL_DB:
            raise SchemaValidationError("Training SNR interval differs from the frozen contract")
        if self.robustness_snr_db != ROBUSTNESS_SNR_DB or self.robustness_realizations != 10:
            raise SchemaValidationError("Robustness protocol differs from the frozen contract")
        if not self.paired_across_snr or not self.validation_is_clean or self.normalization_ddof != 0:
            raise SchemaValidationError("Phase-2 policy differs from the frozen contract")
        if self.task_noise_seeds != TASK_NOISE_SEEDS or self.rng_algorithm != "NumPy PCG64DXSM":
            raise SchemaValidationError("Phase-2 RNG identity differs from the frozen contract")

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))
