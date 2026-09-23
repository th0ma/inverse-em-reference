from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re

from inverse_em.config.phase2 import CHANNEL_ORDER, InverseTask
from inverse_em.errors import SchemaValidationError
from inverse_em.populations import PopulationDefinition, PopulationRole
from inverse_em.provenance.canonical import canonical_sha256


@dataclass(frozen=True)
class TrainingPopulationDefinitionReceipt:
    population_id: str
    role: str
    count: int
    source_count: int
    rho_min: float
    rho_max: float
    amplitude: float
    radial_distribution: str
    azimuth_distribution: str
    rng_algorithm: str
    rng_seed: int
    stage: int | None
    separation_radial: float | None
    separation_angular_degrees: float | None

    @classmethod
    def from_definition(cls, definition: PopulationDefinition):
        separation = definition.separation
        return cls(
            population_id=definition.population_id,
            role=definition.role.value,
            count=definition.count,
            source_count=definition.source_count,
            rho_min=float(definition.source_law.rho_min),
            rho_max=float(definition.source_law.rho_max),
            amplitude=float(definition.source_law.amplitude),
            radial_distribution=definition.source_law.radial_distribution,
            azimuth_distribution=definition.source_law.azimuth_distribution,
            rng_algorithm=definition.rng.algorithm,
            rng_seed=definition.rng.seed,
            stage=definition.stage,
            separation_radial=None if separation is None else float(separation.radial),
            separation_angular_degrees=None if separation is None else float(separation.angular_degrees),
        )

    def __post_init__(self):
        if self.role != PopulationRole.TRAINING.value:
            raise SchemaValidationError("Normalization receipts describe training populations only")
        if self.count < 1 or self.source_count not in range(1, 6) or self.rng_seed < 0:
            raise SchemaValidationError("Invalid training population receipt")
        if self.rng_algorithm != "NumPy PCG64DXSM":
            raise SchemaValidationError("Unexpected population RNG")
        values = (self.rho_min, self.rho_max, self.amplitude)
        if not all(math.isfinite(value) for value in values):
            raise SchemaValidationError("Population receipt contains non-finite values")


@dataclass(frozen=True)
class NormalizationReceipt:
    task: InverseTask
    specification_edition: str
    population_definitions: tuple[TrainingPopulationDefinitionReceipt, ...]
    training_role: str
    observation_count: int
    scalar_count_per_channel: int
    source_counts: tuple[int, ...]
    curriculum_stages: tuple[int, ...]
    training_master_seeds: tuple[int, ...]
    augmentation_seed: int
    robustness_seed: int
    training_snr_interval_db: tuple[float, float]
    robustness_snr_db: tuple[float, ...]
    robustness_realizations: int
    channel_order: tuple[str, ...]
    ddof: int
    physics_revision: str
    physics_source_sha256: str
    angle_grid_definition: str
    normalizer_mean: tuple[float, ...]
    normalizer_std: tuple[float, ...]
    normalizer_sha256: str
    noise_model_identity: str
    paired_across_snr: bool
    repository_revision: str
    phase2_configuration_sha256: str

    def __post_init__(self):
        if not isinstance(self.task, InverseTask) or not self.population_definitions:
            raise SchemaValidationError("Receipt requires a task and training definitions")
        if self.training_role != PopulationRole.TRAINING.value:
            raise SchemaValidationError("Receipt role must be TRAINING")
        if self.observation_count != sum(item.count for item in self.population_definitions):
            raise SchemaValidationError("Receipt observation count differs from population definitions")
        if self.scalar_count_per_channel != self.observation_count * 30 or self.ddof != 0:
            raise SchemaValidationError("Receipt normalization counts are inconsistent")
        if self.source_counts != tuple(sorted({item.source_count for item in self.population_definitions})):
            raise SchemaValidationError("Receipt source counts are inconsistent")
        stages = tuple(sorted(item.stage for item in self.population_definitions if item.stage is not None))
        if self.curriculum_stages != stages:
            raise SchemaValidationError("Receipt curriculum stages are inconsistent")
        if self.channel_order != CHANNEL_ORDER or self.angle_grid_definition != "2*pi*arange(30,dtype=float64)/30":
            raise SchemaValidationError("Receipt observation convention differs from Phase 2")
        if len(self.normalizer_mean) != 4 or len(self.normalizer_std) != 4:
            raise SchemaValidationError("Receipt requires four channel statistics")
        if not all(math.isfinite(value) for value in self.normalizer_mean + self.normalizer_std):
            raise SchemaValidationError("Receipt statistics must be finite")
        if not all(value > 0 for value in self.normalizer_std):
            raise SchemaValidationError("Receipt standard deviations must be positive")
        for digest in (self.physics_source_sha256, self.normalizer_sha256, self.phase2_configuration_sha256):
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise SchemaValidationError("Receipt contains an invalid SHA-256")
        if self.repository_revision != "UNAVAILABLE" and not re.fullmatch(r"[0-9a-f]{40}", self.repository_revision):
            raise SchemaValidationError("Receipt repository revision is invalid")

    @property
    def sha256(self) -> str:
        """Deterministic descriptive identity; this digest confers no authority."""
        return canonical_sha256(asdict(self))
