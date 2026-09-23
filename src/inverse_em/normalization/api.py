from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import numpy as np

from inverse_em.config.phase2 import CHANNEL_ORDER, InverseTask, Phase2ScientificConfig, noise_seeds_for
from inverse_em.errors import SchemaValidationError
from inverse_em.observations import ComplexFields, complex_fields_to_channels
from inverse_em.physics import PhysicsTMForward, Source, observation_angles
from inverse_em.populations import (
    GeneratedPopulation,
    PopulationRole,
    final_population_identity,
    generate_classifier,
    generate_s1,
    generate_s2,
    generate_s3,
)
from inverse_em.provenance.canonical import canonical_sha256
from inverse_em.provenance.phase1 import PhysicsPin
from inverse_em.provenance.phase2 import NormalizationReceipt, TrainingPopulationDefinitionReceipt


def _frozen_vector(value, *, positive=False):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4,) or not np.isfinite(array).all() or (positive and np.any(array <= 0)):
        raise SchemaValidationError("Normalizer vectors require four finite values")
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.float64)


@dataclass(frozen=True)
class FrozenChannelNormalizer:
    task: InverseTask
    mean: np.ndarray
    std: np.ndarray
    observation_count: int
    scalar_count_per_channel: int
    curriculum_stages: tuple[int, ...] = ()
    ddof: int = 0

    def __post_init__(self):
        if not isinstance(self.task, InverseTask):
            raise SchemaValidationError("Normalizer task must be an InverseTask")
        object.__setattr__(self, "mean", _frozen_vector(self.mean))
        object.__setattr__(self, "std", _frozen_vector(self.std, positive=True))
        if type(self.observation_count) is not int or self.observation_count < 1:
            raise SchemaValidationError("Invalid normalizer observation count")
        if self.scalar_count_per_channel != self.observation_count * 30 or self.ddof != 0:
            raise SchemaValidationError("Invalid normalizer scalar count or ddof")
        expected = tuple(range(1, 9)) if self.task is InverseTask.S3 else ()
        if self.curriculum_stages != expected:
            raise SchemaValidationError("Invalid curriculum coverage")

    def transform(self, observations):
        values = np.asarray(observations, dtype=np.float64)
        if values.shape[-2:] != (4, 30) or not np.isfinite(values).all():
            raise SchemaValidationError("Observations must end in finite shape (4,30)")
        prefix = (1,) * (values.ndim - 2)
        return (values - self.mean.reshape(prefix + (4, 1))) / self.std.reshape(prefix + (4, 1))


class _StreamingMoments:
    def __init__(self):
        self._count = 0
        self._observation_count = 0
        self._mean = np.zeros(4, dtype=np.float64)
        self._m2 = np.zeros(4, dtype=np.float64)

    def _merge_statistics(self, count, mean, m2):
        if self._count == 0:
            self._count = int(count)
            self._mean = mean.copy()
            self._m2 = m2.copy()
            return
        total = self._count + int(count)
        delta = mean - self._mean
        self._m2 += m2 + delta * delta * (self._count * count / total)
        self._mean += delta * (count / total)
        self._count = total

    def update(self, values):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3 or array.shape[1:] != (4, 30) or array.shape[0] < 1 or not np.isfinite(array).all():
            raise SchemaValidationError("Moment input must have finite shape (N,4,30)")
        mean = array.mean(axis=(0, 2), dtype=np.float64)
        centered = array - mean[None, :, None]
        m2 = np.sum(centered * centered, axis=(0, 2), dtype=np.float64)
        self._merge_statistics(array.shape[0] * 30, mean, m2)
        self._observation_count += array.shape[0]
        return self

    def merge(self, other):
        if type(other) is not _StreamingMoments:
            raise SchemaValidationError("Only streaming moments may merge")
        if other._count:
            self._merge_statistics(other._count, other._mean, other._m2)
            self._observation_count += other._observation_count
        return self

    @property
    def mean(self):
        return self._mean.copy()

    @property
    def std(self):
        if self._count < 1:
            raise SchemaValidationError("Cannot compute empty statistics")
        return np.sqrt(self._m2 / self._count)

    @property
    def observation_count(self):
        return self._observation_count

    @property
    def scalar_count_per_channel(self):
        return self._count


@dataclass(frozen=True)
class GenericChannelStatistics:
    mean: np.ndarray
    std: np.ndarray
    observation_count: int
    scalar_count_per_channel: int

    def __post_init__(self):
        object.__setattr__(self, "mean", _frozen_vector(self.mean))
        object.__setattr__(self, "std", _frozen_vector(self.std, positive=True))


class GenericChannelStatsAccumulator:
    def __init__(self):
        self._moments = _StreamingMoments()

    def update(self, values):
        self._moments.update(values)
        return self

    def merge(self, other):
        if type(other) is not GenericChannelStatsAccumulator:
            raise SchemaValidationError("Only generic channel accumulators may merge")
        self._moments.merge(other._moments)
        return self

    def freeze(self):
        if self._moments.scalar_count_per_channel < 1:
            raise SchemaValidationError("Cannot freeze empty generic statistics")
        return GenericChannelStatistics(
            self._moments.mean,
            self._moments.std,
            self._moments.observation_count,
            self._moments.scalar_count_per_channel,
        )


def _normalizer_sha256(normalizer: FrozenChannelNormalizer) -> str:
    return canonical_sha256(
        {
            "task": normalizer.task,
            "mean": tuple(float(value) for value in normalizer.mean),
            "std": tuple(float(value) for value in normalizer.std),
            "observation_count": normalizer.observation_count,
            "scalar_count_per_channel": normalizer.scalar_count_per_channel,
            "curriculum_stages": normalizer.curriculum_stages,
            "ddof": normalizer.ddof,
        }
    )


def _repository_revision() -> str:
    repository = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "UNAVAILABLE"
    return result if len(result) == 40 else "UNAVAILABLE"


def _validate_population_set(task: InverseTask, populations: tuple[GeneratedPopulation, ...]):
    if not populations or any(type(population) is not GeneratedPopulation for population in populations):
        raise SchemaValidationError("Fitting requires generated Phase-1 populations")
    definitions = tuple(population.definition for population in populations)
    if any(definition.role is not PopulationRole.TRAINING for definition in definitions):
        raise SchemaValidationError("Normalizer fitting uses TRAINING populations only")
    expected_id = {
        InverseTask.CLASSIFIER: "classifier_final",
        InverseTask.S1: "s1_final",
        InverseTask.S2: "s2_final",
        InverseTask.S3: "s3_final",
    }[task]
    if any(definition.population_id != expected_id for definition in definitions):
        raise SchemaValidationError("Population task differs from fitting task")
    source_counts = tuple(sorted(definition.source_count for definition in definitions))
    expected_counts = {
        InverseTask.CLASSIFIER: (1, 2, 3, 4, 5),
        InverseTask.S1: (1,),
        InverseTask.S2: (2,),
        InverseTask.S3: (3,) * 8,
    }[task]
    if source_counts != expected_counts:
        raise SchemaValidationError("Training source-count coverage is incomplete")
    stages = tuple(sorted(definition.stage for definition in definitions if definition.stage is not None))
    expected_stages = tuple(range(1, 9)) if task is InverseTask.S3 else ()
    if stages != expected_stages:
        raise SchemaValidationError("Training curriculum coverage is incomplete")


def _fit_generated_training_populations(
    task: InverseTask,
    populations,
    training_master_seeds: tuple[int, ...],
) -> tuple[FrozenChannelNormalizer, NormalizationReceipt]:
    """Private numerical seam shared by production fitting and small reference tests."""
    population_tuple = tuple(populations)
    _validate_population_set(task, population_tuple)
    moments = _StreamingMoments()
    physics = PhysicsTMForward()
    angles = observation_angles(30)
    for population in population_tuple:
        for index in range(population.definition.count):
            electric = np.zeros(30, dtype=np.complex128)
            magnetic = np.zeros(30, dtype=np.complex128)
            for rho, phi, amplitude in zip(
                population.rho[index], population.phi[index], population.amplitude[index]
            ):
                fields = physics.evaluate(Source(float(rho), float(phi), float(amplitude)), angles)
                electric += fields.electric
                magnetic += fields.magnetic
            moments.update(complex_fields_to_channels(ComplexFields(electric, magnetic)))
    stages = tuple(range(1, 9)) if task is InverseTask.S3 else ()
    normalizer = FrozenChannelNormalizer(
        task,
        moments.mean,
        moments.std,
        moments.observation_count,
        moments.scalar_count_per_channel,
        stages,
    )
    config = Phase2ScientificConfig()
    seeds = noise_seeds_for(task)
    definitions = tuple(
        TrainingPopulationDefinitionReceipt.from_definition(population.definition)
        for population in population_tuple
    )
    physics_pin = PhysicsPin()
    receipt = NormalizationReceipt(
        task=task,
        specification_edition=config.specification_edition,
        population_definitions=definitions,
        training_role=PopulationRole.TRAINING.value,
        observation_count=normalizer.observation_count,
        scalar_count_per_channel=normalizer.scalar_count_per_channel,
        source_counts=tuple(sorted({definition.source_count for definition in definitions})),
        curriculum_stages=stages,
        training_master_seeds=training_master_seeds,
        augmentation_seed=seeds.augmentation_seed,
        robustness_seed=seeds.robustness_seed,
        training_snr_interval_db=config.training_snr_interval_db,
        robustness_snr_db=config.robustness_snr_db,
        robustness_realizations=config.robustness_realizations,
        channel_order=CHANNEL_ORDER,
        ddof=0,
        physics_revision=physics_pin.upstream_commit,
        physics_source_sha256=physics_pin.upstream_source_sha256,
        angle_grid_definition="2*pi*arange(30,dtype=float64)/30",
        normalizer_mean=tuple(float(value) for value in normalizer.mean),
        normalizer_std=tuple(float(value) for value in normalizer.std),
        normalizer_sha256=_normalizer_sha256(normalizer),
        noise_model_identity=config.noise_model_identity,
        paired_across_snr=config.paired_across_snr,
        repository_revision=_repository_revision(),
        phase2_configuration_sha256=config.sha256,
    )
    return normalizer, receipt


def _frozen_training_populations(task: InverseTask):
    if not isinstance(task, InverseTask):
        raise SchemaValidationError("Task must be an InverseTask")
    identity = final_population_identity(task.value, PopulationRole.TRAINING)
    if task is InverseTask.CLASSIFIER:
        populations = generate_classifier(identity.per_class, identity.seed, PopulationRole.TRAINING)
        return tuple(populations[source_count] for source_count in range(1, 6)), (identity.seed,)
    if task is InverseTask.S1:
        return (generate_s1(identity.count, identity.seed, PopulationRole.TRAINING),), (identity.seed,)
    if task is InverseTask.S2:
        return (generate_s2(identity.count, identity.seed, PopulationRole.TRAINING),), (identity.seed,)
    populations = tuple(
        generate_s3(identity.count_per_stage, identity.seed, stage, PopulationRole.TRAINING)
        for stage in range(1, 9)
    )
    return populations, (identity.seed,)


def fit_clean_training_normalizer(task: InverseTask) -> tuple[FrozenChannelNormalizer, NormalizationReceipt]:
    """Fit from the task's frozen clean analytical TRAINING population(s)."""
    populations, master_seeds = _frozen_training_populations(task)
    return _fit_generated_training_populations(task, populations, master_seeds)
