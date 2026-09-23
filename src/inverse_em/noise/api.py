from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from inverse_em.config.phase2 import ROBUSTNESS_SNR_DB,TRAINING_SNR_INTERVAL_DB
from inverse_em.errors import SchemaValidationError
from inverse_em.observations import ANGLE_COUNT,ComplexFields,NoisyComplexFields


def _immutable_direction(value):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (ANGLE_COUNT,) or not np.isfinite(array).all():
        raise SchemaValidationError("Noise directions must be finite length-30 float64 arrays")
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.float64)


@dataclass(frozen=True)
class StandardizedNoiseDirections:
    electric_real: np.ndarray
    electric_imag: np.ndarray
    magnetic_real: np.ndarray
    magnetic_imag: np.ndarray
    master_seed: int
    sample_index: int
    realization_index: int

    def __post_init__(self):
        for name in ("master_seed", "sample_index", "realization_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise SchemaValidationError(f"{name} must be a nonnegative integer")
        for name in ("electric_real", "electric_imag", "magnetic_real", "magnetic_imag"):
            object.__setattr__(self, name, _immutable_direction(getattr(self, name)))


@dataclass(frozen=True)
class ScaledComplexNoise:
    electric: np.ndarray
    magnetic: np.ndarray
    electric_power: float
    magnetic_power: float
    electric_sigma: float
    magnetic_sigma: float
    snr_db: float

    def __post_init__(self):
        for name in ("electric_power", "magnetic_power", "electric_sigma", "magnetic_sigma", "snr_db"):
            if not math.isfinite(float(getattr(self, name))):
                raise SchemaValidationError(f"{name} must be finite")
        if self.electric_power<0 or self.magnetic_power<0 or self.electric_sigma<0 or self.magnetic_sigma<0:
            raise SchemaValidationError("Noise powers and scales must be nonnegative")
        for name in ("electric", "magnetic"):
            value = np.asarray(getattr(self, name), dtype=np.complex128)
            if value.shape != (ANGLE_COUNT,) or not np.isfinite(value).all():
                raise SchemaValidationError("Scaled noise must be finite complex128 length-30 arrays")
            object.__setattr__(self, name, np.frombuffer(np.ascontiguousarray(value).tobytes(), dtype=np.complex128))


def draw_standardized_directions(master_seed: int, sample_index: int, realization_index: int) -> StandardizedNoiseDirections:
    if any(type(x) is not int or x < 0 for x in (master_seed, sample_index, realization_index)):
        raise SchemaValidationError("Noise direction identity must contain nonnegative integers")
    draws = []
    for component in range(4):
        seed = np.random.SeedSequence((master_seed, sample_index, realization_index, component))
        rng = np.random.Generator(np.random.PCG64DXSM(seed))
        draws.append(rng.standard_normal(ANGLE_COUNT, dtype=np.float64))
    return StandardizedNoiseDirections(*draws, master_seed, sample_index, realization_index)


def field_power(field: np.ndarray) -> float:
    value = np.asarray(field, dtype=np.complex128)
    if value.shape != (ANGLE_COUNT,) or not np.isfinite(value).all():
        raise SchemaValidationError("Field must be a finite complex128 length-30 array")
    return float(np.mean(np.abs(value) ** 2, dtype=np.float64))


def scale_directions(fields: ComplexFields, directions: StandardizedNoiseDirections, snr_db: float) -> ScaledComplexNoise:
    gamma = float(snr_db)
    if not math.isfinite(gamma):
        raise SchemaValidationError("SNR must be finite")
    electric_power = field_power(fields.electric)
    magnetic_power = field_power(fields.magnetic)
    factor = 10.0 ** (-gamma / 10.0)
    electric_sigma = math.sqrt(electric_power * factor / 2.0)
    magnetic_sigma = math.sqrt(magnetic_power * factor / 2.0)
    electric = electric_sigma * (directions.electric_real + 1j * directions.electric_imag)
    magnetic = magnetic_sigma * (directions.magnetic_real + 1j * directions.magnetic_imag)
    return ScaledComplexNoise(electric, magnetic, electric_power, magnetic_power, electric_sigma, magnetic_sigma, gamma)


def add_scaled_noise(fields:ComplexFields,noise:ScaledComplexNoise)->NoisyComplexFields:
    if type(fields)is not ComplexFields:raise SchemaValidationError("Noise may be added only to clean analytical fields")
    return NoisyComplexFields(fields.electric+noise.electric,fields.magnetic+noise.magnetic)


def scale_paired_robustness(fields:ComplexFields,directions:StandardizedNoiseDirections)->tuple[ScaledComplexNoise,...]:
    """Scale one fixed direction identity over the frozen robustness grid."""
    return tuple(scale_directions(fields,directions,snr) for snr in ROBUSTNESS_SNR_DB)


def draw_training_snr(rng: np.random.Generator) -> float:
    if not isinstance(rng, np.random.Generator):
        raise SchemaValidationError("Training SNR requires an owned NumPy Generator")
    low, high = TRAINING_SNR_INTERVAL_DB
    return float(rng.uniform(low, high))


class TrainingNoiseStream:
    def __init__(self, augmentation_seed: int):
        if type(augmentation_seed) is not int or augmentation_seed < 0:
            raise SchemaValidationError("augmentation_seed must be a nonnegative integer")
        self._seed = augmentation_seed
        self._rng = np.random.Generator(np.random.PCG64DXSM(augmentation_seed))

    @property
    def augmentation_seed(self) -> int:
        return self._seed

    def draw_snr_db(self) -> float:
        return draw_training_snr(self._rng)

    def draw_directions(self, sample_index: int, realization_index: int) -> StandardizedNoiseDirections:
        if any(type(x) is not int or x < 0 for x in (sample_index, realization_index)):
            raise SchemaValidationError("Noise indices must be nonnegative integers")
        draws = [self._rng.standard_normal(ANGLE_COUNT, dtype=np.float64) for _ in range(4)]
        return StandardizedNoiseDirections(*draws, self._seed, sample_index, realization_index)

    def draw_event(self, sample_index: int, realization_index: int = 0) -> tuple[float, StandardizedNoiseDirections]:
        """Draw one frozen training event: SNR first, then E-real/E-imag/H-real/H-imag."""
        snr_db = self.draw_snr_db()
        directions = self.draw_directions(sample_index, realization_index)
        return snr_db, directions
