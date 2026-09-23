from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inverse_em.errors import SchemaValidationError


ANGLE_COUNT = 30


def _immutable(value, dtype, shape):
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape or not np.isfinite(array).all():
        raise SchemaValidationError(f"Expected finite array with shape {shape}")
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=dtype).reshape(shape)


@dataclass(frozen=True)
class ComplexFields:
    electric: np.ndarray
    magnetic: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "electric", _immutable(self.electric, np.complex128, (ANGLE_COUNT,)))
        object.__setattr__(self, "magnetic", _immutable(self.magnetic, np.complex128, (ANGLE_COUNT,)))


@dataclass(frozen=True)
class NoisyComplexFields:
    electric: np.ndarray
    magnetic: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "electric", _immutable(self.electric, np.complex128, (ANGLE_COUNT,)))
        object.__setattr__(self, "magnetic", _immutable(self.magnetic, np.complex128, (ANGLE_COUNT,)))


def complex_fields_to_channels(fields: ComplexFields | NoisyComplexFields) -> np.ndarray:
    if type(fields) not in (ComplexFields, NoisyComplexFields):
        raise SchemaValidationError("Expected clean or noisy complex fields")
    channels = np.stack(
        (fields.electric.real, fields.electric.imag, fields.magnetic.real, fields.magnetic.imag)
    )
    return _immutable(channels, np.float64, (4, ANGLE_COUNT))
