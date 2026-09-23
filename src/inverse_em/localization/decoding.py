from __future__ import annotations

from dataclasses import dataclass
import torch

from inverse_em.errors import SchemaValidationError


RADIUS_LOWER = 0.05
RADIUS_UPPER = 0.95
RADIUS_SPAN = 0.90


def _cpu_float64(value: torch.Tensor, name: str, *, columns: int | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype is not torch.float64 or value.device.type != "cpu":
        raise SchemaValidationError(f"{name} must be a CPU torch.float64 tensor")
    if value.ndim != 2 or (columns is not None and value.shape[1] != columns):
        raise SchemaValidationError(f"{name} has an invalid shape")
    if not torch.isfinite(value).all():
        raise SchemaValidationError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class RawLocalizerOutput:
    """Internal three-slot network output before physical decoding."""

    raw_radius: torch.Tensor
    cos_like: torch.Tensor
    sin_like: torch.Tensor

    def __post_init__(self) -> None:
        for name, value in (("raw_radius", self.raw_radius), ("cos_like", self.cos_like), ("sin_like", self.sin_like)):
            _cpu_float64(value, name, columns=3)
        if self.raw_radius.shape != self.cos_like.shape or self.raw_radius.shape != self.sin_like.shape:
            raise SchemaValidationError("Raw localizer outputs must have matching (B,3) shapes")


@dataclass(frozen=True)
class ScientificPrediction:
    """Task-facing physical prediction containing active slots only."""

    rho: torch.Tensor
    cos_like: torch.Tensor
    sin_like: torch.Tensor
    phi: torch.Tensor

    def __post_init__(self) -> None:
        columns = self.rho.shape[1] if isinstance(self.rho, torch.Tensor) and self.rho.ndim == 2 else None
        if columns not in (1, 2, 3):
            raise SchemaValidationError("Scientific predictions require one to three active slots")
        for name, value in (("rho", self.rho), ("cos_like", self.cos_like), ("sin_like", self.sin_like), ("phi", self.phi)):
            _cpu_float64(value, name, columns=columns)
        if any(value.shape != self.rho.shape for value in (self.cos_like, self.sin_like, self.phi)):
            raise SchemaValidationError("Scientific prediction tensors must have matching shapes")

    @property
    def active_count(self) -> int:
        return self.rho.shape[1]


def decode_radius(raw_radius: torch.Tensor) -> torch.Tensor:
    _cpu_float64(raw_radius, "raw_radius")
    return RADIUS_LOWER + RADIUS_SPAN * torch.sigmoid(raw_radius)


def decode_angle(cos_like: torch.Tensor, sin_like: torch.Tensor) -> torch.Tensor:
    _cpu_float64(cos_like, "cos_like")
    _cpu_float64(sin_like, "sin_like")
    if cos_like.shape != sin_like.shape:
        raise SchemaValidationError("cos_like and sin_like must have matching shapes")
    return torch.atan2(sin_like, cos_like)


def decode_scientific(raw: RawLocalizerOutput, active_count: int) -> ScientificPrediction:
    if not isinstance(raw, RawLocalizerOutput):
        raise TypeError("decode_scientific requires RawLocalizerOutput")
    if type(active_count) is not int or active_count not in (1, 2, 3):
        raise SchemaValidationError("active_count must be 1, 2, or 3")
    rho = decode_radius(raw.raw_radius)[:, :active_count]
    cos_like = raw.cos_like[:, :active_count]
    sin_like = raw.sin_like[:, :active_count]
    return ScientificPrediction(rho, cos_like, sin_like, decode_angle(cos_like, sin_like))
