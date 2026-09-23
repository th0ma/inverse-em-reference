from __future__ import annotations

from dataclasses import dataclass
import torch

from inverse_em.errors import SchemaValidationError
from .decoding import ScientificPrediction, _cpu_float64


@dataclass(frozen=True)
class CanonicalTargets:
    """Canonical fixed-slot targets; diagnostic matches are deliberately a different type."""

    rho: torch.Tensor
    phi: torch.Tensor

    def __post_init__(self) -> None:
        _cpu_float64(self.rho, "rho")
        _cpu_float64(self.phi, "phi")
        if self.rho.shape != self.phi.shape or self.rho.shape[1] not in (1, 2, 3):
            raise SchemaValidationError("Canonical targets require matching (B,S), S=1..3")

    @property
    def active_count(self) -> int:
        return self.rho.shape[1]


@dataclass(frozen=True)
class LocalizationComponents:
    rho: torch.Tensor
    phi: torch.Tensor
    circle: torch.Tensor

    @property
    def uniform(self) -> torch.Tensor:
        return self.rho + 2.0 * self.phi + 0.1 * self.circle


def localization_components(prediction: ScientificPrediction, targets: CanonicalTargets) -> LocalizationComponents:
    if not isinstance(prediction, ScientificPrediction) or not isinstance(targets, CanonicalTargets):
        raise TypeError("Canonical localization loss requires ScientificPrediction and CanonicalTargets")
    if prediction.rho.shape != targets.rho.shape:
        raise SchemaValidationError("Prediction and canonical target shapes differ")
    radial = (prediction.rho - targets.rho).square()
    angular = (prediction.cos_like - torch.cos(targets.phi)).square() + (prediction.sin_like - torch.sin(targets.phi)).square()
    circle = (prediction.cos_like.square() + prediction.sin_like.square() - 1.0).square()
    return LocalizationComponents(radial, angular, circle)


def reduce_s1(components: LocalizationComponents) -> torch.Tensor:
    if components.rho.shape[1] != 1:
        raise SchemaValidationError("S1 reduction requires one active slot")
    return components.uniform.mean()


def reduce_s2_canonical(components: LocalizationComponents) -> torch.Tensor:
    if components.rho.shape[1] != 2:
        raise SchemaValidationError("S2 canonical reduction requires two active slots")
    return components.uniform.sum(dim=1).mean()


def reduce_outer_s3(components: LocalizationComponents, targets: CanonicalTargets, outer_constant: float = 1.635) -> torch.Tensor:
    if components.rho.shape[1] != 3 or targets.active_count != 3 or components.rho.shape != targets.rho.shape:
        raise SchemaValidationError("S3 OUTER reduction requires three canonical active slots")
    if type(outer_constant) is not float or outer_constant != 1.635:
        raise SchemaValidationError("OUTER constant differs from frozen value 1.635")
    weights = (1.0 + targets.rho) / outer_constant
    radial = (weights * components.rho).mean()
    angular = (weights * components.phi).mean()
    circle = components.circle.mean()
    return radial + 2.0 * angular + 0.1 * circle
