from __future__ import annotations

import math
import numpy as np
import torch

from inverse_em.errors import SchemaValidationError
from .decoding import ScientificPrediction
from .losses import CanonicalTargets


def wrapped_angular_error(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.remainder(predicted - target + math.pi, 2.0 * math.pi) - math.pi)


def cartesian_coordinates(rho: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return torch.stack((rho * torch.cos(phi), rho * torch.sin(phi)), dim=-1)


def canonical_errors(prediction: ScientificPrediction, targets: CanonicalTargets) -> dict[str, torch.Tensor]:
    if not isinstance(prediction, ScientificPrediction) or not isinstance(targets, CanonicalTargets):
        raise TypeError("Canonical metrics require ScientificPrediction and CanonicalTargets")
    if prediction.rho.shape != targets.rho.shape:
        raise SchemaValidationError("Prediction and canonical target shapes differ")
    predicted_xy = cartesian_coordinates(prediction.rho, prediction.phi)
    target_xy = cartesian_coordinates(targets.rho, targets.phi)
    return {
        "cartesian": torch.linalg.vector_norm(predicted_xy - target_xy, dim=-1),
        "radial": torch.abs(prediction.rho - targets.rho),
        "angular_radians": wrapped_angular_error(prediction.phi, targets.phi),
    }


def summarize(values: torch.Tensor | np.ndarray) -> dict[str, float]:
    array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise SchemaValidationError("Metric summary requires nonempty finite values")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "rmse": float(np.sqrt(np.mean(np.square(array)))),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def canonical_metrics(prediction: ScientificPrediction, targets: CanonicalTargets) -> dict[str, object]:
    errors = canonical_errors(prediction, targets)
    angular_degrees = torch.rad2deg(errors["angular_radians"])
    return {
        "aggregation": "pooled_active_sources_fixed_canonical_slots",
        "coordinate_units": "normalized_R_equals_1",
        "sample_count": prediction.rho.shape[0],
        "source_count": prediction.rho.numel(),
        "cartesian": summarize(errors["cartesian"]),
        "radial": summarize(errors["radial"]),
        "angular_radians": summarize(errors["angular_radians"]),
        "angular_degrees": summarize(angular_degrees),
    }
