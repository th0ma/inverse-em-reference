from __future__ import annotations

from dataclasses import dataclass
import itertools
import torch

from inverse_em.errors import SchemaValidationError
from .decoding import ScientificPrediction
from .losses import CanonicalTargets
from .metrics import cartesian_coordinates


@dataclass(frozen=True)
class DiagnosticAssignment:
    """Diagnostic-only result; intentionally cannot serve as canonical loss targets."""

    chosen_indices: torch.Tensor
    permutations: tuple[tuple[int, ...], ...]
    matched_target_xy: torch.Tensor
    candidate_costs: torch.Tensor
    assignment_changed: torch.Tensor


def diagnostic_assignment(prediction: ScientificPrediction, targets: CanonicalTargets) -> DiagnosticAssignment:
    if not isinstance(prediction, ScientificPrediction) or not isinstance(targets, CanonicalTargets):
        raise TypeError("Diagnostic matching requires scientific predictions and canonical targets")
    if prediction.rho.shape != targets.rho.shape or prediction.active_count not in (2, 3):
        raise SchemaValidationError("Diagnostic matching supports matching canonical S2 or S3 shapes")
    count = prediction.active_count
    permutations = tuple(itertools.permutations(range(count)))
    predicted_xy = cartesian_coordinates(prediction.rho, prediction.phi)
    target_xy = cartesian_coordinates(targets.rho, targets.phi)
    costs = torch.stack([
        (predicted_xy - target_xy[:, permutation, :]).square().sum(dim=(1, 2))
        for permutation in permutations
    ], dim=1)
    chosen = torch.zeros(costs.shape[0], dtype=torch.int64, device="cpu")
    best = costs[:, 0]
    for index in range(1, len(permutations)):
        lower = costs[:, index] < best
        chosen = torch.where(lower, torch.full_like(chosen, index), chosen)
        best = torch.where(lower, costs[:, index], best)
    matched = torch.stack([target_xy[row, permutations[int(chosen[row])], :] for row in range(len(target_xy))])
    return DiagnosticAssignment(chosen, permutations, matched, costs, chosen != 0)
