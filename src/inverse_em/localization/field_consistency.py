from __future__ import annotations

import torch
from torch import nn

from inverse_em.errors import SchemaValidationError
from inverse_em.surrogates.inference import FrozenSurrogate, freeze_surrogate, superpose_complex
from .decoding import ScientificPrediction


FIELD_CHANNELS = ("Re(E)", "Im(E)", "Re(H)", "Im(H)")


class FieldConsistency(nn.Module):
    """Training-only differentiable FC using frozen Phase-3 surrogates."""

    def __init__(self, electric: FrozenSurrogate, magnetic: FrozenSurrogate, observation_angles: torch.Tensor) -> None:
        super().__init__()
        if not isinstance(electric, FrozenSurrogate) or not isinstance(magnetic, FrozenSurrogate):
            raise TypeError("FieldConsistency requires Phase-3 FrozenSurrogate instances")
        if not isinstance(observation_angles, torch.Tensor) or observation_angles.dtype is not torch.float64 or observation_angles.device.type != "cpu" or observation_angles.shape != (30,) or not torch.isfinite(observation_angles).all():
            raise SchemaValidationError("observation_angles must be finite CPU float64 shape (30,)")
        self.electric = electric
        self.magnetic = magnetic
        freeze_surrogate(self.electric)
        freeze_surrogate(self.magnetic)
        self.log_variances = nn.Parameter(torch.zeros(4, dtype=torch.float64, device="cpu"))
        self.register_buffer("observation_angles", observation_angles.detach().clone())

    def reconstruct(self, prediction: ScientificPrediction) -> torch.Tensor:
        if not isinstance(prediction, ScientificPrediction):
            raise TypeError("FC reconstruction requires an active-only ScientificPrediction")
        rho = prediction.rho[:, :, None]
        phi = prediction.phi[:, :, None]
        theta = self.observation_angles[None, None, :]
        amplitude = torch.ones_like(rho)
        electric = superpose_complex(self.electric, rho, phi, theta, amplitude, source_dim=1)
        magnetic = superpose_complex(self.magnetic, rho, phi, theta, amplitude, source_dim=1)
        return torch.stack((electric.real, electric.imag, magnetic.real, magnetic.imag), dim=1)

    def channel_losses(self, prediction: ScientificPrediction, clean_analytical_raw: torch.Tensor) -> torch.Tensor:
        if not isinstance(clean_analytical_raw, torch.Tensor) or clean_analytical_raw.dtype is not torch.float64 or clean_analytical_raw.device.type != "cpu":
            raise SchemaValidationError("FC target must be a CPU torch.float64 tensor")
        if clean_analytical_raw.shape != (prediction.rho.shape[0], 4, 30) or not torch.isfinite(clean_analytical_raw).all():
            raise SchemaValidationError("FC target must be finite clean analytical raw channels (B,4,30)")
        return (self.reconstruct(prediction) - clean_analytical_raw).square().mean(dim=(0, 2))

    def kendall_loss(self, channel_losses: torch.Tensor) -> torch.Tensor:
        if not isinstance(channel_losses, torch.Tensor) or channel_losses.dtype is not torch.float64 or channel_losses.device.type != "cpu" or channel_losses.shape != (4,) or not torch.isfinite(channel_losses).all():
            raise SchemaValidationError("Kendall channel losses must be finite CPU float64 shape (4,)")
        return 0.5 * (torch.exp(-self.log_variances) * channel_losses + self.log_variances).sum()

    def forward(self, prediction: ScientificPrediction, clean_analytical_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        losses = self.channel_losses(prediction, clean_analytical_raw)
        return self.kendall_loss(losses), losses


def compose_fc_objective(localization_loss: torch.Tensor, kendall_loss: torch.Tensor, coefficient: float = 0.03) -> torch.Tensor:
    if not all(isinstance(value, torch.Tensor) and value.ndim == 0 and value.dtype is torch.float64 and torch.isfinite(value) for value in (localization_loss, kendall_loss)):
        raise SchemaValidationError("Objective terms must be finite scalar float64 tensors")
    if type(coefficient) is not float or coefficient != 0.03:
        raise SchemaValidationError("FC coefficient differs from frozen value 0.03")
    return localization_loss + coefficient * kendall_loss
