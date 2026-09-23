from __future__ import annotations

import torch
from torch import nn

from inverse_em.errors import SchemaValidationError
from .decoding import RawLocalizerOutput


class CircularLocalizer(nn.Module):
    """Frozen three-slot localization architecture shared by S1, S2, and S3."""

    def __init__(self) -> None:
        super().__init__()
        activation = lambda: nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.features = nn.Sequential(
            nn.Conv1d(4, 32, 5, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode="circular"),
            activation(),
            nn.Conv1d(32, 64, 5, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode="circular"),
            activation(),
            nn.Conv1d(64, 96, 3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode="circular"),
            activation(),
        )
        self.dense = nn.Sequential(nn.Flatten(), nn.Linear(96 * 30, 128, bias=True), activation())
        self.radius_head = nn.Linear(128, 3, bias=True)
        self.angle_head = nn.Linear(128, 6, bias=True)
        self.to(device="cpu", dtype=torch.float64)

    def forward(self, inputs: torch.Tensor) -> RawLocalizerOutput:
        if not isinstance(inputs, torch.Tensor) or inputs.device.type != "cpu" or inputs.dtype is not torch.float64:
            raise SchemaValidationError("CircularLocalizer requires a CPU torch.float64 tensor")
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (4, 30) or not torch.isfinite(inputs).all():
            raise SchemaValidationError("CircularLocalizer requires finite input shape (B,4,30)")
        hidden = self.dense(self.features(inputs))
        angle = self.angle_head(hidden)
        return RawLocalizerOutput(self.radius_head(hidden), angle[:, :3], angle[:, 3:])


def make_localizer() -> CircularLocalizer:
    return CircularLocalizer()


def parameter_count(module: nn.Module, *, trainable_only: bool = True) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if not trainable_only or parameter.requires_grad)
