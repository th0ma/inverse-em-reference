from __future__ import annotations

import hashlib
import numpy as np
import torch
from torch import nn

from inverse_em.config.classifier import CLASSIFIER_INITIALIZATION_SEED
from inverse_em.errors import SchemaValidationError


class ResidualClassifierBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad1 = nn.CircularPad1d(1)
        self.conv1 = nn.Conv1d(64, 64, 3, bias=True)
        self.act1 = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.1)
        self.pad2 = nn.CircularPad1d(1)
        self.conv2 = nn.Conv1d(64, 64, 3, bias=True)
        self.act2 = nn.LeakyReLU(0.01)

    def forward(self, value):
        transformed = self.act1(self.conv1(self.pad1(value)))
        transformed = self.dropout(transformed)
        transformed = self.act2(self.conv2(self.pad2(transformed)))
        return value + transformed


class SourceCountClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = nn.CircularPad1d(1)
        self.conv = nn.Conv1d(4, 64, 3, bias=True)
        self.act = nn.LeakyReLU(0.01)
        self.blocks = nn.Sequential(*(ResidualClassifierBlock() for _ in range(4)))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 5, bias=True)

    def forward(self, observations):
        if (not isinstance(observations, torch.Tensor) or observations.device.type != "cpu"
                or observations.dtype is not torch.float64 or observations.ndim != 3
                or observations.shape[1:] != (4, 30)):
            raise SchemaValidationError("SourceCountClassifier requires CPU float64 shape (B,4,30)")
        value = self.act(self.conv(self.pad(observations)))
        for block in self.blocks:
            value = block(value)
        return self.fc(self.pool(value).squeeze(-1))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def historical_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        item = hashlib.sha256(str(array.dtype).encode() + repr(array.shape).encode() + array.tobytes()).hexdigest()
        digest.update(name.encode())
        digest.update(item.encode())
    return digest.hexdigest()


def make_initialized_classifier(seed: int = CLASSIFIER_INITIALIZATION_SEED) -> SourceCountClassifier:
    if seed != CLASSIFIER_INITIALIZATION_SEED:
        raise SchemaValidationError("Only the frozen Phase-4 initialization seed is permitted")
    caller_state = torch.get_rng_state().clone()
    try:
        torch.manual_seed(seed)
        model = SourceCountClassifier()
        model.double()
    finally:
        torch.set_rng_state(caller_state)
    if parameter_count(model) != 99_973:
        raise RuntimeError("Classifier parameter count differs from frozen architecture")
    return model
