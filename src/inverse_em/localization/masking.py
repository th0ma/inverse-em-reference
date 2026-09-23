from __future__ import annotations

import torch

from inverse_em.errors import SchemaValidationError
from .decoding import RawLocalizerOutput, ScientificPrediction, decode_scientific


def active_slot_mask(batch_size: int, active_count: int) -> torch.Tensor:
    if type(batch_size) is not int or batch_size < 0:
        raise SchemaValidationError("batch_size must be a non-negative integer")
    if type(active_count) is not int or active_count not in (1, 2, 3):
        raise SchemaValidationError("active_count must be 1, 2, or 3")
    return (torch.arange(3, device="cpu")[None, :] < active_count).expand(batch_size, -1)


def scientific_view(raw: RawLocalizerOutput, active_count: int) -> ScientificPrediction:
    return decode_scientific(raw, active_count)
