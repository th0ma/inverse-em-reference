from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F

from inverse_em.errors import SchemaValidationError


@dataclass(frozen=True)
class ClassificationMetrics:
    cross_entropy: float
    accuracy: float
    macro_f1: float
    per_class_precision: tuple[float, ...]
    per_class_recall: tuple[float, ...]
    per_class_f1: tuple[float, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> ClassificationMetrics:
    if (not isinstance(logits, torch.Tensor) or logits.device.type != "cpu" or logits.dtype is not torch.float64
            or logits.ndim != 2 or logits.shape[1] != 5 or not torch.isfinite(logits).all()):
        raise SchemaValidationError("Metrics require finite CPU float64 logits of shape (N,5)")
    if (not isinstance(targets, torch.Tensor) or targets.device.type != "cpu" or targets.dtype is not torch.int64
            or targets.shape != (len(logits),) or len(targets) == 0 or torch.any((targets < 0) | (targets > 4))):
        raise SchemaValidationError("Metrics require nonempty int64 labels 0..4")
    predictions = torch.argmax(logits, dim=1); matrix = np.zeros((5, 5), dtype=np.int64)
    for truth, prediction in zip(targets.tolist(), predictions.tolist()): matrix[truth, prediction] += 1
    precision, recall, f1 = [], [], []
    for label in range(5):
        tp = int(matrix[label, label]); fp = int(matrix[:, label].sum() - tp); fn = int(matrix[label, :].sum() - tp)
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        score = 2 * p * r / (p + r) if p + r else 0.0
        precision.append(p); recall.append(r); f1.append(score)
    return ClassificationMetrics(float(F.cross_entropy(logits, targets, reduction="mean", label_smoothing=0.0)),
        float((predictions == targets).to(torch.float64).mean()), float(np.mean(f1)), tuple(precision), tuple(recall),
        tuple(f1), tuple(tuple(int(value) for value in row) for row in matrix))
