import math
import torch

from inverse_em.classifier import SelectionState, classification_metrics


def test_metrics_confusion_orientation_and_zero_denominator():
    targets = torch.tensor([0, 0, 1, 2, 3], dtype=torch.int64)
    predictions = [0, 1, 1, 1, 3]
    logits = torch.full((5, 5), -3.0, dtype=torch.float64)
    for row, prediction in enumerate(predictions): logits[row, prediction] = 3
    result = classification_metrics(logits, targets)
    assert result.confusion_matrix[0] == (1, 1, 0, 0, 0)
    assert result.confusion_matrix[2] == (0, 1, 0, 0, 0)
    assert result.accuracy == 0.6 and result.per_class_f1[4] == 0.0
    assert math.isclose(result.macro_f1, sum(result.per_class_f1) / 5)


def test_cross_entropy_is_record_weighted_full_population_value():
    logits = torch.arange(35, dtype=torch.float64).reshape(7, 5) / 10
    targets = torch.arange(7, dtype=torch.int64) % 5
    result = classification_metrics(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits, targets, reduction="mean", label_smoothing=0.0)
    assert result.cross_entropy == float(expected)


def test_strict_best_ties_and_small_improvement_are_distinct_from_early_reference():
    state = SelectionState()
    assert state.update(0.5, 1, 1) == (True, False)
    assert state.update(0.5, 2, 2)[0] is False and state.best_epoch == 1
    improved, _ = state.update(0.50005, 3, 3)
    assert improved and state.best_epoch == 3 and state.early_reference == 0.5 and state.bad_epochs == 2


def test_early_stop_exact_threshold_and_patience_boundary():
    state = SelectionState(); state.update(0.5, 1, 1)
    for epoch in range(2, 21): assert state.update(0.5001, epoch, epoch)[1] is False
    assert state.update(0.5001, 21, 21)[1] is True
