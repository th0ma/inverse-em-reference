import numpy as np
import pytest
import torch

from inverse_em.classifier import (BoundedClassifierTrainer, capture_noise_state, make_initialized_classifier,
                                   restore_noise_state)
from inverse_em.config.phase2 import InverseTask
from inverse_em.noise import TrainingNoiseStream
from inverse_em.normalization import FrozenChannelNormalizer
from inverse_em.observations import ComplexFields


def fixtures():
    fields = tuple(ComplexFields(np.full(30, i + 1j * (i + 1)), np.full(30, 2 * i - 1j)) for i in range(5))
    labels = torch.arange(5, dtype=torch.int64)
    norm = FrozenChannelNormalizer(InverseTask.CLASSIFIER, np.zeros(4), np.ones(4), 1, 30)
    return fields, labels, norm


def test_noise_state_round_trip_exact():
    stream = TrainingNoiseStream(20261905); stream.draw_event(0)
    receipt = capture_noise_state(stream, 1); expected = stream.draw_event(1)
    restored, count = restore_noise_state(receipt); actual = restored.draw_event(1)
    assert count == 1 and expected[0] == actual[0]
    assert all(np.array_equal(getattr(expected[1], name), getattr(actual[1], name))
               for name in ("electric_real", "electric_imag", "magnetic_real", "magnetic_imag"))


def test_optimizer_scheduler_and_bounded_update_arithmetic():
    trainer = BoundedClassifierTrainer(make_initialized_classifier()); fields, labels, norm = fixtures()
    trainer.run(fields, labels, fields, labels, norm, 1)
    group = trainer.optimizer.param_groups[0]
    assert (group["lr"], group["betas"], group["eps"], group["weight_decay"], group["amsgrad"]) == (1e-3, (0.9, 0.999), 1e-8, 0, False)
    assert trainer.scheduler.mode == "max" and trainer.scheduler.factor == .5 and trainer.scheduler.patience == 8
    assert trainer.updates == 1 and trainer.history[0]["epoch"] == 1 and trainer.noise_event_count == 5


def test_scientific_scale_and_more_than_two_epochs_rejected():
    trainer = BoundedClassifierTrainer(make_initialized_classifier()); fields, labels, norm = fixtures()
    with pytest.raises(PermissionError): trainer.run(fields * 21, labels.repeat(21), fields, labels, norm, 1)
    with pytest.raises(PermissionError): trainer.run(fields, labels, fields, labels, norm, 3)
