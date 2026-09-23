from dataclasses import replace

import numpy as np
import pytest

from inverse_em.config import *
from inverse_em.errors import SchemaValidationError
from inverse_em.provenance.phase2 import NormalizationReceipt, TrainingPopulationDefinitionReceipt


def test_literal_phase2_configuration():
    assert CHANNEL_ORDER == ("Re(E_z)", "Im(E_z)", "Re(H_phi)", "Im(H_phi)")
    assert TRAINING_SNR_INTERVAL_DB == (30.0, 40.0)
    assert ROBUSTNESS_SNR_DB == (40.0, 30.0, 20.0, 15.0, 10.0)
    assert ROBUSTNESS_REALIZATIONS == 10
    assert NORMALIZATION_DDOF == 0
    assert [(x.task.value, x.augmentation_seed, x.robustness_seed) for x in TASK_NOISE_SEEDS] == [
        ("classifier", 20261905, 20261907),
        ("s1", 20261815, 20261817),
        ("s2", 20262005, 20262007),
        ("s3", 20262205, 20262207),
    ]


def test_configuration_identity_is_deterministic():
    assert Phase2ScientificConfig().sha256 == Phase2ScientificConfig().sha256
    assert len(Phase2ScientificConfig().sha256) == 64


def receipt():
    config = Phase2ScientificConfig()
    definition = TrainingPopulationDefinitionReceipt(
        "s1_final", "TRAINING", 2, 1, 0.05, 0.95, 1.0,
        "area_uniform", "uniform_[0,2pi)", "NumPy PCG64DXSM", 7,
        None, None, None,
    )
    return NormalizationReceipt(
        InverseTask.S1, config.specification_edition, (definition,), "TRAINING",
        2, 60, (1,), (), (7,), 20261815, 20261817,
        config.training_snr_interval_db, config.robustness_snr_db,
        config.robustness_realizations, CHANNEL_ORDER, 0,
        "a079c3899bd33f636ab9c8b1571d8e919c839335",
        "b0d8200ef30d05720f0b697822dc0e037935243beea9084edf410c9725c32005",
        "2*pi*arange(30,dtype=float64)/30", (1.0, 2.0, 3.0, 4.0),
        (2.0, 3.0, 4.0, 5.0), "a" * 64, config.noise_model_identity,
        True, "42371b246cc7dd37cefecf1f184f6dfa7efb161f", config.sha256,
    )


def test_receipt_is_descriptive_complete_and_deterministic():
    first = receipt()
    second = receipt()
    assert first == second
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64


@pytest.mark.parametrize("field,value", [("observation_count", 3), ("ddof", 1), ("training_role", "SEALED")])
def test_receipt_internal_inconsistency_rejects(field, value):
    with pytest.raises(SchemaValidationError):
        replace(receipt(), **{field: value})


def test_receipt_is_plain_data_not_required_for_numerical_access():
    assert not hasattr(NormalizationReceipt, "authorize")
    assert not hasattr(NormalizationReceipt, "verify_authority")
