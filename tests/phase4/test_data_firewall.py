import numpy as np
import pytest
import torch
from dataclasses import replace

from inverse_em.classifier import (analytical_observations, classifier_label, clean_observation_tensor,
                                   epoch_minibatches, epoch_permutation, noisy_observation_tensor,
                                   validate_population_mapping)
from inverse_em.config.phase2 import InverseTask
from inverse_em.noise import TrainingNoiseStream
from inverse_em.normalization import FrozenChannelNormalizer
from inverse_em.observations import ComplexFields
from inverse_em.populations import GeneratedPopulation, PopulationRole, RNGIdentity, generate_classifier


def normalizer(): return FrozenChannelNormalizer(InverseTask.CLASSIFIER, np.zeros(4), np.ones(4), 1, 30)


def test_labels_and_role_firewall():
    assert [classifier_label(value) for value in range(1, 6)] == list(range(5))
    validate_population_mapping(generate_classifier(1, 20261901, PopulationRole.TRAINING), PopulationRole.TRAINING)
    validate_population_mapping(generate_classifier(1, 20261902, PopulationRole.VALIDATION), PopulationRole.VALIDATION)
    with pytest.raises(PermissionError): validate_population_mapping({}, PopulationRole.SEALED)
    with pytest.raises(PermissionError): generate_classifier(1, 20261906, PopulationRole.SEALED)


def test_tiny_analytical_pipeline_and_clean_validation():
    fields, labels = analytical_observations(generate_classifier(1, 20261902, PopulationRole.VALIDATION), PopulationRole.VALIDATION)
    tensor = clean_observation_tensor(fields, normalizer())
    assert tensor.shape == (5, 4, 30) and tensor.dtype is torch.float64
    assert labels.tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("role,seed", [
    (PopulationRole.TRAINING, 20261902), (PopulationRole.TRAINING, 20261906),
    (PopulationRole.TRAINING, 20261907), (PopulationRole.VALIDATION, 20261901),
    (PopulationRole.VALIDATION, 20261906), (PopulationRole.VALIDATION, 20261907),
])
def test_role_seed_mismatch_rejected_without_rng_construction(role, seed):
    source = generate_classifier(1, 20261901 if role is PopulationRole.TRAINING else 20261902, role)
    populations = {}
    for count, population in source.items():
        definition = replace(population.definition, rng=RNGIdentity("NumPy PCG64DXSM", seed))
        populations[count] = GeneratedPopulation(definition, population.rho, population.phi,
                                                 population.amplitude, population.proposal_ordinal)
    with pytest.raises(PermissionError): validate_population_mapping(populations, role)


@pytest.mark.parametrize("task", [InverseTask.S1, InverseTask.S2, InverseTask.S3])
def test_non_classifier_normalizers_rejected(task):
    stages = tuple(range(1, 9)) if task is InverseTask.S3 else ()
    wrong = FrozenChannelNormalizer(task, np.zeros(4), np.ones(4), 1, 30, stages)
    field = ComplexFields(np.ones(30, dtype=np.complex128), np.ones(30, dtype=np.complex128))
    with pytest.raises(Exception): clean_observation_tensor((field,), wrong)


def test_non_normalizer_object_rejected():
    field = ComplexFields(np.ones(30, dtype=np.complex128), np.ones(30, dtype=np.complex128))
    with pytest.raises(Exception): clean_observation_tensor((field,), object())


def test_noise_precedes_channelization_and_normalization():
    field = ComplexFields(np.linspace(1, 2, 30) + 1j, np.linspace(2, 3, 30) - 2j)
    clean = clean_observation_tensor((field,), normalizer())
    noisy, receipts = noisy_observation_tensor((field,), normalizer(), TrainingNoiseStream(20261905))
    assert receipts[0][0] == 0 and 30 <= receipts[0][1] < 40 and not torch.equal(clean, noisy)


def test_exact_keyed_order_and_tail_batch():
    assert epoch_permutation(10, 1, 20261904).tolist() == [1, 3, 7, 0, 4, 8, 9, 5, 2, 6]
    assert epoch_permutation(10, 2, 20261904).tolist() == [6, 2, 7, 8, 1, 4, 9, 3, 5, 0]
    batches = epoch_minibatches(257, 1, 20261904, 128)
    assert tuple(map(len, batches)) == (128, 128, 1)
    torch.manual_seed(1); before = epoch_permutation(20, 3, 20261904)
    torch.rand(100); assert np.array_equal(before, epoch_permutation(20, 3, 20261904))
