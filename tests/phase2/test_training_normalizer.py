import numpy as np
import pytest

import inverse_em.normalization.api as api
from inverse_em.config import InverseTask
from inverse_em.normalization import FrozenChannelNormalizer, fit_clean_training_normalizer
from inverse_em.physics import PhysicsTMForward, Source, observation_angles
from inverse_em.populations import (
    PopulationRole,
    generate_classifier,
    generate_s1,
    generate_s2,
    generate_s3,
)


def direct_values(populations):
    physics = PhysicsTMForward()
    angles = observation_angles(30)
    output = []
    for population in populations:
        for index in range(population.definition.count):
            electric = np.zeros(30, dtype=np.complex128)
            magnetic = np.zeros(30, dtype=np.complex128)
            for rho, phi, amplitude in zip(
                population.rho[index], population.phi[index], population.amplitude[index]
            ):
                fields = physics.evaluate(Source(float(rho), float(phi), float(amplitude)), angles)
                electric += fields.electric
                magnetic += fields.magnetic
            output.append(np.stack((electric.real, electric.imag, magnetic.real, magnetic.imag)))
    return np.asarray(output)


def assert_reference(task, populations, master_seed, expected_count):
    normalizer, receipt = api._fit_generated_training_populations(task, populations, (master_seed,))
    expected = direct_values(populations)
    assert normalizer.observation_count == expected_count
    assert normalizer.scalar_count_per_channel == expected_count * 30
    assert np.allclose(normalizer.mean, expected.mean((0, 2)), atol=2e-13)
    assert np.allclose(normalizer.std, expected.std((0, 2), ddof=0), atol=2e-13)
    assert receipt.observation_count == expected_count
    assert receipt.normalizer_mean == tuple(normalizer.mean)
    assert receipt.normalizer_std == tuple(normalizer.std)
    assert len(receipt.sha256) == 64


def test_small_s1_direct_reference():
    population = generate_s1(3, 740620, PopulationRole.TRAINING)
    assert_reference(InverseTask.S1, (population,), 740620, 3)


def test_small_s2_superposition_direct_reference(monkeypatch):
    population = generate_s2(3, 740630, PopulationRole.TRAINING)
    real = api.PhysicsTMForward
    calls = {"evaluate": 0}

    class Spy:
        def __init__(self):
            self.physics = real()

        def evaluate(self, *args):
            calls["evaluate"] += 1
            return self.physics.evaluate(*args)

    monkeypatch.setattr(api, "PhysicsTMForward", Spy)
    assert_reference(InverseTask.S2, (population,), 740630, 3)
    assert calls["evaluate"] == 6


def test_small_s3_eight_stage_count_weighted_reference():
    populations = tuple(
        generate_s3(stage, 740640, stage, PopulationRole.TRAINING)
        for stage in range(1, 9)
    )
    assert_reference(InverseTask.S3, populations, 740640, 36)
    normalizer, receipt = api._fit_generated_training_populations(InverseTask.S3, populations, (740640,))
    assert normalizer.curriculum_stages == tuple(range(1, 9))
    assert receipt.curriculum_stages == tuple(range(1, 9))
    assert receipt.scalar_count_per_channel == 1080


def test_small_classifier_uses_all_five_source_counts():
    populations_by_count = generate_classifier(2, 740650, PopulationRole.TRAINING)
    populations = tuple(populations_by_count[source_count] for source_count in range(1, 6))
    normalizer, receipt = api._fit_generated_training_populations(
        InverseTask.CLASSIFIER, populations, (740650,)
    )
    assert normalizer.observation_count == 10
    assert receipt.source_counts == (1, 2, 3, 4, 5)


def test_public_workflow_has_task_only_signature():
    import inspect

    assert tuple(inspect.signature(fit_clean_training_normalizer).parameters) == ("task",)


@pytest.mark.parametrize(
    "task,expected",
    [
        (InverseTask.CLASSIFIER, ("classifier", 20261901, 14000, PopulationRole.TRAINING)),
        (InverseTask.S1, ("s1", 20261811, 70000, PopulationRole.TRAINING)),
        (InverseTask.S2, ("s2", 20262001, 70000, PopulationRole.TRAINING)),
        (InverseTask.S3, ("s3", 20262201, 70000, PopulationRole.TRAINING)),
    ],
)
def test_production_selection_is_frozen_training_only(monkeypatch, task, expected):
    calls = []
    classifier = generate_classifier(1, 741001, PopulationRole.TRAINING)
    s1 = generate_s1(1, 741002, PopulationRole.TRAINING)
    s2 = generate_s2(1, 741003, PopulationRole.TRAINING)
    s3 = {stage: generate_s3(1, 741004, stage, PopulationRole.TRAINING) for stage in range(1, 9)}

    def classifier_spy(per_class, seed, role):
        calls.append(("classifier", seed, per_class, role))
        return classifier

    def s1_spy(count, seed, role):
        calls.append(("s1", seed, count, role))
        return s1

    def s2_spy(count, seed, role):
        calls.append(("s2", seed, count, role))
        return s2

    def s3_spy(count, seed, stage, role):
        calls.append(("s3", seed, count, role))
        return s3[stage]

    monkeypatch.setattr(api, "generate_classifier", classifier_spy)
    monkeypatch.setattr(api, "generate_s1", s1_spy)
    monkeypatch.setattr(api, "generate_s2", s2_spy)
    monkeypatch.setattr(api, "generate_s3", s3_spy)
    populations, seeds = api._frozen_training_populations(task)
    assert seeds == (expected[1],)
    assert calls and all(call == expected for call in calls)
    assert len(populations) == (5 if task is InverseTask.CLASSIFIER else 8 if task is InverseTask.S3 else 1)


def test_fitting_constructs_no_measurement_noise_rng(monkeypatch):
    population = generate_s1(2, 740660, PopulationRole.TRAINING)
    calls = {"rng": 0}

    def forbidden(*args, **kwargs):
        calls["rng"] += 1
        raise AssertionError("measurement-noise RNG used during clean fitting")

    monkeypatch.setattr(np.random, "PCG64DXSM", forbidden)
    api._fit_generated_training_populations(InverseTask.S1, (population,), (740660,))
    assert calls["rng"] == 0


def test_normalizer_transform_is_non_mutating():
    values = np.arange(2 * 4 * 30, dtype=float).reshape(2, 4, 30) + 1
    before = values.copy()
    normalizer = FrozenChannelNormalizer(
        InverseTask.S1, values.mean((0, 2)), values.std((0, 2)), 2, 60
    )
    transformed = normalizer.transform(values)
    assert np.array_equal(values, before)
    assert np.allclose(transformed, (values - normalizer.mean[None, :, None]) / normalizer.std[None, :, None])
