from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import numpy as np
import torch

from inverse_em.config.classifier import CLASSIFIER_COUNTS
from inverse_em.config.phase2 import InverseTask
from inverse_em.errors import SchemaValidationError
from inverse_em.noise import TrainingNoiseStream, add_scaled_noise, scale_directions
from inverse_em.normalization import FrozenChannelNormalizer
from inverse_em.observations import ComplexFields, complex_fields_to_channels
from inverse_em.physics import PhysicsTMForward, Source, observation_angles
from inverse_em.populations import GeneratedPopulation, PopulationRole


def classifier_label(source_count: int) -> int:
    if source_count not in CLASSIFIER_COUNTS:
        raise SchemaValidationError("Classifier source count must be 1..5")
    return source_count - 1


def validate_population_mapping(populations: Mapping[int, GeneratedPopulation], role: PopulationRole):
    if role not in (PopulationRole.TRAINING, PopulationRole.VALIDATION):
        raise PermissionError("Phase-4 implementation accepts TRAINING or VALIDATION populations only")
    if tuple(sorted(populations)) != CLASSIFIER_COUNTS:
        raise SchemaValidationError("Classifier population must contain every source count 1..5")
    expected_seed = {PopulationRole.TRAINING: 20261901, PopulationRole.VALIDATION: 20261902}[role]
    for source_count, population in populations.items():
        if type(population) is not GeneratedPopulation or population.definition.role is not role:
            raise SchemaValidationError("Population role mismatch")
        if population.definition.population_id != "classifier_final" or population.definition.source_count != source_count:
            raise SchemaValidationError("Population classifier identity mismatch")
        if population.definition.rng.seed != expected_seed:
            raise PermissionError("Classifier population role/seed identity mismatch")


def _validate_classifier_normalizer(normalizer):
    if not isinstance(normalizer, FrozenChannelNormalizer):
        raise SchemaValidationError("Classifier observations require a frozen channel normalizer")
    if normalizer.task is not InverseTask.CLASSIFIER:
        raise SchemaValidationError("Classifier observations require the CLASSIFIER normalizer")


def analytical_observations(populations: Mapping[int, GeneratedPopulation], role: PopulationRole):
    """Evaluate already-authorized populations; this function never generates a population."""
    validate_population_mapping(populations, role)
    physics = PhysicsTMForward(); angles = observation_angles(30)
    fields, labels = [], []
    for source_count in CLASSIFIER_COUNTS:
        population = populations[source_count]
        for row in range(population.definition.count):
            electric = np.zeros(30, dtype=np.complex128); magnetic = np.zeros(30, dtype=np.complex128)
            for rho, phi, amplitude in zip(population.rho[row], population.phi[row], population.amplitude[row]):
                current = physics.evaluate(Source(float(rho), float(phi), float(amplitude)), angles)
                electric += current.electric; magnetic += current.magnetic
            fields.append(ComplexFields(electric, magnetic)); labels.append(classifier_label(source_count))
    return tuple(fields), torch.tensor(labels, dtype=torch.int64, device="cpu")


def clean_observation_tensor(fields: Sequence[ComplexFields], normalizer: FrozenChannelNormalizer) -> torch.Tensor:
    _validate_classifier_normalizer(normalizer)
    if not fields or any(type(field) is not ComplexFields for field in fields):
        raise SchemaValidationError("Clean classifier observations require ComplexFields")
    raw = np.stack([complex_fields_to_channels(field) for field in fields])
    return torch.from_numpy(np.ascontiguousarray(normalizer.transform(raw))).to(dtype=torch.float64, device="cpu")


def noisy_observation_tensor(fields: Sequence[ComplexFields], normalizer: FrozenChannelNormalizer,
                             stream: TrainingNoiseStream, event_offset: int = 0):
    _validate_classifier_normalizer(normalizer)
    if not fields or any(type(field) is not ComplexFields for field in fields):
        raise SchemaValidationError("Noisy classifier observations require clean ComplexFields")
    if not isinstance(stream, TrainingNoiseStream) or type(event_offset) is not int or event_offset < 0:
        raise SchemaValidationError("Invalid training-noise stream or event offset")
    noisy, receipts = [], []
    for index, field in enumerate(fields):
        event_index = event_offset + index
        snr_db, directions = stream.draw_event(event_index, 0)
        noisy.append(complex_fields_to_channels(add_scaled_noise(field, scale_directions(field, directions, snr_db))))
        receipts.append((event_index, snr_db))
    values = normalizer.transform(np.stack(noisy))
    return torch.from_numpy(np.ascontiguousarray(values)).to(dtype=torch.float64, device="cpu"), tuple(receipts)


def epoch_permutation(count: int, epoch: int, seed: int) -> np.ndarray:
    if any(type(value) is not int or value < 1 for value in (count, epoch)) or type(seed) is not int or seed < 0:
        raise SchemaValidationError("Invalid minibatch-order identity")
    key = f"classification_analytical_noise_aware_final_v1|{seed}|{epoch}|ALL|ORDER"
    derived = int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], "little")
    return np.random.Generator(np.random.PCG64DXSM(derived)).permutation(count)


def epoch_minibatches(count: int, epoch: int, seed: int, batch_size: int = 128):
    if type(batch_size) is not int or batch_size < 1:
        raise SchemaValidationError("batch_size must be positive")
    order = epoch_permutation(count, epoch, seed)
    return tuple(order[start:start + batch_size] for start in range(0, count, batch_size))
