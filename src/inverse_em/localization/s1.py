"""S1 task adapters. Common population, noise, normalization and loss math is reused."""
from types import MappingProxyType
import hashlib
import math
import numpy as np
import torch

from inverse_em.config.s1 import S1Config, SeedRole, execution_seed
from inverse_em.config.phase2 import InverseTask
from inverse_em.errors import SchemaValidationError
from inverse_em import noise
from inverse_em.normalization import FrozenChannelNormalizer
from inverse_em.observations import ComplexFields, complex_fields_to_channels
from inverse_em.physics import Source, observation_angles
from inverse_em.populations import PopulationRole, generate_s1
from .masking import scientific_view
from .losses import CanonicalTargets, localization_components, reduce_s1
from .field_consistency import compose_fc_objective
from .metrics import canonical_metrics

MAX_FIXTURE_CASES = 16


def population_plan(role):
    if not isinstance(role, SeedRole) or role not in (SeedRole.TRAIN_POPULATION, SeedRole.VALIDATION_POPULATION):
        raise PermissionError("Only S1 train/validation plans exist")
    c = S1Config()
    return {"count": c.train_count if role is SeedRole.TRAIN_POPULATION else c.validation_count,
            "seed": execution_seed(role, role), "role": role.value, "source_count": 1}


def tiny_population(count, seed):
    if type(count) is not int or not 1 <= count <= MAX_FIXTURE_CASES:
        raise PermissionError("Phase 6 permits only tiny population fixtures")
    if type(seed) is not int or seed < 0 or seed in dict(S1Config().seed_roles).values():
        raise PermissionError("Fixture seed must not use a frozen S1 role")
    return generate_s1(count, seed, PopulationRole.REFERENCE_FIXTURE)


def production_normalizer_entrypoint():
    """Reject the former executable surface during bounded Phase 6."""
    raise PermissionError("Production normalizer fitting is disabled in bounded Phase 6")


def production_normalizer_plan():
    """Immutable descriptive metadata only; not an execution credential."""
    return MappingProxyType({
        "api": "inverse_em.normalization.fit_clean_training_normalizer",
        "task": InverseTask.S1,
        "inputs": "clean analytical training observations only",
        "training_count": 70000,
        "angles": 30,
        "channels": ("Re(E)", "Im(E)", "Re(H)", "Im(H)"),
        "statistics": ("mean", "population standard deviation"),
        "ddof": 0,
        "production_execution_enabled": False,
    })


def analytical_fixture(population, forward):
    """Explicit tiny-fixture seam; forward implements the Phase-1 evaluate interface."""
    d = population.definition
    if d.role is not PopulationRole.REFERENCE_FIXTURE or d.source_count != 1 or not 1 <= d.count <= MAX_FIXTURE_CASES:
        raise PermissionError("Only tiny S1 reference fixtures may execute here")
    if d.rng.seed in dict(S1Config().seed_roles).values():
        raise PermissionError("Production/reserved seeds cannot execute as fixtures")
    fields = []
    for r, p, a in zip(population.rho[:, 0], population.phi[:, 0], population.amplitude[:, 0]):
        pair = forward.evaluate(Source(float(r), float(p), float(a)), observation_angles(30))
        fields.append(ComplexFields(pair.electric, pair.magnetic))
    return tuple(fields)


def configuration_id(role, row, rho, phi, *, population_seed=None):
    """Historical UTF-8 prefix + little-endian float64 source-triple recipe."""
    plan = population_plan(role)
    seed = plan["seed"] if population_seed is None else population_seed
    if type(seed) is not int or seed < 0:
        raise SchemaValidationError("Invalid population seed")
    if seed != plan["seed"] and seed in dict(S1Config().seed_roles).values():
        raise PermissionError("Population seed role mismatch")
    if type(row) is not int or row < 0 or not (.05 <= rho < .95 and 0 <= phi < 2*math.pi):
        raise SchemaValidationError("Invalid historical configuration identity")
    label = "train" if role is SeedRole.TRAIN_POPULATION else "validation"
    triple = np.array([rho, phi, 1.], dtype="<f8").tobytes()
    prefix = f"{S1Config().namespace}|{seed}|{label}|{row}".encode("utf-8")
    return "S1FINAL-" + label + "-" + hashlib.sha256(prefix + triple).hexdigest()[:32]


def keyed_seed(role, epoch, identity, stream):
    expected = SeedRole.MINIBATCH_ORDER if stream == "ORDER" else SeedRole.TRAINING_AUGMENTATION
    master = execution_seed(role, expected)
    if type(epoch) is not int or not 1 <= epoch <= 200:
        raise SchemaValidationError("S1 epochs are one-based, within the fixed budget")
    if stream not in ("SNR", "E", "H", "ORDER") or not isinstance(identity, str) or not identity.strip():
        raise SchemaValidationError("Invalid keyed stream identity")
    if stream == "ORDER" and identity != "ALL":
        raise SchemaValidationError("Order stream uses ALL")
    key = f"{S1Config().namespace}|{master}|{epoch}|{identity}|{stream}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:16], "little")


def _rng(role, epoch, identity, stream):
    return np.random.Generator(np.random.PCG64DXSM(keyed_seed(role, epoch, identity, stream)))


def noise_event(identity, epoch, sample_index=0, role=SeedRole.TRAINING_AUGMENTATION):
    gamma = float(_rng(role, epoch, identity, "SNR").uniform(30, 40))
    electric = _rng(role, epoch, identity, "E").standard_normal((30, 2))
    magnetic = _rng(role, epoch, identity, "H").standard_normal((30, 2))
    directions = noise.StandardizedNoiseDirections(electric[:, 0], electric[:, 1],
        magnetic[:, 0], magnetic[:, 1], execution_seed(role, SeedRole.TRAINING_AUGMENTATION), sample_index, epoch)
    return gamma, directions


def epoch_minibatches(count, epoch, role=SeedRole.MINIBATCH_ORDER):
    if type(count) is not int or count < 1 or count > S1Config().train_count:
        raise SchemaValidationError("Invalid S1 order count")
    order = _rng(role, epoch, "ALL", "ORDER").permutation(count)
    return tuple(order[start:start+128] for start in range(0, count, 128))


def _inputs(fields, normalizer):
    if type(normalizer) is not FrozenChannelNormalizer or normalizer.task is not InverseTask.S1:
        raise SchemaValidationError("S1 requires its frozen training normalizer")
    if not fields or any(type(f) is not ComplexFields for f in fields):
        raise SchemaValidationError("Expected clean analytical complex fields")


def clean_tensor(fields, normalizer):
    _inputs(fields, normalizer)
    raw = np.stack([complex_fields_to_channels(f) for f in fields])
    return torch.from_numpy(normalizer.transform(raw))


def training_tensors(fields, ids, normalizer, epoch):
    _inputs(fields, normalizer)
    if len(ids) != len(fields) or len(set(ids)) != len(ids):
        raise SchemaValidationError("Unique configuration identities required")
    clean, noisy = [], []
    for i, (field, identity) in enumerate(zip(fields, ids)):
        gamma, directions = noise_event(identity, epoch, i)
        scaled = noise.scale_directions(field, directions, gamma)
        noisy.append(complex_fields_to_channels(noise.add_scaled_noise(field, scaled)))
        clean.append(complex_fields_to_channels(field))
    return torch.from_numpy(normalizer.transform(np.stack(noisy))), torch.from_numpy(np.stack(clean))


def objective(model, normalized, targets, clean_raw, fc):
    if not isinstance(targets, CanonicalTargets) or targets.active_count != 1:
        raise SchemaValidationError("S1 objective requires exactly one target slot")
    prediction = scientific_view(model(normalized), 1)
    supervised = reduce_s1(localization_components(prediction, targets))
    kendall, channels = fc(prediction, clean_raw)
    return compose_fc_objective(supervised, kendall), supervised, kendall, channels


def validate_clean(model, fields, targets, normalizer):
    if targets.active_count != 1 or targets.rho.shape[0] != len(fields):
        raise SchemaValidationError("S1 validation shape mismatch")
    from .decoding import ScientificPrediction
    model.eval()
    x = clean_tensor(fields, normalizer)
    parts = []
    with torch.no_grad():
        for first in range(0, len(fields), 256):
            parts.append(scientific_view(model(x[first:first+256]), 1))
        prediction = ScientificPrediction(*(torch.cat([getattr(p, key) for p in parts])
            for key in ("rho", "cos_like", "sin_like", "phi")))
        return canonical_metrics(prediction, targets)
