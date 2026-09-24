"""S2 task adapters. Common population, noise, normalization and loss math is reused."""
from types import MappingProxyType
import hashlib
import math
import numpy as np
import torch

from inverse_em.config.s2 import S2Config, SeedRole, execution_seed
from inverse_em.config.phase2 import InverseTask
from inverse_em.errors import SchemaValidationError
from inverse_em import noise
from inverse_em.normalization import FrozenChannelNormalizer
from inverse_em.observations import ComplexFields, complex_fields_to_channels
from inverse_em.physics import Source, observation_angles
from inverse_em.populations import PopulationRole
from inverse_em.populations.generation import admissible as _admissible
from inverse_em.populations.contracts import SeparationConstraint
from .masking import scientific_view
from .losses import CanonicalTargets, localization_components, reduce_s2_canonical
from .field_consistency import compose_fc_objective
from .metrics import canonical_metrics

MAX_FIXTURE_CASES = 16


def validate_targets(targets, count):
    if (type(targets) is not CanonicalTargets or targets.active_count != 2
            or targets.rho.shape != (count, 2) or not 1 <= count <= MAX_FIXTURE_CASES):
        raise SchemaValidationError("Bounded canonical S2 targets required")
    targets.__post_init__()
    r = targets.rho.detach().numpy()
    p = targets.phi.detach().numpy()
    if (np.any((r < .05) | (r >= .95)) or np.any((p < 0) | (p >= 2*np.pi))
            or any(not np.array_equal(np.lexsort((ph, rh)), [0, 1])
                   or not _admissible(rh, ph, SeparationConstraint(.05, 5.))
                   for rh, ph in zip(r, p))):
        raise SchemaValidationError("Targets violate canonical conditioned-pair law")


def population_plan(role):
    if not isinstance(role, SeedRole) or role not in (SeedRole.TRAIN_POPULATION, SeedRole.VALIDATION_POPULATION):
        raise PermissionError("Only S2 train/validation plans exist")
    c = S2Config()
    return MappingProxyType({"count": c.train_count if role is SeedRole.TRAIN_POPULATION else c.validation_count,
            "seed": c.seed(role), "role": role.value, "source_count": 2, "production_execution_enabled": False})


def tiny_population(count, seed):
    if type(count) is not int or not 1 <= count <= MAX_FIXTURE_CASES:
        raise PermissionError("Phase 7 permits only tiny population fixtures")
    if type(seed) is not int or seed < 0 or seed in dict(S2Config().seed_roles).values():
        raise PermissionError("Fixture seed must not use a frozen S2 role")
    from inverse_em.populations import generate_s2
    return generate_s2(count, seed, PopulationRole.REFERENCE_FIXTURE)


def production_normalizer_entrypoint():
    """Reject the former executable surface during bounded Phase 7."""
    raise PermissionError("Production normalizer fitting is disabled in bounded Phase 7")


def production_normalizer_plan():
    """Immutable descriptive metadata only; not an execution credential."""
    return MappingProxyType({
        "api": "inverse_em.normalization.fit_clean_training_normalizer",
        "task": InverseTask.S2,
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
    if d.role is not PopulationRole.REFERENCE_FIXTURE or d.source_count != 2 or not 1 <= d.count <= MAX_FIXTURE_CASES:
        raise PermissionError("Only tiny S2 reference fixtures may execute here")
    if d.rng.seed in dict(S2Config().seed_roles).values():
        raise PermissionError("Production/reserved seeds cannot execute as fixtures")
    fields = []
    for radii, azimuths, amplitudes in zip(population.rho, population.phi, population.amplitude):
        electric = np.zeros(30, dtype=np.complex128)
        magnetic = np.zeros(30, dtype=np.complex128)
        for r, p, a in zip(radii, azimuths, amplitudes):
            pair = forward.evaluate(Source(float(r), float(p), float(a)), observation_angles(30))
            electric += pair.electric
            magnetic += pair.magnetic
        fields.append(ComplexFields(electric, magnetic))
    return tuple(fields)


def configuration_id(role, row, rho, phi, *, population_seed=None):
    """Historical canonical pair bytes; amplitude/study are NOT in the digest."""
    plan = population_plan(role)
    seed = plan["seed"] if population_seed is None else population_seed
    if type(seed) is not int or seed < 0:
        raise SchemaValidationError("Invalid population seed")
    if seed != plan["seed"] and seed in dict(S2Config().seed_roles).values():
        raise PermissionError("Population seed role mismatch")
    r, p = np.asarray(rho, dtype=np.float64), np.asarray(phi, dtype=np.float64)
    if (type(row) is not int or row < 0 or r.shape != (2,) or p.shape != (2,)
            or not np.isfinite(r).all() or not np.isfinite(p).all()
            or np.any((r < .05) | (r >= .95)) or np.any((p < 0) | (p >= 2*np.pi))
            or not np.array_equal(np.lexsort((p, r)), [0, 1])):
        raise SchemaValidationError("Canonical pair coordinates required")
    label = "train" if role is SeedRole.TRAIN_POPULATION else "validation"
    raw = np.stack((r, p), axis=1).astype("<f8").tobytes()
    prefix = f"{seed}|{row}".encode("utf-8")
    return "S2ANF-" + label + "-" + hashlib.sha256(prefix + raw).hexdigest()[:32]


def keyed_seed(role, epoch, identity, stream):
    expected = SeedRole.MINIBATCH_ORDER if stream == "ORDER" else SeedRole.TRAINING_AUGMENTATION
    master = execution_seed(role, expected)
    if type(epoch) is not int or not 1 <= epoch <= 600:
        raise SchemaValidationError("S2 epochs are one-based, within the fixed budget")
    if stream not in ("SNR", "E", "H", "ORDER") or not isinstance(identity, str) or not identity.strip():
        raise SchemaValidationError("Invalid keyed stream identity")
    if stream == "ORDER" and identity != "ALL":
        raise SchemaValidationError("Order stream uses ALL")
    key = f"{S2Config().namespace}|{master}|{epoch}|{identity}|{stream}"
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
    if type(count) is not int or count < 1 or count > MAX_FIXTURE_CASES:
        raise SchemaValidationError("Invalid S2 order count")
    order = _rng(role, epoch, "ALL", "ORDER").permutation(count)
    return tuple(order[start:start+128] for start in range(0, count, 128))


def _inputs(fields, normalizer):
    if type(normalizer) is not FrozenChannelNormalizer or normalizer.task is not InverseTask.S2:
        raise SchemaValidationError("S2 requires its frozen training normalizer")
    if not 1 <= len(fields) <= MAX_FIXTURE_CASES or any(type(f) is not ComplexFields for f in fields):
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
    validate_targets(targets, len(normalized))
    prediction = scientific_view(model(normalized), 2)
    supervised = reduce_s2_canonical(localization_components(prediction, targets))
    kendall, channels = fc(prediction, clean_raw)
    return compose_fc_objective(supervised, kendall), supervised, kendall, channels


def validate_clean(model, fields, targets, normalizer):
    validate_targets(targets, len(fields))
    from .decoding import ScientificPrediction
    model.eval()
    x = clean_tensor(fields, normalizer)
    parts = []
    with torch.no_grad():
        for first in range(0, len(fields), 256):
            parts.append(scientific_view(model(x[first:first+256]), 2))
        prediction = ScientificPrediction(*(torch.cat([getattr(p, key) for p in parts])
            for key in ("rho", "cos_like", "sin_like", "phi")))
        return canonical_metrics(prediction, targets)


def production_training_plan():
    from inverse_em.config.s2 import budget
    return MappingProxyType({**budget(), "production_execution_enabled": False})


def production_training_entrypoint(*args, **kwargs):
    raise PermissionError("Production S2 training is unavailable")


def protected_evaluation(*args, **kwargs):
    raise PermissionError("Sealed and robustness execution are unavailable in Phase 7")
