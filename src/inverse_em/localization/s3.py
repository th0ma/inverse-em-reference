"""Bounded S3 adapters; no unrestricted generator or production fitter alias."""
from types import MappingProxyType
import hashlib
import numpy as np
import torch
from inverse_em import noise
from inverse_em.config.s3 import S3Config, SeedRole, execution_seed
from inverse_em.config.phase2 import InverseTask
from inverse_em.errors import SchemaValidationError
from inverse_em.normalization import FrozenChannelNormalizer
from inverse_em.observations import ComplexFields, complex_fields_to_channels
from inverse_em.populations import PopulationRole
from inverse_em.localization.losses import CanonicalTargets, localization_components, reduce_outer_s3
from inverse_em.localization.masking import scientific_view
from inverse_em.localization.field_consistency import compose_fc_objective
from inverse_em.localization.metrics import canonical_errors, canonical_metrics

MAX_FIXTURE_CASES = 16


def stage_contract(stage):
    if type(stage) is not int or not 1 <= stage <= 8:
        raise SchemaValidationError("S3 stage must be 1..8")
    return S3Config().curriculum[stage-1]


def stage_population_seed(master, stage):
    stage_contract(stage)
    if type(master) is not int or master < 0:
        raise SchemaValidationError("Invalid population master")
    # Pure metadata derivation, not a population execution capability.
    from inverse_em.populations.generation import derived_s3_stage_seed
    return derived_s3_stage_seed(master, stage)


def tiny_population(count, seed, stage, role=PopulationRole.REFERENCE_FIXTURE):
    if role is not PopulationRole.REFERENCE_FIXTURE:
        raise PermissionError("Only reference-fixture generation is executable")
    if type(count) is not int or not 1 <= count <= MAX_FIXTURE_CASES:
        raise PermissionError("At most 16 fixture configurations")
    if type(seed) is not int or seed < 0 or seed in dict(S3Config().seed_roles).values():
        raise PermissionError("Frozen S3 identities cannot generate fixtures")
    stage_contract(stage)
    if seed in tuple(stage_population_seed(20262201, s) for s in range(1, 9)):
        raise PermissionError("Production derived seeds cannot generate fixtures")
    from inverse_em.populations import generate_s3
    return generate_s3(count, seed, stage, role)


def population_plan(role, stage=8):
    stage_contract(stage)
    if role not in (SeedRole.TRAIN_POPULATION, SeedRole.VALIDATION_POPULATION) or not isinstance(role, SeedRole):
        raise PermissionError("No protected population plans")
    if role is SeedRole.VALIDATION_POPULATION and stage != 8:
        raise SchemaValidationError("Validation uses Stage 8")
    seed = S3Config().seed(role)
    return MappingProxyType({"role": role.value, "stage": stage,
        "count": 70000 if role is SeedRole.TRAIN_POPULATION else 15000,
        "master_seed": seed, "population_seed": stage_population_seed(seed, stage)
        if role is SeedRole.TRAIN_POPULATION else seed, "production_execution_enabled": False})


def production_normalizer_plan():
    return MappingProxyType({"api": "inverse_em.normalization.fit_clean_training_normalizer",
        "task": "S3", "stages": tuple(range(1, 9)), "cases": 560000,
        "scalar_count_per_channel": 16800000, "ddof": 0, "clean_training_only": True,
        "accumulation": "closed_phase2_30_angle_block_moments", "epsilon": None,
        "production_execution_enabled": False})


def production_normalizer_entrypoint(*args, **kwargs):
    raise PermissionError("Production S3 normalization is unavailable")


def analytical_fixture(population, forward):
    """Bounded superposition seam using the existing Phase-1 forward interface."""
    d = population.definition
    if d.role is not PopulationRole.REFERENCE_FIXTURE or d.source_count != 3 or not 1 <= d.count <= MAX_FIXTURE_CASES:
        raise PermissionError("Only tiny three-source reference fixtures")
    protected = tuple(dict(S3Config().seed_roles).values()) + tuple(stage_population_seed(20262201, i) for i in range(1, 9))
    if d.rng.seed in protected:
        raise PermissionError("Protected analytical population")
    from inverse_em.physics import Source, observation_angles
    theta = observation_angles(30)
    result = []
    for r, p, a in zip(population.rho, population.phi, population.amplitude):
        electric, magnetic = np.zeros(30, np.complex128), np.zeros(30, np.complex128)
        for rho, phi, amplitude in zip(r, p, a):
            pair = forward.evaluate(Source(float(rho), float(phi), float(amplitude)), theta)
            electric += pair.electric
            magnetic += pair.magnetic
        result.append(ComplexFields(electric, magnetic))
    return tuple(result)


def production_training_entrypoint(*args, **kwargs):
    raise PermissionError("Production S3 training is unavailable")


def protected_evaluation(*args, **kwargs):
    raise PermissionError("Sealed/robustness execution is unavailable")


def validate_targets(targets, count, stage):
    _, dr, dp = stage_contract(stage)
    if type(targets) is not CanonicalTargets or targets.rho.shape != (count, 3) or not 1 <= count <= MAX_FIXTURE_CASES:
        raise SchemaValidationError("Tiny canonical three-source targets required")
    targets.__post_init__()
    from inverse_em.populations.generation import admissible
    from inverse_em.populations.contracts import SeparationConstraint
    r, p = targets.rho.detach().numpy(), targets.phi.detach().numpy()
    if (np.any((r < .05) | (r >= .95)) or np.any((p < 0) | (p >= 2*np.pi))
            or any(not np.array_equal(np.lexsort((ph, rh)), [0, 1, 2])
                   or not admissible(rh, ph, SeparationConstraint(dr, dp)) for rh, ph in zip(r, p))):
        raise SchemaValidationError("Targets violate S3 canonical stage law")


def configuration_id(role, stage, rho, phi):
    if role not in (SeedRole.TRAIN_POPULATION, SeedRole.VALIDATION_POPULATION) or not isinstance(role, SeedRole):
        raise PermissionError("Only train/validation configuration IDs")
    stage_contract(stage)
    r, p = np.asarray(rho, np.float64), np.asarray(phi, np.float64)
    if (r.shape != (3,) or p.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(p).all()
            or np.any((r < .05) | (r >= .95)) or np.any((p < 0) | (p >= 2*np.pi))
            or not np.array_equal(np.lexsort((p, r)), [0, 1, 2])):
        raise SchemaValidationError("Canonical S3 coordinates required")
    label = "train" if role is SeedRole.TRAIN_POPULATION else "validation"
    h = hashlib.sha256(np.stack((r, p), axis=1).astype("<f8").tobytes()).hexdigest()
    return f"S3ANF-{label}-ST{stage:02d}-{h[:24]}"


def keyed_seed(role, stage, epoch, exposure, identity, stream):
    expected = SeedRole.MINIBATCH_ORDER if stream == "ORDER" else SeedRole.TRAINING_AUGMENTATION
    master = execution_seed(role, expected)
    stage_contract(stage)
    if (type(epoch) is not int or not 1 <= epoch <= 400 or type(exposure) is not int or exposure < 0
            or not isinstance(identity, str) or not identity.strip()
            or stream not in ("ORDER", "E_SNR", "E_NOISE", "H_SNR", "H_NOISE")
            or (stream == "ORDER" and (identity != "ALL" or exposure != 0))):
        raise SchemaValidationError("Invalid S3 keyed address")
    key = f"{S3Config().namespace}|{master}|{stage}|{epoch}|{exposure}|{identity}|{stream}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:16], "little")


def _rng(role, stage, epoch, exposure, identity, stream):
    return np.random.Generator(np.random.PCG64DXSM(keyed_seed(role, stage, epoch, exposure, identity, stream)))


def epoch_minibatches(count, stage, epoch):
    if type(count) is not int or not 1 <= count <= MAX_FIXTURE_CASES:
        raise PermissionError("Only tiny ordering fixtures")
    order = _rng(SeedRole.MINIBATCH_ORDER, stage, epoch, 0, "ALL", "ORDER").permutation(count)
    return tuple(order[i:i+128] for i in range(0, count, 128))


def augment_field(fields, identity, stage, epoch, exposure=0):
    if type(fields) is not ComplexFields:
        raise SchemaValidationError("Clean complex field required")
    outputs, gammas = [], []
    role = SeedRole.TRAINING_AUGMENTATION
    zeros = np.zeros(30, np.float64)
    for label in ("E", "H"):
        gamma = float(_rng(role, stage, epoch, exposure, identity, label+"_SNR").uniform(30, 40))
        z = _rng(role, stage, epoch, exposure, identity, label+"_NOISE").standard_normal((30, 2))
        dirs = (z[:, 0], z[:, 1], zeros, zeros) if label == "E" else (zeros, zeros, z[:, 0], z[:, 1])
        directions = noise.StandardizedNoiseDirections(*dirs, S3Config().seed(role), 0, exposure)
        noisy = noise.add_scaled_noise(fields, noise.scale_directions(fields, directions, gamma))
        outputs.append(noisy.electric if label == "E" else noisy.magnetic)
        gammas.append(gamma)
    return np.stack((outputs[0].real, outputs[0].imag, outputs[1].real, outputs[1].imag)), tuple(gammas)


def _inputs(fields, normalizer):
    if type(normalizer) is not FrozenChannelNormalizer or normalizer.task is not InverseTask.S3:
        raise SchemaValidationError("Frozen Phase-2 S3 normalizer required")
    if not 1 <= len(fields) <= MAX_FIXTURE_CASES or any(type(f) is not ComplexFields for f in fields):
        raise PermissionError("Only tiny clean fields")


def training_tensors(fields, ids, normalizer, stage, epoch):
    _inputs(fields, normalizer)
    if len(ids) != len(fields) or len(set(ids)) != len(ids):
        raise SchemaValidationError("Unique IDs required")
    pairs = [augment_field(f, identity, stage, epoch) for f, identity in zip(fields, ids)]
    raw = np.stack([p[0] for p in pairs])
    clean = np.stack([complex_fields_to_channels(f) for f in fields])
    receipt = {"gammas_E_H": tuple(p[1] for p in pairs), "noise_sha256": hashlib.sha256(raw.tobytes()).hexdigest()}
    return torch.from_numpy(normalizer.transform(raw)), torch.from_numpy(clean), receipt


def objective(model, x, targets, clean_raw, fc):
    if type(targets) is not CanonicalTargets or targets.active_count != 3:
        raise SchemaValidationError("Three canonical slots required")
    prediction = scientific_view(model(x), 3)
    loc = reduce_outer_s3(localization_components(prediction, targets), targets)
    k, channels = fc(prediction, clean_raw)
    return compose_fc_objective(loc, k), loc, k, channels


def validate_clean(model, fields, targets, normalizer):
    _inputs(fields, normalizer)
    validate_targets(targets, len(fields), 8)
    from inverse_em.localization.decoding import ScientificPrediction
    raw = np.stack([complex_fields_to_channels(f) for f in fields])
    x = torch.from_numpy(normalizer.transform(raw))
    model.eval()
    with torch.no_grad():
        parts = [scientific_view(model(x[i:i+256]), 3) for i in range(0, len(x), 256)]
        pred = ScientificPrediction(*(torch.cat([getattr(p, key) for p in parts])
            for key in ("rho", "cos_like", "sin_like", "phi")))
        score = float(torch.sqrt(canonical_errors(pred, targets)["cartesian"].square().mean()))
        metrics = canonical_metrics(pred, targets)
    return score, metrics
