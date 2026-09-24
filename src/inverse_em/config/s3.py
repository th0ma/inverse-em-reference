"""Frozen production descriptors, separate from bounded fixture execution."""
from dataclasses import dataclass, fields
from enum import Enum
from inverse_em.errors import SchemaValidationError
from inverse_em.provenance.canonical import canonical_sha256


class SeedRole(str, Enum):
    TRAIN_POPULATION = "TRAIN_POPULATION"
    VALIDATION_POPULATION = "VALIDATION_POPULATION"
    MODEL_INITIALIZATION = "MODEL_INITIALIZATION"
    MINIBATCH_ORDER = "MINIBATCH_ORDER"
    TRAINING_AUGMENTATION = "TRAINING_AUGMENTATION"
    SEALED_RESERVED = "SEALED_RESERVED"
    ROBUSTNESS_RESERVED = "ROBUSTNESS_RESERVED"


@dataclass(frozen=True)
class S3Config:
    task: str = "S3"
    namespace: str = "s3_analytical_noise_final_v1"
    curriculum: tuple = ((1, .10, 40.), (2, .08, 25.), (3, .08, 20.),
                         (4, .07, 15.), (5, .06, 10.), (6, .05, 8.),
                         (7, .05, 6.), (8, .05, 5.))
    seed_roles: tuple = tuple((r.value, 20262201+i) for i, r in enumerate(SeedRole))
    source_count: int = 3
    amplitude: float = 1.0
    radius_interval: tuple = (.05, .95)
    population_law: str = "phase1_area_uniform_whole_triple_canonical_deduplicated"
    threshold_ulps: int = 8
    train_count_per_stage: int = 70000
    validation_count: int = 15000
    validation_stage: int = 8
    epochs_per_stage: int = 50
    batch_size: int = 128
    validation_batch_size: int = 256
    drop_last: bool = False
    early_stopping: bool = False
    angles: int = 30
    active_slots: tuple = (0, 1, 2)
    snr_interval: tuple = (30., 40.)
    independent_field_snr: bool = True
    exposure: int = 0
    rng_convention: str = "study_master_stage_globalepoch_exposure_identity_stream_sha256_first128le_pcg64dxsm"
    normalizer: str = "closed_phase2_30_angle_block_moments_all8_clean_train_ddof0_no_epsilon"
    loss_coefficients: tuple = (1., 2., .1, .03)
    outer_constant: float = 1.635
    circle_weighted: bool = False
    reduction: str = "mean_batch_source"
    optimizer: str = "Adam"
    lr: float = .0005
    betas: tuple = (.9, .999)
    eps: float = 1e-8
    weight_decay: float = 1e-5
    reset_optimizer_each_stage: bool = True
    scheduler: str = "CosineAnnealingWarmRestarts"
    T_0: int = 200
    T_mult: int = 2
    eta_min: float = 1e-6
    reset_scheduler_each_stage: bool = True
    epoch_order: str = "validation_best_history_scheduler_localepoch_continuation"
    selection: str = "global_strict_lower_torch_sqrt_mean_squared_canonical_errors"
    dtype: str = "float64"
    device: str = "cpu"
    phase1_sha256: str = "fcf09ece12dcd75c99af55a5bf0a281e43d7965d1761b2c40d914e4b62a99bcd"
    phase2_sha256: str = "04ddcf80a56f37eae29fc42befa1fc77713360e2d314e38d8cd6fb8f2a06102b"
    phase5_sha256: str = "8303dc42fce7f495bd882b07573948861bd7a35025a141b2224be6e21dd6c363"
    electric_sha256: str = "d3a274f0ab4d514cdbb6b5edb32f3976b9926bb82e838e0277ba0f69f5ffa80b"
    magnetic_sha256: str = "0285679d940bb20ecbea334e0815018bf7c8aa6b7210d5e2ce3a2a57a053173c"

    def __post_init__(self):
        for f in fields(self):
            if canonical_sha256(getattr(self, f.name)) != canonical_sha256(f.default):
                raise SchemaValidationError("Frozen S3 setting changed: " + f.name)

    @property
    def sha256(self):
        return canonical_sha256(self)

    def seed(self, role):
        if not isinstance(role, SeedRole):
            raise SchemaValidationError("Explicit S3 seed role required")
        return dict(self.seed_roles)[role.value]


def execution_seed(role, expected):
    if not isinstance(role, SeedRole) or role not in (SeedRole.MODEL_INITIALIZATION,
            SeedRole.MINIBATCH_ORDER, SeedRole.TRAINING_AUGMENTATION):
        raise PermissionError("Protected S3 execution role")
    if role is not expected:
        raise SchemaValidationError("S3 seed-role mismatch")
    return S3Config().seed(role)


def budget():
    c = S3Config()
    full, tail = divmod(c.train_count_per_stage, c.batch_size)
    updates = full + bool(tail)
    return {"full_batches": full, "tail": tail, "updates_per_epoch": updates,
            "updates_per_stage": updates*c.epochs_per_stage, "epochs": 400,
            "total_updates": updates*c.epochs_per_stage*8, "training_cases": 560000}
