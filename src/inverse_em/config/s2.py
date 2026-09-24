"""Frozen S2 science and descriptive seed roles; no execution on import."""
from dataclasses import dataclass, fields
from enum import Enum
from inverse_em.config.localization import LocalizationScientificConfig
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
class S2Config:
    task: str = "S2"
    train_count: int = 70000
    validation_count: int = 15000
    source_count: int = 2
    amplitude: float = 1.0
    radius_interval: tuple = (0.05, 0.95)
    radial_law: str = "area_uniform_upper_excluded"
    azimuth_law: str = "uniform_[0,2pi)"
    angles: int = 30
    active_slots: int = 2
    seed_roles: tuple = tuple((role.value, 20262001 + i) for i, role in enumerate(SeedRole))
    namespace: str = "s2_analytical_noise_final_v1"
    rng_convention: str = "historical_sha256_first128le_pcg64dxsm_v1"
    snr_interval: tuple = (30.0, 40.0)
    batch_size: int = 128
    validation_batch_size: int = 256
    epochs: int = 600
    early_stopping: bool = True
    patience: int = 200
    material_test: str = "score < es_reference - 1e-5"
    minimum_radial_separation: float = 0.05
    minimum_angular_degrees: float = 5.0
    threshold_ulps: int = 8
    source_reduction: str = "sum_then_batch_mean"
    epoch_order: str = "validation_best_material_history_scheduler_stop_v1"
    optimizer: str = "Adam"
    lr: float = 0.0005
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-5
    scheduler: str = "CosineAnnealingWarmRestarts"
    T_0: int = 200
    T_mult: int = 2
    eta_min: float = 1e-6
    loss_coefficients: tuple = (1.0, 2.0, 0.1, 0.03)
    selection: str = "clean_cartesian_rmse_strict_lower_earliest_tie"
    dtype: str = "float64"
    device: str = "cpu"
    amp: bool = False
    phase5_sha256: str = "8303dc42fce7f495bd882b07573948861bd7a35025a141b2224be6e21dd6c363"
    electric_sha256: str = "d3a274f0ab4d514cdbb6b5edb32f3976b9926bb82e838e0277ba0f69f5ffa80b"
    magnetic_sha256: str = "0285679d940bb20ecbea334e0815018bf7c8aa6b7210d5e2ce3a2a57a053173c"

    def __post_init__(self):
        for field in fields(self):
            if canonical_sha256(getattr(self, field.name)) != canonical_sha256(field.default):
                raise SchemaValidationError(f"Frozen S2 setting changed: {field.name}")
        if LocalizationScientificConfig().sha256 != self.phase5_sha256:
            raise SchemaValidationError("Phase-5 configuration mismatch")

    @property
    def sha256(self):
        return canonical_sha256(self)

    def seed(self, role: SeedRole):
        if not isinstance(role, SeedRole):
            raise SchemaValidationError("An explicit S2 seed role is required")
        return dict(self.seed_roles)[role.value]


def execution_seed(role: SeedRole, expected: SeedRole) -> int:
    if not isinstance(role, SeedRole) or role not in (SeedRole.MODEL_INITIALIZATION, SeedRole.MINIBATCH_ORDER, SeedRole.TRAINING_AUGMENTATION):
        raise PermissionError("Protected or invalid S2 execution role")
    if role is not expected:
        raise SchemaValidationError("S2 seed role mismatch")
    return S2Config().seed(role)


def budget():
    c = S2Config()
    full, tail = divmod(c.train_count, c.batch_size)
    per_epoch = full + bool(tail)
    return {"full_batches": full, "tail": tail, "updates_per_epoch": per_epoch,
            "epochs": c.epochs, "total_updates": per_epoch * c.epochs}
