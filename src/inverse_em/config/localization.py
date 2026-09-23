from __future__ import annotations

from dataclasses import asdict, dataclass

from inverse_em.errors import SchemaValidationError
from inverse_em.provenance.canonical import canonical_sha256
from inverse_em.config.surrogate import HISTORICAL_CHECKPOINT_SHA256


@dataclass(frozen=True)
class LocalizationModelConfig:
    input_channels: int = 4
    angle_count: int = 30
    convolution_channels: tuple[int, int, int] = (32, 64, 96)
    convolution_kernels: tuple[int, int, int] = (5, 5, 3)
    convolution_padding: tuple[int, int, int] = (2, 2, 1)
    dense_input: int = 2880
    dense_hidden: int = 128
    source_slots: int = 3
    negative_slope: float = 0.01
    dtype: str = "float64"
    device: str = "cpu"

    def __post_init__(self) -> None:
        expected = (4, 30, (32, 64, 96), (5, 5, 3), (2, 2, 1), 2880, 128, 3, 0.01, "float64", "cpu")
        if tuple(asdict(self).values()) != expected:
            raise SchemaValidationError("Localization architecture differs from frozen Phase-5 contract")


@dataclass(frozen=True)
class LocalizationLossConfig:
    radius_lower: float = 0.05
    radius_upper: float = 0.95
    radial_coefficient: float = 1.0
    angular_coefficient: float = 2.0
    circle_coefficient: float = 0.1
    outer_constant: float = 1.635
    field_consistency_coefficient: float = 0.03
    field_channels: tuple[str, str, str, str] = ("Re(E)", "Im(E)", "Re(H)", "Im(H)")
    kendall_initial_value: float = 0.0

    def __post_init__(self) -> None:
        expected = (0.05, 0.95, 1.0, 2.0, 0.1, 1.635, 0.03, ("Re(E)", "Im(E)", "Re(H)", "Im(H)"), 0.0)
        if tuple(asdict(self).values()) != expected:
            raise SchemaValidationError("Localization loss configuration differs from frozen Phase-5 contract")


@dataclass(frozen=True)
class LocalizationSurrogateConfig:
    deployment_seed: int = 20260917
    representation: str = "relative_angle"
    electric_checkpoint_sha256: str = HISTORICAL_CHECKPOINT_SHA256["E_z_seed20260917"]
    magnetic_checkpoint_sha256: str = HISTORICAL_CHECKPOINT_SHA256["H_phi_seed20260917"]

    def __post_init__(self) -> None:
        expected = (20260917, "relative_angle", HISTORICAL_CHECKPOINT_SHA256["E_z_seed20260917"], HISTORICAL_CHECKPOINT_SHA256["H_phi_seed20260917"])
        if tuple(asdict(self).values()) != expected:
            raise SchemaValidationError("Localization surrogate configuration differs from frozen Phase-5 contract")


@dataclass(frozen=True)
class LocalizationScientificConfig:
    model: LocalizationModelConfig = LocalizationModelConfig()
    loss: LocalizationLossConfig = LocalizationLossConfig()
    surrogates: LocalizationSurrogateConfig = LocalizationSurrogateConfig()
    canonical_correspondence: str = "fixed_slot"
    matching_role: str = "diagnostic_only"

    def __post_init__(self) -> None:
        if (self.canonical_correspondence, self.matching_role) != ("fixed_slot", "diagnostic_only"):
            raise SchemaValidationError("Localization correspondence configuration differs from frozen Phase-5 contract")

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))
