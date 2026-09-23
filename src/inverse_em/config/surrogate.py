from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType

from inverse_em.errors import SchemaValidationError
from inverse_em.provenance.canonical import canonical_sha256


MODEL_SEEDS = (20260917, 20260918, 20260919)
RESERVED_MODEL_SEEDS = (20260920, 20260921)
HISTORICAL_ARRAY_SHA256 = MappingProxyType({
    "rho_s": "f45f76b23aac87cf08bfbe15c62bf3a7fcca07ccb81fe0f6fec2192c1958ca9a",
    "phi_s": "eaf30099ca7a6d4af932a5b245be6971ab5f423fd831f10bd395dc37b5865db2",
    "source_id": "90aae2c90bbc5f2c7277e50646cc43fbe1640b3b6a872e58f7439857bbf8a046",
    "theta": "2b76004236554504affbf3231eddb5b294aa1125e2412451cfcbe8be2f721bad",
})
HISTORICAL_SPLIT_SHA256 = MappingProxyType({
    "training": "6fffec2650167fad23dfa4579902e4ca3c2b2ec1034d038921642beb501f9b2f",
    "validation": "f95a4673dc35059eede3dfd22f3aa4cfd330bc6fadbb6f65bcd3baae2efc87ec",
    "test": "49c4df10dd1fe3fb58c88ed472b6db4887dc509611e838da82cbe5fb783bceb0",
})
HISTORICAL_CHECKPOINT_SHA256 = MappingProxyType({
    "E_z_seed20260917": "d3a274f0ab4d514cdbb6b5edb32f3976b9926bb82e838e0277ba0f69f5ffa80b",
    "H_phi_seed20260917": "0285679d940bb20ecbea334e0815018bf7c8aa6b7210d5e2ce3a2a57a053173c",
})


@dataclass(frozen=True)
class SurrogateDatasetConfig:
    generation_seed: int = 20260915
    split_seed: int = 20260916
    source_count: int = 10000
    training_sources: int = 7000
    validation_sources: int = 1500
    held_out_sources: int = 1500
    angle_count: int = 72
    rho_min: float = .01
    rho_max: float = .99
    rng_algorithm: str = "NumPy default_rng / PCG64"

    def __post_init__(self):
        actual=(self.generation_seed,self.split_seed,self.source_count,self.training_sources,self.validation_sources,self.held_out_sources,self.angle_count,self.rho_min,self.rho_max,self.rng_algorithm)
        expected=(20260915,20260916,10000,7000,1500,1500,72,.01,.99,"NumPy default_rng / PCG64")
        if actual != expected: raise SchemaValidationError("Surrogate dataset configuration differs from frozen Phase-3 contract")


@dataclass(frozen=True)
class SurrogateTrainingConfig:
    learning_rate: float = 1e-3
    betas: tuple[float,float] = (.9,.999)
    epsilon: float = 1e-8
    weight_decay: float = 0.
    batch_size: int = 512
    shuffle: bool = True
    num_workers: int = 0
    drop_last: bool = False
    maximum_epochs: int = 200
    early_stopping_patience: int = 20
    scheduler_mode: str = "min"
    scheduler_factor: float = .5
    scheduler_patience: int = 5
    scheduler_threshold: float = 1e-4
    scheduler_threshold_mode: str = "rel"
    scheduler_cooldown: int = 0
    scheduler_min_lr: float = 0.
    scheduler_epsilon: float = 1e-8
    required_torch_version: str = "2.8.0"

    def __post_init__(self):
        expected=(1e-3,(.9,.999),1e-8,0.,512,True,0,False,200,20,"min",.5,5,1e-4,"rel",0,0.,1e-8,"2.8.0")
        if tuple(asdict(self).values()) != expected: raise SchemaValidationError("Surrogate training configuration differs from frozen Phase-3 contract")

    @property
    def sha256(self)->str: return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class SurrogateScientificConfig:
    representation: str = "relative_angle_B"
    fields: tuple[str,str] = ("E_z","H_phi")
    architecture: tuple[int,...] = (3,128,128,128,128,2)
    activation: str = "ReLU"
    dtype: str = "float64"
    device: str = "cpu"
    target_scaling: None = None
    physics_revision: str = "a079c3899bd33f636ab9c8b1571d8e919c839335"
    physics_source_sha256: str = "b0d8200ef30d05720f0b697822dc0e037935243beea9084edf410c9725c32005"
    dataset: SurrogateDatasetConfig = SurrogateDatasetConfig()
    training: SurrogateTrainingConfig = SurrogateTrainingConfig()

    def __post_init__(self):
        if (self.representation,self.fields,self.architecture,self.activation,self.dtype,self.device,self.target_scaling)!=("relative_angle_B",("E_z","H_phi"),(3,128,128,128,128,2),"ReLU","float64","cpu",None):
            raise SchemaValidationError("Surrogate scientific configuration differs from frozen Phase-3 contract")

    @property
    def sha256(self)->str: return canonical_sha256(asdict(self))
