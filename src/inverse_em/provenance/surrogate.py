from __future__ import annotations
from dataclasses import asdict,dataclass
from inverse_em.provenance.canonical import canonical_sha256


@dataclass(frozen=True)
class SurrogateExecutionReceipt:
    field:str;seed:int;epoch:int;updates:int;best_validation_mse:float;configuration_sha256:str;dataset_identity:str;split_identity:str;physics_revision:str;physics_source_sha256:str;repository_revision:str;torch_version:str;loader_seed:int
    def __post_init__(self):
        if self.field not in {"E_z","H_phi"} or self.seed not in {20260917,20260918,20260919} or self.epoch<1 or self.updates<1 or self.best_validation_mse<0:raise ValueError("Invalid surrogate execution receipt")
    @property
    def sha256(self):return canonical_sha256(asdict(self))
