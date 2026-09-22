from dataclasses import asdict,dataclass
from types import MappingProxyType
from inverse_em.physics import AngleGridConfig,PhysicsTMConfig
from inverse_em.populations import CLASSIFIER_FINAL,S1_FINAL,S2_FINAL,S3_FINAL,RNGIdentity,S3_CURRICULUM,SeparationConstraint,SourceLaw
from inverse_em.provenance.canonical import canonical_sha256
@dataclass(frozen=True)
class Phase1ScientificConfig:
    physics:PhysicsTMConfig=PhysicsTMConfig();surrogate_grid:AngleGridConfig=AngleGridConfig(72);inverse_grid:AngleGridConfig=AngleGridConfig(30);rng_algorithm:str="NumPy PCG64DXSM"
    @property
    def sha256(self):return canonical_sha256(asdict(self))
SURROGATE_SOURCE_LAW=SourceLaw(.01,.99);LOCALIZATION_SOURCE_LAW=SourceLaw(.05,.95);S2_SEPARATION=SeparationConstraint(.05,5.)
PHASE1_SEEDS=MappingProxyType({name:RNGIdentity("NumPy PCG64DXSM",seed) for name,seed in {"classifier_training":20261901,"classifier_validation":20261902,"classifier_sealed":20261906,"s1_training":20261811,"s1_validation":20261812,"s1_sealed":20261816,"s2_training":20262001,"s2_validation":20262002,"s2_sealed":20262006,"s3_training_master":20262201,"s3_validation":20262202,"s3_sealed":20262206}.items()})
FINAL_POPULATION_DEFINITIONS=(CLASSIFIER_FINAL,S1_FINAL,S2_FINAL,S3_FINAL)
