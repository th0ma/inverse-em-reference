from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
from inverse_em.errors import SchemaValidationError

class PopulationRole(str,Enum):
    TRAINING="TRAINING";VALIDATION="VALIDATION";SEALED="SEALED";REFERENCE_FIXTURE="REFERENCE_FIXTURE"

@dataclass(frozen=True)
class ProtectedPopulationIdentity:
    study:str;population_id:str;role:PopulationRole;seed:int;scientific_signature:tuple;stage:int|None=None
    def __post_init__(self):
        if not self.study.strip() or not self.population_id.strip() or not isinstance(self.role,PopulationRole) or self.role is not PopulationRole.SEALED or type(self.seed)is not int or self.seed<0 or not isinstance(self.scientific_signature,tuple) or not self.scientific_signature: raise SchemaValidationError("Invalid protected population identity")
        if self.stage is not None and (type(self.stage)is not int or self.stage<1): raise SchemaValidationError("Invalid protected stage")

@dataclass(frozen=True)
class FinalPopulationIdentity:
    study:str;role:PopulationRole;seed:int;count:int;source_count:int;per_class:int|None=None;stage:int|None=None;stages:int|None=None;count_per_stage:int|None=None
    def __post_init__(self):
        if not self.study.strip() or not isinstance(self.role,PopulationRole) or self.role is PopulationRole.REFERENCE_FIXTURE or type(self.seed)is not int or self.seed<0: raise SchemaValidationError("Invalid final population identity")
        if type(self.count)is not int or self.count<1 or self.source_count not in range(1,6): raise SchemaValidationError("Invalid final population size")
        if self.per_class is not None and (type(self.per_class)is not int or self.per_class<1 or self.per_class*5!=self.count): raise SchemaValidationError("Invalid balanced classifier size")
        if self.stage is not None and (type(self.stage)is not int or self.stage not in range(1,9)): raise SchemaValidationError("Invalid final population stage")
        if (self.stages is None)!=(self.count_per_stage is None): raise SchemaValidationError("Stage count fields must be paired")
        if self.stages is not None and (self.stages!=8 or type(self.count_per_stage)is not int or self.count_per_stage<1 or self.count!=self.stages*self.count_per_stage): raise SchemaValidationError("Invalid curriculum population size")

@dataclass(frozen=True)
class SurrogatePopulationIdentity:
    generation_seed:int=20260915;split_seed:int=20260916;total:int=10000;training:int=7000;validation:int=1500;test:int=1500;rng_algorithm:str="NumPy PCG64DXSM"
    def __post_init__(self):
        if any(type(x)is not int or x<0 for x in (self.generation_seed,self.split_seed,self.total,self.training,self.validation,self.test)): raise SchemaValidationError("Invalid surrogate population identity")
        if self.total!=self.training+self.validation+self.test or (self.total,self.training,self.validation,self.test)!=(10000,7000,1500,1500) or self.rng_algorithm!="NumPy PCG64DXSM": raise SchemaValidationError("Surrogate identity differs from frozen contract")

@dataclass(frozen=True)
class RNGIdentity:
    algorithm:str;seed:int
    def __post_init__(self):
        if self.algorithm!="NumPy PCG64DXSM" or type(self.seed)is not int or self.seed<0: raise SchemaValidationError("Invalid RNG identity")

@dataclass(frozen=True)
class SourceLaw:
    rho_min:float;rho_max:float;amplitude:float=1.0;radial_distribution:str="area_uniform";azimuth_distribution:str="uniform_[0,2pi)"
    def __post_init__(self):
        if not(0<self.rho_min<self.rho_max<1) or self.amplitude!=1.0 or self.radial_distribution!="area_uniform" or self.azimuth_distribution!="uniform_[0,2pi)": raise SchemaValidationError("Invalid source law")

@dataclass(frozen=True)
class SeparationConstraint:
    radial:float;angular_degrees:float
    def __post_init__(self):
        if not(math.isfinite(self.radial) and self.radial>=0 and math.isfinite(self.angular_degrees) and 0<=self.angular_degrees<=180): raise SchemaValidationError("Invalid separation")

@dataclass(frozen=True)
class CurriculumStage:
    stage:int;separation:SeparationConstraint

S3_CURRICULUM=(CurriculumStage(1,SeparationConstraint(.10,40.)),CurriculumStage(2,SeparationConstraint(.08,25.)),CurriculumStage(3,SeparationConstraint(.08,20.)),CurriculumStage(4,SeparationConstraint(.07,15.)),CurriculumStage(5,SeparationConstraint(.06,10.)),CurriculumStage(6,SeparationConstraint(.05,8.)),CurriculumStage(7,SeparationConstraint(.05,6.)),CurriculumStage(8,SeparationConstraint(.05,5.)))

SURROGATE_FINAL=SurrogatePopulationIdentity()
CLASSIFIER_FINAL=(FinalPopulationIdentity("classifier",PopulationRole.TRAINING,20261901,70000,5,14000),FinalPopulationIdentity("classifier",PopulationRole.VALIDATION,20261902,15000,5,3000),FinalPopulationIdentity("classifier",PopulationRole.SEALED,20261906,15000,5,3000))
S1_FINAL=(FinalPopulationIdentity("s1",PopulationRole.TRAINING,20261811,70000,1),FinalPopulationIdentity("s1",PopulationRole.VALIDATION,20261812,15000,1),FinalPopulationIdentity("s1",PopulationRole.SEALED,20261816,15000,1))
S2_FINAL=(FinalPopulationIdentity("s2",PopulationRole.TRAINING,20262001,70000,2),FinalPopulationIdentity("s2",PopulationRole.VALIDATION,20262002,15000,2),FinalPopulationIdentity("s2",PopulationRole.SEALED,20262006,15000,2))
S3_FINAL=(FinalPopulationIdentity("s3",PopulationRole.TRAINING,20262201,560000,3,stages=8,count_per_stage=70000),FinalPopulationIdentity("s3",PopulationRole.VALIDATION,20262202,15000,3,stage=8),FinalPopulationIdentity("s3",PopulationRole.SEALED,20262206,15000,3,stage=8))
FINAL_POPULATION_IDENTITIES=CLASSIFIER_FINAL+S1_FINAL+S2_FINAL+S3_FINAL
PROTECTED_SEALED_IDENTITIES=(
    ProtectedPopulationIdentity("classifier","classifier_final",PopulationRole.SEALED,20261906,("classifier_factory",15000,3000)),
    ProtectedPopulationIdentity("s1","s1_final",PopulationRole.SEALED,20261816,("interleaved",15000,1,.05,.95,None,None)),
    ProtectedPopulationIdentity("s2","s2_final",PopulationRole.SEALED,20262006,("interleaved",15000,2,.05,.95,.05,5.)),
    ProtectedPopulationIdentity("s3","s3_final",PopulationRole.SEALED,20262206,("interleaved",15000,3,.05,.95,.05,5.),8),
)

def final_population_identity(study:str,role:PopulationRole)->FinalPopulationIdentity:
    matches=tuple(x for x in FINAL_POPULATION_IDENTITIES if x.study==study and x.role is role)
    if len(matches)!=1: raise SchemaValidationError("Unknown final population identity")
    return matches[0]

@dataclass(frozen=True)
class PopulationDefinition:
    population_id:str;role:PopulationRole;count:int;source_count:int;source_law:SourceLaw;rng:RNGIdentity;stage:int|None=None;separation:SeparationConstraint|None=None
    def __post_init__(self):
        if not self.population_id.strip() or type(self.count)is not int or self.count<1 or self.source_count not in range(1,6): raise SchemaValidationError("Invalid population definition")
        if self.population_id in {"s2_final","s3_final"} and self.separation is None: raise SchemaValidationError("Separation required")
