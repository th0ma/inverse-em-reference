from __future__ import annotations
from dataclasses import dataclass
import hashlib,math
import numpy as np
from inverse_em.errors import SchemaValidationError
from .contracts import PROTECTED_SEALED_IDENTITIES,PopulationDefinition,PopulationRole,RNGIdentity,S3_CURRICULUM,SeparationConstraint,SourceLaw

def _immutable_array(value,dtype,shape):
    source=np.asarray(value,dtype=dtype)
    if source.shape!=shape: raise SchemaValidationError("Invalid immutable array shape")
    return np.frombuffer(np.ascontiguousarray(source).tobytes(),dtype=dtype).reshape(shape)

@dataclass(frozen=True)
class GeneratedPopulation:
    definition:PopulationDefinition;rho:np.ndarray;phi:np.ndarray;amplitude:np.ndarray;proposal_ordinal:np.ndarray
    def __post_init__(self):
        shape=(self.definition.count,self.definition.source_count);arrays=tuple(_immutable_array(x,np.float64,shape) for x in (self.rho,self.phi,self.amplitude));ords=_immutable_array(self.proposal_ordinal,np.uint64,(self.definition.count,))
        if any(a.shape!=shape or a.dtype!=np.float64 or not np.isfinite(a).all() for a in arrays): raise SchemaValidationError("Invalid population geometry")
        if ords.shape!=(self.definition.count,) or ords.dtype!=np.uint64 or np.any(ords<1): raise SchemaValidationError("Invalid proposal ordinals")
        if np.any(arrays[0]<self.definition.source_law.rho_min) or np.any(arrays[0]>=self.definition.source_law.rho_max): raise SchemaValidationError("Radius outside law")
        if np.any(arrays[1]<0) or np.any(arrays[1]>=2*np.pi) or np.any(arrays[2]!=1): raise SchemaValidationError("Angle/amplitude outside law")
        object.__setattr__(self,"rho",arrays[0]);object.__setattr__(self,"phi",arrays[1]);object.__setattr__(self,"amplitude",arrays[2]);object.__setattr__(self,"proposal_ordinal",ords)

@dataclass(frozen=True)
class SourceSplit:
    training:tuple[str,...];validation:tuple[str,...];test:tuple[str,...]
    def __post_init__(self):
        groups=tuple(set(x) for x in (self.training,self.validation,self.test))
        if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)): raise SchemaValidationError("Split overlap")

def area_uniform_radius(unit,law:SourceLaw)->np.ndarray:
    u=np.asarray(unit,dtype=np.float64)
    if np.any((u<0)|(u>=1)): raise SchemaValidationError("Uniform variates must be in [0,1)")
    r=np.sqrt(np.float64(law.rho_min)**2+(np.float64(law.rho_max)**2-np.float64(law.rho_min)**2)*u)
    return np.minimum(r,np.nextafter(np.float64(law.rho_max),np.float64(law.rho_min)))
def wrapped_angular_separation(a,b)->np.ndarray:
    d=np.abs(np.asarray(a)-np.asarray(b))%(2*np.pi);return np.minimum(d,2*np.pi-d)
def eight_ulp_atol(threshold:float)->float:return float(8*np.spacing(np.float64(threshold)))
def accepts_threshold(value:float,threshold:float)->bool:
    t=np.float64(threshold);return bool(value>=t or math.isclose(float(value),float(t),rel_tol=0,abs_tol=eight_ulp_atol(float(t))))
def admissible(rho,phi,separation:SeparationConstraint)->bool:
    angular=float(np.deg2rad(np.float64(separation.angular_degrees)))
    for i in range(len(rho)):
        for j in range(i+1,len(rho)):
            if not accepts_threshold(abs(float(rho[i]-rho[j])),separation.radial):return False
            if not accepts_threshold(float(wrapped_angular_separation(phi[i],phi[j])),angular):return False
    return True
def canonicalize(rho,phi):
    order=np.lexsort((np.asarray(phi),np.asarray(rho)));return np.asarray(rho,np.float64)[order],np.asarray(phi,np.float64)[order]
def _guard_identity(population_id,role,seed,stage,scientific_signature,allow_sealed):
    protected=tuple(x for x in PROTECTED_SEALED_IDENTITIES if x.seed==seed)
    if protected:
        exact=tuple(x for x in protected if x.population_id==population_id and x.role is role and x.stage==stage and x.scientific_signature==scientific_signature)
        if not exact: raise PermissionError("Protected sealed seed requires its exact sealed population identity")
        if not allow_sealed: raise PermissionError("Sealed generation requires separate authorization")
    elif role is PopulationRole.SEALED and not allow_sealed: raise PermissionError("Sealed generation requires separate authorization")
def generate_interleaved(d:PopulationDefinition,*,canonical:bool,allow_sealed:bool=False)->GeneratedPopulation:
    separation=(None,None) if d.separation is None else (d.separation.radial,d.separation.angular_degrees)
    signature=("interleaved",d.count,d.source_count,d.source_law.rho_min,d.source_law.rho_max,*separation)
    _guard_identity(d.population_id,d.role,d.rng.seed,d.stage,signature,allow_sealed)
    rng=np.random.Generator(np.random.PCG64DXSM(d.rng.seed));rs=[];ps=[];ords=[];attempt=0;seen=set()
    while len(rs)<d.count:
        attempt+=1;u=rng.random((d.source_count,2));r=area_uniform_radius(u[:,0],d.source_law);p=2*np.pi*u[:,1]
        if d.separation and not admissible(r,p,d.separation):continue
        if canonical:r,p=canonicalize(r,p)
        if d.population_id=="s3_final":
            identity=hashlib.sha256(np.stack((r,p),axis=1).astype("<f8").tobytes()).digest()
            if identity in seen:continue
            seen.add(identity)
        rs.append(r);ps.append(p);ords.append(attempt)
    r=np.asarray(rs,np.float64);p=np.asarray(ps,np.float64)
    return GeneratedPopulation(d,r,p,np.ones_like(r),np.asarray(ords,np.uint64))
def generate_s1(count,seed,role=PopulationRole.REFERENCE_FIXTURE,*,allow_sealed=False):
    _guard_identity("s1_final",role,seed,None,("interleaved",count,1,.05,.95,None,None),allow_sealed);d=PopulationDefinition("s1_final",role,count,1,SourceLaw(.05,.95),RNGIdentity("NumPy PCG64DXSM",seed));return generate_interleaved(d,canonical=False,allow_sealed=allow_sealed)
def generate_s2(count,seed,role=PopulationRole.REFERENCE_FIXTURE,*,allow_sealed=False):
    _guard_identity("s2_final",role,seed,None,("interleaved",count,2,.05,.95,.05,5.),allow_sealed);sep=SeparationConstraint(.05,5.);d=PopulationDefinition("s2_final",role,count,2,SourceLaw(.05,.95),RNGIdentity("NumPy PCG64DXSM",seed),separation=sep);return generate_interleaved(d,canonical=True,allow_sealed=allow_sealed)
def derived_s3_stage_seed(master_seed:int,stage:int)->int:
    if stage not in range(1,9):raise SchemaValidationError("S3 stage must be 1..8")
    raw=hashlib.sha256(f"s3_analytical_noise_final_v1|POPULATION|{int(master_seed)}|STAGE|{stage}".encode()).digest();return int.from_bytes(raw[:16],"little")
def generate_s3(count,master_seed,stage,role=PopulationRole.REFERENCE_FIXTURE,*,allow_sealed=False):
    if stage not in range(1,9):raise SchemaValidationError("S3 stage must be 1..8")
    contract=S3_CURRICULUM[stage-1]
    _guard_identity("s3_final",role,master_seed,stage,("interleaved",count,3,.05,.95,contract.separation.radial,contract.separation.angular_degrees),allow_sealed)
    seed=master_seed if role in (PopulationRole.VALIDATION,PopulationRole.SEALED) else derived_s3_stage_seed(master_seed,stage)
    d=PopulationDefinition("s3_final",role,count,3,SourceLaw(.05,.95),RNGIdentity("NumPy PCG64DXSM",seed),stage,contract.separation);return generate_interleaved(d,canonical=True,allow_sealed=allow_sealed)
def generate_classifier(per_class,seed,role=PopulationRole.REFERENCE_FIXTURE,*,allow_sealed=False):
    if per_class<1:raise SchemaValidationError("per_class must be positive")
    _guard_identity("classifier_final",role,seed,None,("classifier_factory",per_class*5,per_class),allow_sealed)
    rng=np.random.Generator(np.random.PCG64DXSM(seed));law=SourceLaw(.01,.99);out={};proposal=0
    for s in range(1,6):
        rs=[];ps=[];ords=[]
        for _ in range(per_class):proposal+=1;rs.append(area_uniform_radius(rng.random(s),law));ps.append(rng.random(s)*(2*np.pi));ords.append(proposal)
        d=PopulationDefinition("classifier_final",role,per_class,s,law,RNGIdentity("NumPy PCG64DXSM",seed));r=np.asarray(rs,np.float64);p=np.asarray(ps,np.float64)
        out[s]=GeneratedPopulation(d,r,p,np.ones_like(r),np.asarray(ords,dtype=np.uint64))
    return out
def generate_surrogate(count,seed,role=PopulationRole.REFERENCE_FIXTURE,*,allow_sealed=False):
    d=PopulationDefinition("surrogate_reference",role,count,1,SourceLaw(.01,.99),RNGIdentity("NumPy PCG64DXSM",seed));
    _guard_identity(d.population_id,d.role,d.rng.seed,d.stage,("surrogate",count,1,.01,.99),allow_sealed)
    rng=np.random.Generator(np.random.PCG64DXSM(seed));r=area_uniform_radius(rng.random(count),d.source_law)[:,None];p=(rng.random(count)*(2*np.pi))[:,None]
    return GeneratedPopulation(d,r,p,np.ones_like(r),np.arange(1,count+1,dtype=np.uint64))
def split_surrogate_source_ids(source_ids,split_seed:int)->SourceSplit:
    if len(source_ids)!=10000 or len(set(source_ids))!=len(source_ids):raise SchemaValidationError("Need 10,000 unique source identities")
    ordered=tuple(v for _,v in sorted((hashlib.sha256(f"1.0|{split_seed}|{v}".encode()).digest(),v) for v in source_ids));return SourceSplit(ordered[:7000],ordered[7000:8500],ordered[8500:])
