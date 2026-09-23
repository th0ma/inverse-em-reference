from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
import numpy as np
import torch
from torch.utils.data import Dataset
from inverse_em.config.surrogate import HISTORICAL_ARRAY_SHA256,HISTORICAL_SPLIT_SHA256,SurrogateDatasetConfig
from inverse_em.errors import SchemaValidationError
from inverse_em.physics import PhysicsTMForward,Source,observation_angles
from inverse_em.populations import split_surrogate_source_ids


def array_sha256(value)->str:
    a=np.asarray(value);canonical=np.ascontiguousarray(a)
    if canonical.dtype.byteorder==">" or (canonical.dtype.byteorder=="=" and not np.little_endian):canonical=canonical.byteswap().view(canonical.dtype.newbyteorder("<"))
    payload=json.dumps({"dtype":a.dtype.str,"shape":list(a.shape)},sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()+b"\0"+canonical.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()
def _immutable(value,dtype,shape):
    a=np.asarray(value,dtype=dtype)
    if a.shape!=shape or (np.issubdtype(a.dtype,np.number) and not np.isfinite(a).all()): raise SchemaValidationError("Invalid surrogate array")
    return np.frombuffer(np.ascontiguousarray(a).tobytes(),dtype=dtype).reshape(shape)


@dataclass(frozen=True)
class HistoricalSurrogateSources:
    source_id:np.ndarray;rho:np.ndarray;phi:np.ndarray;theta:np.ndarray
    def __post_init__(self):
        n=len(self.source_id);object.__setattr__(self,"source_id",_immutable(self.source_id,"U12",(n,)))
        object.__setattr__(self,"rho",_immutable(self.rho,np.float64,(n,)));object.__setattr__(self,"phi",_immutable(self.phi,np.float64,(n,)))
        object.__setattr__(self,"theta",_immutable(self.theta,np.float64,(72,)))
        if len(np.unique(self.source_id))!=n or np.any((self.rho<.01)|(self.rho>=.99)) or np.any((self.phi<0)|(self.phi>=2*np.pi)): raise SchemaValidationError("Historical surrogate source contract violated")


@dataclass(frozen=True)
class SurrogateSplit:
    training:tuple[str,...];validation:tuple[str,...];test:tuple[str,...]
    def __post_init__(self):
        groups=tuple(set(x) for x in (self.training,self.validation,self.test))
        if tuple(map(len,(self.training,self.validation,self.test)))!=(7000,1500,1500) or any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)) or len(set.union(*groups))!=10000: raise SchemaValidationError("Invalid historical surrogate split")


def generate_historical_sources(config:SurrogateDatasetConfig=SurrogateDatasetConfig())->HistoricalSurrogateSources:
    rng=np.random.default_rng(config.generation_seed)
    rho=np.sqrt(rng.uniform(config.rho_min**2,config.rho_max**2,size=config.source_count))
    phi=rng.uniform(0.,2*np.pi,size=config.source_count)
    ids=np.array([f"src-{i:08d}" for i in range(config.source_count)],dtype="U12")
    return HistoricalSurrogateSources(ids,rho,phi,observation_angles(config.angle_count))


def historical_split(source_ids,split_seed:int=20260916)->SurrogateSplit:
    ids=tuple(str(x) for x in np.asarray(source_ids).tolist())
    if len(ids)!=10000 or len(set(ids))!=10000 or split_seed!=20260916: raise SchemaValidationError("Historical split requires the frozen source identity")
    split=split_surrogate_source_ids(ids,split_seed)
    return SurrogateSplit(split.training,split.validation,split.test)


def generate_analytical_targets(sources:HistoricalSurrogateSources,source_indices,forward:PhysicsTMForward|None=None)->tuple[np.ndarray,np.ndarray]:
    indices=np.asarray(source_indices,dtype=np.int64)
    if indices.ndim!=1 or np.any(indices<0) or np.any(indices>=len(sources.source_id)): raise SchemaValidationError("Invalid source indices")
    solver=PhysicsTMForward() if forward is None else forward;e=np.empty((len(indices),72,2),np.float64);h=np.empty_like(e)
    for j,i in enumerate(indices.tolist()):
        fields=solver.evaluate(Source(float(sources.rho[i]),float(sources.phi[i])),sources.theta)
        e[j,:,0]=fields.electric.real;e[j,:,1]=fields.electric.imag;h[j,:,0]=fields.magnetic.real;h[j,:,1]=fields.magnetic.imag
    return e,h


class LazyPointwiseFieldDataset(Dataset):
    def __init__(self,source_id,rho,phi,theta,targets,*,partition:str,split:SurrogateSplit,sources:HistoricalSurrogateSources):
        if partition not in {"TRAINING","VALIDATION"}: raise PermissionError("Training APIs reject held-out surrogate views")
        source_id=np.asarray(source_id,dtype="U12")
        self.rho=np.asarray(rho,dtype=np.float64);self.phi=np.asarray(phi,dtype=np.float64);self.theta=np.asarray(theta,dtype=np.float64);self.targets=np.asarray(targets,dtype=np.float64);self.partition=partition
        n=len(self.rho);m=len(self.theta)
        if source_id.shape!=(n,) or len(set(source_id.tolist()))!=n or self.rho.shape!=(n,) or self.phi.shape!=(n,) or self.theta.shape!=(m,) or self.targets.shape!=(n,m,2) or not all(np.isfinite(x).all() for x in (self.rho,self.phi,self.theta,self.targets)): raise SchemaValidationError("Invalid lazy pointwise dataset arrays")
        source_hashes={"rho_s":array_sha256(sources.rho),"phi_s":array_sha256(sources.phi),"source_id":array_sha256(sources.source_id),"theta":array_sha256(sources.theta)}
        split_hashes={name:array_sha256(np.asarray(getattr(split,name),dtype="U12")) for name in ("training","validation","test")}
        if source_hashes!=dict(HISTORICAL_ARRAY_SHA256) or split_hashes!=dict(HISTORICAL_SPLIT_SHA256): raise SchemaValidationError("Dataset view is not bound to the frozen historical source and split identity")
        allowed=set(split.training if partition=="TRAINING" else split.validation)
        if not set(source_id.tolist()).issubset(allowed): raise PermissionError(f"{partition} view contains source IDs outside its frozen partition")
        lookup={value:i for i,value in enumerate(sources.source_id.tolist())};indices=np.asarray([lookup[value] for value in source_id.tolist()],dtype=np.int64)
        if not np.array_equal(self.rho,sources.rho[indices]) or not np.array_equal(self.phi,sources.phi[indices]) or not np.array_equal(self.theta,sources.theta): raise SchemaValidationError("Dataset geometry is not aligned with its frozen source IDs")
        self.source_id=source_id
    def __len__(self):return len(self.rho)*len(self.theta)
    def __getitem__(self,index):
        source_index,angle_index=divmod(int(index),len(self.theta))
        return (torch.tensor(self.rho[source_index],dtype=torch.float64),torch.tensor(self.phi[source_index],dtype=torch.float64),torch.tensor(self.theta[angle_index],dtype=torch.float64),torch.from_numpy(self.targets[source_index,angle_index]))
