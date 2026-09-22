from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import numpy as np
from inverse_em.errors import SchemaValidationError
from .vendor import PhysicsTM, UPSTREAM_SOURCE_SHA256

@dataclass(frozen=True)
class PhysicsTMConfig:
    radius: float = 1.0
    omega: float = 4.0
    eps0: float = 1.0
    mu0: float = 1.0
    eps1: float = 1.3225
    mu1: float = 1.0
    amplitude: float = 1.0
    modal_order: int = 20
    dtype: str = "complex128"
    def __post_init__(self) -> None:
        expected=(1.0,4.0,1.0,1.0,1.3225,1.0,1.0,20,"complex128")
        actual=(self.radius,self.omega,self.eps0,self.mu0,self.eps1,self.mu1,self.amplitude,self.modal_order,self.dtype)
        if actual != expected: raise SchemaValidationError("PhysicsTM configuration differs from the frozen normalized contract")
    @property
    def modal_indices(self) -> np.ndarray:
        out=np.arange(-self.modal_order,self.modal_order+1,dtype=np.int64);out.setflags(write=False);return out

@dataclass(frozen=True)
class AngleGridConfig:
    count: int
    endpoint: bool = False
    lower: float = 0.0
    upper: float = 2.0*math.pi
    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count<1 or self.endpoint or self.lower!=0.0 or self.upper!=2.0*math.pi: raise SchemaValidationError("Only positive endpoint-excluded [0, 2*pi) grids are frozen")
    def values(self)->np.ndarray:
        out=2.0*np.pi*np.arange(self.count,dtype=np.float64)/self.count;out.setflags(write=False);return out

def observation_angles(count:int)->np.ndarray: return AngleGridConfig(count).values()

@dataclass(frozen=True)
class Source:
    rho: float
    phi: float
    amplitude: float = 1.0
    def __post_init__(self)->None:
        if not(math.isfinite(self.rho) and 0.0<=self.rho<1.0): raise SchemaValidationError("rho must be finite and in [0, 1)")
        if not(math.isfinite(self.phi) and 0.0<=self.phi<2.0*math.pi): raise SchemaValidationError("phi must be finite and in [0, 2*pi)")
        if self.amplitude!=1.0: raise SchemaValidationError("Only unit-amplitude sources are frozen")
    @property
    def cartesian(self)->tuple[float,float]: return self.rho*math.cos(self.phi),self.rho*math.sin(self.phi)

@dataclass(frozen=True)
class BoundaryFields:
    electric: np.ndarray
    magnetic: np.ndarray
    def __post_init__(self)->None:
        e=np.asarray(self.electric);h=np.asarray(self.magnetic)
        if e.shape!=h.shape or e.dtype!=np.complex128 or h.dtype!=np.complex128 or not np.isfinite(e).all() or not np.isfinite(h).all(): raise SchemaValidationError("Boundary fields must be finite shape-matched complex128 arrays")
        e=np.frombuffer(np.ascontiguousarray(e).tobytes(),dtype=np.complex128).reshape(e.shape);h=np.frombuffer(np.ascontiguousarray(h).tobytes(),dtype=np.complex128).reshape(h.shape)
        object.__setattr__(self,"electric",e);object.__setattr__(self,"magnetic",h)

def vendored_source_sha256()->str:
    return hashlib.sha256((Path(__file__).with_name("vendor")/"physics_tm.py").read_bytes()).hexdigest()
def verify_vendored_source()->bool:
    if vendored_source_sha256()!=UPSTREAM_SOURCE_SHA256: raise RuntimeError("Vendored PhysicsTM source differs from pinned upstream bytes")
    return True

class PhysicsTMForward:
    def __init__(self,config:PhysicsTMConfig=PhysicsTMConfig())->None:
        verify_vendored_source();self.config=config
        self._solver=PhysicsTM(R=config.radius,OMEGA=config.omega,EPS0=config.eps0,MU0=config.mu0,EPS1=config.eps1,MU1=config.mu1,N=config.modal_order,I0=config.amplitude)
    def evaluate(self,source:Source,angles:np.ndarray)->BoundaryFields:
        theta=np.asarray(angles,dtype=np.float64)
        if theta.ndim!=1 or not np.isfinite(theta).all(): raise SchemaValidationError("Observation angles must be a finite one-dimensional array")
        return BoundaryFields(np.asarray(self._solver.Esurf(source.rho,source.phi,theta),dtype=np.complex128),np.asarray(self._solver.Hsurf(source.rho,source.phi,theta),dtype=np.complex128))
