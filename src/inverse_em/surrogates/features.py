from __future__ import annotations
import torch
from inverse_em.errors import SchemaValidationError


def _physical_tensor(value, name:str)->torch.Tensor:
    if not isinstance(value,torch.Tensor) or value.dtype is not torch.float64 or value.device.type!="cpu":
        raise SchemaValidationError(f"{name} must be a CPU torch.float64 tensor")
    if not torch.isfinite(value).all(): raise SchemaValidationError(f"{name} must be finite")
    return value


def relative_angle_features(rho,phi,theta,R:float=1.0)->torch.Tensor:
    if type(R) is not float or R<=0: raise SchemaValidationError("R must be a positive float")
    rho,phi,theta=(_physical_tensor(v,n) for v,n in ((rho,"rho"),(phi,"phi"),(theta,"theta")))
    rho,phi,theta=torch.broadcast_tensors(rho,phi,theta)
    delta=theta-phi
    return torch.stack((rho/R,torch.cos(delta),torch.sin(delta)),dim=-1)
