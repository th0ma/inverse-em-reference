from __future__ import annotations
import hashlib,math
import torch
from torch import nn
from inverse_em.config.surrogate import MODEL_SEEDS
from inverse_em.errors import SchemaValidationError


class BoundaryFieldMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(3,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,2))
        self.to(device="cpu",dtype=torch.float64)
    def forward(self,x):
        if not isinstance(x,torch.Tensor) or x.device.type!="cpu" or x.dtype is not torch.float64 or x.shape[-1:]!=(3,): raise SchemaValidationError("BoundaryFieldMLP requires CPU float64 (...,3) features")
        return self.net(x)


def _layer_seed(base:int,index:int,role:str)->int:
    return int.from_bytes(hashlib.sha256(f"{base}|{index}|{role}".encode()).digest()[:8],"little")%(2**63-1)


def initialize_historical(model:BoundaryFieldMLP,seed:int)->dict[str,int|str]:
    if seed not in MODEL_SEEDS: raise SchemaValidationError("Only frozen final Phase-3 initialization seeds are permitted")
    receipt={}
    for i,layer in enumerate(x for x in model.modules() if isinstance(x,nn.Linear)):
        for role,tensor in (("weight",layer.weight),("bias",layer.bias)):
            key=f"linear_{i}.{role}"
            if i==0 and role=="bias":
                with torch.no_grad(): tensor.zero_()
                receipt[key]="exact_zero"
            else:
                derived=_layer_seed(seed,i,role);gen=torch.Generator(device="cpu");gen.manual_seed(derived)
                with torch.no_grad(): tensor.uniform_(-1/math.sqrt(layer.in_features),1/math.sqrt(layer.in_features),generator=gen)
                receipt[key]=derived
    return receipt


def make_initialized_model(seed:int)->tuple[BoundaryFieldMLP,dict[str,int|str]]:
    state=torch.get_rng_state().clone()
    try: model=BoundaryFieldMLP()
    finally: torch.set_rng_state(state)
    return model,initialize_historical(model,seed)
