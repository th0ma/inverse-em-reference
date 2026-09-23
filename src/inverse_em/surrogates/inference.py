from __future__ import annotations
import torch
from torch import nn
from .features import relative_angle_features


class FrozenSurrogate(nn.Module):
    def __init__(self,model:nn.Module,R:float=1.0):super().__init__();self.model=model;self.R=R
    def predict_components(self,rho,phi,theta):return self.model(relative_angle_features(rho,phi,theta,self.R))
    def predict_complex(self,rho,phi,theta):
        y=self.predict_components(rho,phi,theta);return torch.complex(y[...,0],y[...,1])
    def forward(self,rho,phi,theta):return self.predict_components(rho,phi,theta)


def freeze_surrogate(model:nn.Module)->nn.Module:
    model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False);parameter.grad=None
    return model


def superpose_complex(surrogate:FrozenSurrogate,rho,phi,theta,amplitude=None,*,source_dim:int=-2):
    values=surrogate.predict_complex(rho,phi,theta)
    if not isinstance(source_dim,int) or not -values.ndim<=source_dim<values.ndim: raise ValueError("source_dim must identify an axis of the predicted complex field tensor")
    if amplitude is not None: values=values*amplitude
    return values.sum(dim=source_dim)
