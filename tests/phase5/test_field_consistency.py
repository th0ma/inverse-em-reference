import builtins
import math
import torch
from torch import nn

from inverse_em.localization import (FIELD_CHANNELS, FieldConsistency, RawLocalizerOutput, compose_fc_objective,
                                     make_localizer, parameter_count, scientific_view)
from inverse_em.surrogates.inference import FrozenSurrogate


class TinyFieldModel(nn.Module):
    def __init__(self,scale):
        super().__init__();self.scale=nn.Parameter(torch.tensor(float(scale),dtype=torch.float64))
    def forward(self,features):
        real=self.scale*(features[...,0]+.2*features[...,1])
        imag=self.scale*(features[...,2]-.1*features[...,0])
        return torch.stack((real,imag),-1)


def fixture(active=2,batch=2):
    theta=2*torch.pi*torch.arange(30,dtype=torch.float64)/30
    fc=FieldConsistency(FrozenSurrogate(TinyFieldModel(1.)),FrozenSurrogate(TinyFieldModel(2.)),theta)
    raw=RawLocalizerOutput(torch.zeros(batch,3,dtype=torch.float64,requires_grad=True),
        torch.tensor([[1.,.7,.2]],dtype=torch.float64).repeat(batch,1).requires_grad_(),
        torch.tensor([[.2,.8,.5]],dtype=torch.float64).repeat(batch,1).requires_grad_())
    return fc,raw,scientific_view(raw,active)


def test_kendall_parameter_count_order_and_zero_initialization():
    fc,_,_=fixture();localizer=make_localizer()
    assert FIELD_CHANNELS==("Re(E)","Im(E)","Re(H)","Im(H)")
    assert fc.log_variances.shape==(4,) and fc.log_variances.dtype is torch.float64
    assert torch.equal(fc.log_variances,torch.zeros(4,dtype=torch.float64))
    assert parameter_count(localizer)+parameter_count(fc)==399_437


def test_fc_active_slots_raw_channels_and_clean_target_separation():
    fc,_,prediction=fixture(active=1);reconstructed=fc.reconstruct(prediction)
    assert reconstructed.shape==(2,4,30) and reconstructed.dtype is torch.float64
    clean=torch.zeros_like(reconstructed);noisy=torch.ones_like(reconstructed)
    clean_losses=fc.channel_losses(prediction,clean);noisy_losses=fc.channel_losses(prediction,noisy)
    assert clean_losses.shape==(4,) and not torch.equal(clean_losses,noisy_losses)
    three=fixture(active=3)[0].reconstruct(fixture(active=3)[2])
    assert not torch.equal(reconstructed,three)


def test_kendall_formula_and_negative_augmented_objective_is_valid():
    fc,_,_=fixture();fc.log_variances.data.copy_(torch.tensor([-2.,-.5,.25,1.],dtype=torch.float64))
    channels=torch.tensor([.01,.02,.03,.04],dtype=torch.float64)
    expected=.5*(torch.exp(-fc.log_variances)*channels+fc.log_variances).sum()
    actual=fc.kendall_loss(channels);assert torch.allclose(actual,expected,rtol=1e-12,atol=1e-14)
    total=compose_fc_objective(torch.tensor(0.,dtype=torch.float64),actual)
    assert total<0 and torch.isfinite(total)


def test_surrogate_firewall_and_fc_only_coordinate_gradient():
    fc,raw,prediction=fixture();clean=torch.zeros(2,4,30,dtype=torch.float64)
    assert not fc.electric.training and not fc.magnetic.training
    frozen=[p for name,p in fc.named_parameters() if name!="log_variances"]
    assert frozen and all(not p.requires_grad and p.grad is None for p in frozen)
    loss,_=fc(prediction,clean);loss.backward()
    assert all(p.grad is None for p in frozen)
    for tensor in (raw.raw_radius,raw.cos_like,raw.sin_like):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum()>0


def test_physics_import_sentinel_and_surrogate_free_prediction(monkeypatch):
    fc,raw,prediction=fixture();original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.startswith("inverse_em.physics"): raise AssertionError("PhysicsTM entered FC graph")
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",guarded)
    fc(prediction,torch.zeros(2,4,30,dtype=torch.float64))
    assert not any(isinstance(value,FrozenSurrogate) for value in vars(prediction).values())


def test_fc_has_no_no_grad_coordinate_detach():
    fc,raw,prediction=fixture(active=1,batch=1)
    output=fc.reconstruct(prediction)
    gradient=torch.autograd.grad(output.square().mean(),(raw.raw_radius,raw.cos_like,raw.sin_like),allow_unused=False)
    assert all(torch.isfinite(value).all() and value.abs().sum()>0 for value in gradient)
