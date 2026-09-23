import torch
import pytest
from inverse_em.config.surrogate import MODEL_SEEDS
from inverse_em.surrogates.features import relative_angle_features
from inverse_em.surrogates.inference import FrozenSurrogate,freeze_surrogate,superpose_complex
from inverse_em.surrogates.model import BoundaryFieldMLP,make_initialized_model

def t(value,requires_grad=False):return torch.tensor(value,dtype=torch.float64,requires_grad=requires_grad)

def test_relative_features_values_shape_and_broadcast():
    x=relative_angle_features(t([.2,.4]),t(.1),t([.3,.5]));expected=torch.stack((t([.2,.4]),torch.cos(t([.2,.4])),torch.sin(t([.2,.4]))),-1)
    assert x.shape==(2,3) and torch.allclose(x,expected,rtol=0,atol=1e-16)

def test_features_reject_non_float64_or_nonfinite():
    with pytest.raises(Exception):relative_angle_features(torch.tensor(.2),t(.1),t(.3))
    with pytest.raises(Exception):relative_angle_features(t(float("nan")),t(.1),t(.3))

def test_model_architecture_parameter_count_and_independence():
    e=BoundaryFieldMLP();h=BoundaryFieldMLP()
    assert sum(p.numel() for p in e.parameters())==50306 and sum(p.numel() for p in h.parameters())==50306
    assert sum(p.numel() for p in e.parameters())+sum(p.numel() for p in h.parameters())==100612
    assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(e.parameters(),h.parameters()))
    assert all(p.dtype is torch.float64 and p.device.type=="cpu" for p in e.parameters())
    assert [type(x).__name__ for x in e.net]==["Linear","ReLU","Linear","ReLU","Linear","ReLU","Linear","ReLU","Linear"]

def test_initialization_is_deterministic_seeded_and_preserves_global_rng():
    before=torch.get_rng_state().clone();a,receipt=make_initialized_model(MODEL_SEEDS[0]);after=torch.get_rng_state();b,_=make_initialized_model(MODEL_SEEDS[0]);c,_=make_initialized_model(MODEL_SEEDS[1])
    assert torch.equal(before,after) and all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),b.state_dict().values()))
    assert any(not torch.equal(x,y) for x,y in zip(a.state_dict().values(),c.state_dict().values()))
    assert torch.count_nonzero(a.net[0].bias)==0 and receipt["linear_0.bias"]=="exact_zero"

def test_periodicity_joint_rotation_and_differential_relation():
    model,_=make_initialized_model(MODEL_SEEDS[0]);api=FrozenSurrogate(model);rho=t(.43,True);phi=t(.71,True);theta=t(1.37,True);alpha=t(.37);base=api.predict_components(rho,phi,theta)
    assert torch.allclose(base,api.predict_components(rho,phi+2*torch.pi,theta),rtol=1e-12,atol=1e-12)
    assert torch.allclose(base,api.predict_components(rho,phi,theta+2*torch.pi),rtol=1e-12,atol=1e-12)
    assert torch.allclose(base,api.predict_components(rho,phi+alpha,theta+alpha),rtol=1e-12,atol=1e-12)
    gphi,gtheta=torch.autograd.grad(base.sum(),(phi,theta));assert torch.isfinite(gphi) and torch.isfinite(gtheta) and torch.allclose(gphi+gtheta,t(0.),rtol=0,atol=1e-12)

def test_frozen_parameters_and_coordinate_finite_difference():
    model,_=make_initialized_model(MODEL_SEEDS[0]);freeze_surrogate(model);api=FrozenSurrogate(model);rho=t(.43,True);phi=t(.71,True);theta=t(1.37,True);grads=torch.autograd.grad(api.predict_components(rho,phi,theta).sum(),(rho,phi,theta))
    assert all(torch.isfinite(g) and g.abs()>0 for g in grads);step=1e-6
    def f(r,p,q):return api.predict_components(t(r),t(p),t(q)).sum()
    points=(rho.item(),phi.item(),theta.item());finite=[]
    for i in range(3):
        plus=list(points);minus=list(points);plus[i]+=step;minus[i]-=step;finite.append((f(*plus)-f(*minus))/(2*step))
    assert torch.allclose(torch.stack(grads),torch.stack(finite),rtol=1e-6,atol=1e-8)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())

class PredictedComplex(torch.nn.Module):
    def __init__(self,values):super().__init__();self.values=values
    def predict_complex(self,rho,phi,theta):return self.values

def test_superpose_complex_uses_explicit_source_axis_and_preserves_angles():
    values=torch.tensor([[1+1j,2+2j,3+3j],[10+10j,20+20j,30+30j]],dtype=torch.complex128)
    actual=superpose_complex(PredictedComplex(values),None,None,None)
    assert actual.shape==(3,) and actual.dtype==torch.complex128 and torch.equal(actual,torch.tensor([11+11j,22+22j,33+33j],dtype=torch.complex128))
    batched=values.repeat(2,1,1);assert torch.equal(superpose_complex(PredictedComplex(batched),None,None,None),batched.sum(dim=1))
    one=values[:1];assert torch.equal(superpose_complex(PredictedComplex(one),None,None,None),one[0])

def test_superpose_complex_configurable_axis_invalid_axis_and_gradients():
    real=torch.arange(12,dtype=torch.float64).reshape(4,3).requires_grad_();imag=torch.ones((4,3),dtype=torch.float64,requires_grad=True);values=torch.complex(real,imag)
    result=superpose_complex(PredictedComplex(values),None,None,None,source_dim=1)
    assert result.shape==(4,) and torch.equal(result,values.sum(dim=1))
    result.real.sum().backward();assert torch.equal(real.grad,torch.ones_like(real)) and torch.equal(imag.grad,torch.zeros_like(imag))
    for invalid in (2,-3,"sources"):
        with pytest.raises(ValueError): superpose_complex(PredictedComplex(values.detach()),None,None,None,source_dim=invalid)
