import torch
import pytest
from torch import nn

from inverse_em.config.localization import LocalizationScientificConfig
from inverse_em.localization import (CircularLocalizer, RawLocalizerOutput, active_slot_mask, decode_angle,
                                     decode_radius, make_localizer, parameter_count, scientific_view)


def raw(batch=2):
    return RawLocalizerOutput(torch.zeros(batch,3,dtype=torch.float64),
                              torch.tensor([[1.,0.,-1.]],dtype=torch.float64).repeat(batch,1),
                              torch.tensor([[0.,1.,0.]],dtype=torch.float64).repeat(batch,1))


def test_exact_architecture_and_parameter_count():
    model=make_localizer(); layers=list(model.features)
    assert [type(x) for x in layers]==[nn.Conv1d,nn.LeakyReLU,nn.Conv1d,nn.LeakyReLU,nn.Conv1d,nn.LeakyReLU]
    convolutions=layers[::2]
    assert [(x.in_channels,x.out_channels,x.kernel_size,x.stride,x.padding,x.dilation,x.groups,x.bias is not None,x.padding_mode) for x in convolutions]==[
        (4,32,(5,),(1,),(2,),(1,),1,True,"circular"),(32,64,(5,),(1,),(2,),(1,),1,True,"circular"),(64,96,(3,),(1,),(1,),(1,),1,True,"circular")]
    assert all(x.negative_slope==.01 and not x.inplace for x in [layers[1],layers[3],layers[5],model.dense[2]])
    assert isinstance(model.dense[0],nn.Flatten) and (model.dense[1].in_features,model.dense[1].out_features)==(2880,128)
    assert (model.radius_head.in_features,model.radius_head.out_features)==(128,3)
    assert (model.angle_head.in_features,model.angle_head.out_features)==(128,6)
    assert not any(isinstance(x,(nn.Dropout,nn.BatchNorm1d)) for x in model.modules())
    assert parameter_count(model)==399_433


def test_cpu_float64_forward_raw_three_slots():
    model=make_localizer(); result=model(torch.zeros(2,4,30,dtype=torch.float64))
    assert isinstance(result,RawLocalizerOutput)
    assert all(x.shape==(2,3) and x.dtype is torch.float64 and x.device.type=="cpu" for x in (result.raw_radius,result.cos_like,result.sin_like))
    with pytest.raises(Exception): model(torch.zeros(2,4,30))
    with pytest.raises(Exception): model(torch.zeros(2,120,dtype=torch.float64))


def test_radius_formula_ordinary_extreme_no_repair_and_gradient():
    ordinary=torch.tensor([[-3.,0.,3.]],dtype=torch.float64,requires_grad=True)
    decoded=decode_radius(ordinary); expected=.05+.90*torch.sigmoid(ordinary)
    assert torch.equal(decoded,expected) and torch.all((decoded>.05)&(decoded<.95))
    decoded.sum().backward(); assert torch.isfinite(ordinary.grad).all() and torch.all(ordinary.grad>0)
    extreme=torch.tensor([[-1000.,1000.]],dtype=torch.float64)
    actual=decode_radius(extreme); direct=.05+.90*torch.sigmoid(extreme)
    assert torch.equal(actual,direct) and actual[0,0]==.05 and actual[0,1]>=.95


def test_atan2_without_unit_circle_projection():
    c=torch.tensor([[2.,0.,-3.]],dtype=torch.float64);s=torch.tensor([[0.,4.,0.]],dtype=torch.float64)
    assert torch.equal(decode_angle(c,s),torch.atan2(s,c))
    view=scientific_view(RawLocalizerOutput(torch.zeros_like(c),c,s),3)
    assert torch.equal(view.cos_like,c) and torch.equal(view.sin_like,s)
    assert not torch.allclose(view.cos_like.square()+view.sin_like.square(),torch.ones_like(c))


@pytest.mark.parametrize("count,expected",[(1,[[True,False,False]]),(2,[[True,True,False]]),(3,[[True,True,True]])])
def test_active_masks_and_public_scientific_slicing(count,expected):
    mask=active_slot_mask(1,count);view=scientific_view(raw(1),count)
    assert mask.tolist()==expected and view.rho.shape==(1,count) and view.cos_like.shape==(1,count)
    assert not hasattr(view,"raw_radius")


def test_configuration_is_frozen_and_canonically_hashed():
    config=LocalizationScientificConfig()
    assert len(config.sha256)==64 and config.sha256==LocalizationScientificConfig().sha256
    assert config.model.negative_slope==.01 and config.loss.field_consistency_coefficient==.03
    assert config.canonical_correspondence=="fixed_slot" and config.matching_role=="diagnostic_only"
