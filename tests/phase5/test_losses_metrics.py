import math
import numpy as np
import pytest
import torch

from inverse_em.localization import (CanonicalTargets, DiagnosticAssignment, ScientificPrediction, canonical_errors,
    canonical_metrics, localization_components, reduce_outer_s3, reduce_s1, reduce_s2_canonical, summarize)


def prediction(rho,phi,scale=1.):
    rho=torch.tensor(rho,dtype=torch.float64);phi=torch.tensor(phi,dtype=torch.float64)
    return ScientificPrediction(rho,scale*torch.cos(phi),scale*torch.sin(phi),phi)


def targets(rho,phi):return CanonicalTargets(torch.tensor(rho,dtype=torch.float64),torch.tensor(phi,dtype=torch.float64))


def test_elemental_components_uniform_and_inactive_exclusion():
    pred=prediction([[.2],[.5]],[[0.],[math.pi/2]],scale=.5);target=targets([[.3],[.4]],[[0.],[0.]])
    c=localization_components(pred,target)
    assert torch.allclose(c.rho,torch.tensor([[.01],[.01]],dtype=torch.float64),rtol=1e-12,atol=1e-14)
    expected_phi=(pred.cos_like-torch.cos(target.phi)).square()+(pred.sin_like-torch.sin(target.phi)).square()
    assert torch.equal(c.phi,expected_phi) and torch.equal(c.circle,torch.full((2,1),.5625,dtype=torch.float64))
    assert torch.equal(c.uniform,c.rho+2*c.phi+.1*c.circle) and reduce_s1(c)==c.uniform.mean()


def test_s2_canonical_fixed_slot_reduction():
    pred=prediction([[.2,.8]],[[0.,math.pi]]);target=targets([[.8,.2]],[[math.pi,0.]])
    c=localization_components(pred,target);actual=reduce_s2_canonical(c)
    assert torch.equal(actual,c.uniform.sum(1).mean()) and actual>0


def test_outer_weights_rho_phi_but_not_circle():
    pred=prediction([[.2,.4,.6]],[[0.,.2,.4]],scale=.5);target=targets([[.1,.5,.9]],[[.1,.3,.5]])
    c=localization_components(pred,target);w=(1+target.rho)/1.635
    expected=(w*c.rho).mean()+2*(w*c.phi).mean()+.1*c.circle.mean()
    incorrectly_weighted=(w*(c.rho+2*c.phi+.1*c.circle)).mean()
    assert torch.allclose(reduce_outer_s3(c,target),expected,rtol=1e-12,atol=1e-14)
    assert not torch.allclose(expected,incorrectly_weighted)


def test_canonical_metrics_pool_only_active_slots():
    pred=prediction([[.2,.4],[.6,.8]],[[0.,0.],[0.,0.]])
    target=targets([[.1,.2],[.3,.4]],[[0.,0.],[0.,0.]])
    errors=canonical_errors(pred,target)
    expected=torch.tensor([[.1,.2],[.3,.4]],dtype=torch.float64)
    assert torch.allclose(errors["cartesian"],expected,rtol=1e-12,atol=1e-14)
    result=canonical_metrics(pred,target)
    assert result["sample_count"]==2 and result["source_count"]==4
    assert result["aggregation"]=="pooled_active_sources_fixed_canonical_slots"
    assert result["cartesian"]==summarize(expected)
    assert result["angular_radians"]["max"]==0.


def test_summary_mean_median_rmse_percentiles_and_max():
    x=np.array([0.,1.,2.,3.,4.],np.float64);s=summarize(x)
    assert s=={"mean":2.,"median":2.,"rmse":float(np.sqrt(6)),"p95":3.8,"p99":3.96,"max":4.}


def test_canonical_loss_type_firewall_rejects_diagnostic_objects():
    pred=prediction([[.2,.4]],[[0.,1.]])
    diagnostic=DiagnosticAssignment(torch.zeros(1,dtype=torch.int64),((0,1),),torch.zeros(1,2,2,dtype=torch.float64),torch.zeros(1,1,dtype=torch.float64),torch.zeros(1,dtype=torch.bool))
    with pytest.raises(TypeError): localization_components(pred,diagnostic)
