from dataclasses import replace
import numpy as np
import pytest
import torch
from inverse_em.config.s1 import SeedRole
from inverse_em.config.phase2 import InverseTask
from inverse_em.localization import CanonicalTargets, RawLocalizerOutput
from inverse_em.localization import s1
from inverse_em.observations import ComplexFields, complex_fields_to_channels
from .conftest import fc_fixture


def test_population_reuse_and_analytical_fixture():
    pop=s1.tiny_population(3,871)
    assert pop.rho.shape==(3,1) and np.all((pop.rho>=.05)&(pop.rho<.95))
    assert np.all(pop.amplitude==1) and np.all((pop.phi>=0)&(pop.phi<2*np.pi))
    u=np.random.Generator(np.random.PCG64DXSM(871)).random((3,2))
    assert np.allclose(pop.rho[:,0],np.sqrt(.0025+.9*u[:,0]),rtol=1e-12,atol=1e-14)
    calls=[]
    class Forward:
        def evaluate(self,source,angles):
            calls.append((source,angles))
            return ComplexFields(np.full(30,source.rho+1j),np.full(30,source.phi+2j))
    fields=s1.analytical_fixture(pop,Forward())
    assert len(fields)==len(calls)==3
    assert np.array_equal(calls[0][1],2*np.pi*np.arange(30)/30)


@pytest.mark.parametrize('count,seed',[(70000,871),(15000,871),(17,871),(2,20261811),(2,20261816),(2,20261817)])
def test_production_population_rejected_before_generator(count,seed):
    with pytest.raises(PermissionError): s1.tiny_population(count,seed)


def test_normalizer_entrypoint_rejects_before_production(monkeypatch):
    import inverse_em.normalization as public
    import inverse_em.normalization.api as implementation
    def forbidden(*args, **kwargs):
        pytest.fail('Production fitting/generation boundary entered')
    monkeypatch.setattr(public, 'fit_clean_training_normalizer', forbidden)
    monkeypatch.setattr(implementation, 'fit_clean_training_normalizer', forbidden)
    monkeypatch.setattr(implementation, '_frozen_training_populations', forbidden)
    assert not hasattr(s1, 'fit_clean_training_normalizer')
    with pytest.raises(PermissionError, match='disabled'):
        s1.production_normalizer_entrypoint()()
    plan = s1.production_normalizer_plan()
    assert not callable(plan) and not any(callable(v) for v in plan.values())
    with pytest.raises(TypeError): plan()


def test_normalizer_plan_is_immutable_scientific_metadata():
    plan = s1.production_normalizer_plan()
    assert plan['api']=='inverse_em.normalization.fit_clean_training_normalizer'
    assert plan['task'] is InverseTask.S1
    assert plan['inputs']=='clean analytical training observations only'
    assert (plan['training_count'],plan['angles'],plan['ddof'])==(70000,30,0)
    assert plan['channels']==('Re(E)','Im(E)','Re(H)','Im(H)')
    assert plan['statistics']==('mean','population standard deviation')
    assert plan['production_execution_enabled'] is False
    with pytest.raises(TypeError): plan['production_execution_enabled']=True


def test_training_noise_uses_common_physics_and_preserves_clean_target(trainer_factory,monkeypatch):
    t=trainer_factory(); events=[]
    scale=s1.noise.scale_directions; add=s1.noise.add_scaled_noise
    def scale_spy(f,d,g): events.append(('scale',g)); return scale(f,d,g)
    def add_spy(f,n): events.append(('add',n.electric_power,n.magnetic_power)); return add(f,n)
    monkeypatch.setattr(s1.noise,'scale_directions',scale_spy)
    monkeypatch.setattr(s1.noise,'add_scaled_noise',add_spy)
    x,clean=s1.training_tensors(t.training.fields,t.training.ids,t.normalizer,1)
    assert [e[0] for e in events]==['scale','add','scale','add']
    expected=np.stack([complex_fields_to_channels(f) for f in t.training.fields])
    assert np.array_equal(clean.numpy(),expected)
    for i,f in enumerate(t.training.fields):
        gamma,d=s1.noise_event(t.training.ids[i],1,i)
        E=f.electric+np.sqrt(np.mean(abs(f.electric)**2)*10**(-gamma/10)/2)*(d.electric_real+1j*d.electric_imag)
        H=f.magnetic+np.sqrt(np.mean(abs(f.magnetic)**2)*10**(-gamma/10)/2)*(d.magnetic_real+1j*d.magnetic_imag)
        raw=np.stack([E.real,E.imag,H.real,H.imag])
        assert np.allclose(x[i].numpy(),t.normalizer.transform(raw),rtol=1e-12,atol=1e-14)
    assert not np.array_equal(x.numpy(),t.normalizer.transform(expected))


def test_clean_validation_cannot_augment(trainer_factory,monkeypatch):
    t=trainer_factory()
    def forbidden(*a,**k): pytest.fail('Validation entered augmentation')
    monkeypatch.setattr(s1,'noise_event',forbidden)
    metrics=s1.validate_clean(t.model,t.validation.fields,t.validation.targets,t.normalizer)
    assert metrics['source_count']==2 and metrics['sample_count']==2
    assert metrics['aggregation']=='pooled_active_sources_fixed_canonical_slots'
    assert not t.model.training and np.isfinite(metrics['cartesian']['rmse'])


def test_slot_zero_loss_and_fc_gradients():
    raw=RawLocalizerOutput(torch.tensor([[0.,100.,-100.]],dtype=torch.float64,requires_grad=True),
        torch.tensor([[.5,50.,-50.]],dtype=torch.float64,requires_grad=True),
        torch.tensor([[.2,-90.,90.]],dtype=torch.float64,requires_grad=True))
    class Fixed(torch.nn.Module):
        def forward(self,x): return raw
    fc=fc_fixture(); targets=CanonicalTargets(torch.tensor([[.3]],dtype=torch.float64),torch.tensor([[.4]],dtype=torch.float64))
    total,sup,K,ch=s1.objective(Fixed(),None,targets,torch.zeros(1,4,30,dtype=torch.float64),fc)
    expected=(.5-.3)**2+2*((.5-np.cos(.4))**2+(.2-np.sin(.4))**2)+.1*(.5**2+.2**2-1)**2
    assert float(sup.detach())==pytest.approx(expected,rel=1e-12,abs=1e-14)
    assert torch.equal(K,.5*ch.sum()) and torch.equal(total,sup+.03*K)
    K.backward()
    for tensor in (raw.raw_radius,raw.cos_like,raw.sin_like):
        assert tensor.grad[0,0]!=0 and torch.equal(tensor.grad[:,1:],torch.zeros(1,2,dtype=torch.float64))
    assert fc.log_variances.grad is not None
    assert all(p.grad is None for f in (fc.electric,fc.magnetic) for p in f.parameters())


@pytest.mark.parametrize('role',[SeedRole.SEALED_RESERVED,SeedRole.ROBUSTNESS_RESERVED])
def test_protected_roles_cannot_become_fixture(trainer_factory,role):
    with pytest.raises(PermissionError): replace(trainer_factory().training,role=role)
    with pytest.raises(PermissionError): s1.population_plan(role)
    with pytest.raises(PermissionError): s1.noise_event('fixture',1,role=role)
