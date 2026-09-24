from types import SimpleNamespace
import numpy as np
import pytest
import torch
from inverse_em.config.s2 import SeedRole
from inverse_em.errors import SchemaValidationError
from inverse_em.localization import s2
from inverse_em.localization.decoding import RawLocalizerOutput, decode_scientific
from inverse_em.localization.losses import CanonicalTargets, localization_components, reduce_s2_canonical
from inverse_em.localization.matching import diagnostic_assignment
from inverse_em.localization.metrics import canonical_metrics
from inverse_em.observations import complex_fields_to_channels
from inverse_em.populations import PopulationRole, generate_s2, generation
from inverse_em.populations.contracts import SeparationConstraint
from .conftest import fc_fixture


@pytest.mark.parametrize("threshold", [.05,float(np.deg2rad(5))])
def test_eight_ulp_boundary(threshold):
    inside = threshold
    for _ in range(8):
        inside = np.nextafter(inside,-np.inf)
    outside = np.nextafter(inside,-np.inf)
    assert generation.accepts_threshold(threshold,threshold)
    assert generation.accepts_threshold(inside,threshold)
    assert not generation.accepts_threshold(outside,threshold)


def test_both_pair_constraints_and_canonical_tie():
    sep = SeparationConstraint(.05,5.)
    assert generation.admissible(np.array([.2,.6]),np.array([.1,1.]),sep)
    assert not generation.admissible(np.array([.2,.21]),np.array([.1,1.]),sep)
    assert not generation.admissible(np.array([.2,.6]),np.array([0.,2*np.pi-.01]),sep)
    r,p = generation.canonicalize([.3,.3],[2.,1.])
    assert p.tolist() == [1.,2.]


def test_sampler_reuse_and_whole_pair_rejection(monkeypatch):
    calls = []
    proposals = iter([np.array([[.1,.1],[.1,.7]]), np.array([[.9,.8],[.1,.1]])])
    class Generator:
        def random(self,shape):
            calls.append(shape)
            return next(proposals)
    monkeypatch.setattr(generation.np.random,"Generator",lambda seed: Generator())
    pop = s2.tiny_population(1,81003)
    assert calls == [(2,2),(2,2)] and pop.proposal_ordinal.tolist() == [2]
    assert pop.rho[0,0] < pop.rho[0,1]
    assert np.all(pop.amplitude == 1)


def test_analytical_complex_superposition():
    population = s2.tiny_population(2,81003)
    calls = []
    class Forward:
        def evaluate(self,source,theta):
            calls.append(theta.copy())
            return SimpleNamespace(electric=np.full(30,1+2j,np.complex128),
                                   magnetic=np.full(30,3+4j,np.complex128))
    fields = s2.analytical_fixture(population,Forward())
    assert len(calls) == 4
    np.testing.assert_array_equal(calls[0],2*np.pi*np.arange(30)/30)
    assert fields[0].electric.dtype == np.complex128
    np.testing.assert_array_equal(complex_fields_to_channels(fields[0]),
                                  np.tile(np.array([2.,4.,6.,8.])[:,None],(1,30)))


@pytest.mark.parametrize("count", [17,15000,70000])
def test_population_count_firewall(count,firewall):
    with pytest.raises(PermissionError):
        s2.tiny_population(count,81001)
    assert not any(firewall.values())


@pytest.mark.parametrize("count,seed,role", [
    (70000,20262001,PopulationRole.TRAINING),
    (15000,20262002,PopulationRole.VALIDATION),
])
def test_real_surface_rejects_before_generation(count,seed,role,monkeypatch):
    hits = []
    def stop(*args,**kwargs):
        hits.append((args,kwargs))
        raise AssertionError("Phase-1 generation boundary reached")
    monkeypatch.setattr(generation,"generate_interleaved",stop)
    with pytest.raises(PermissionError):
        s2.tiny_population(count,seed)
    with pytest.raises(AttributeError):
        getattr(s2,"_generate_s2")(count,seed,role)
    assert not any(value is generate_s2 for value in vars(s2).values())
    assert hits == []


def test_tiny_population_reaches_phase1_boundary(monkeypatch):
    original = generation.generate_interleaved
    calls = []
    def observe(definition,**kwargs):
        calls.append((definition.count,definition.rng.seed,definition.role))
        return original(definition,**kwargs)
    monkeypatch.setattr(generation,"generate_interleaved",observe)
    population = s2.tiny_population(2,81004)
    assert calls == [(2,81004,PopulationRole.REFERENCE_FIXTURE)]
    assert population.rho.shape == (2,2)


@pytest.mark.parametrize("seed", list(range(20262001,20262008)))
def test_population_seed_firewall(seed,firewall):
    with pytest.raises(PermissionError):
        s2.tiny_population(1,seed)
    assert not any(firewall.values())


def test_production_plans_and_entrypoint_firewalls(trainer_factory,monkeypatch,firewall):
    plans = [s2.population_plan(SeedRole.TRAIN_POPULATION), s2.production_normalizer_plan(),
             s2.production_training_plan()]
    for plan in plans:
        assert not plan["production_execution_enabled"]
        assert not any(callable(v) for v in plan.values())
        with pytest.raises(TypeError):
            plan["production_execution_enabled"] = True
    for fn in (s2.production_normalizer_entrypoint,s2.production_training_entrypoint,s2.protected_evaluation):
        with pytest.raises(PermissionError):
            fn()
    t = trainer_factory()
    def forbidden():
        firewall["production_training"] += 1
        raise AssertionError("Optimizer reached")
    monkeypatch.setattr(t.optimizer,"step",forbidden)
    with pytest.raises(PermissionError):
        t.run(600)
    for role in (SeedRole.SEALED_RESERVED,SeedRole.ROBUSTNESS_RESERVED):
        with pytest.raises(PermissionError):
            s2.population_plan(role)
    assert not any(firewall.values())


def test_noise_primitives_clean_target_and_normalization(trainer_factory,monkeypatch):
    t=trainer_factory()
    calls=[]
    scale,add=s2.noise.scale_directions,s2.noise.add_scaled_noise
    def spy_scale(*args):
        calls.append("scale")
        return scale(*args)
    def spy_add(*args):
        calls.append("add")
        return add(*args)
    monkeypatch.setattr(s2.noise,"scale_directions",spy_scale)
    monkeypatch.setattr(s2.noise,"add_scaled_noise",spy_add)
    x,raw=s2.training_tensors(t.training.fields,t.training.ids,t.normalizer,1)
    expected=np.stack([complex_fields_to_channels(f) for f in t.training.fields])
    np.testing.assert_array_equal(raw.numpy(),expected)
    assert calls == ["scale","add","scale","add"]
    assert not np.array_equal(x.numpy(),t.normalizer.transform(expected))
    gamma,z=s2.noise_event(t.training.ids[0],1)
    noisy=complex_fields_to_channels(add(t.training.fields[0],scale(t.training.fields[0],z,gamma)))
    np.testing.assert_array_equal(x[0].numpy(),t.normalizer.transform(noisy[None])[0])


def test_slot_mask_summed_loss_fc_gradient_and_matching_firewall(trainer_factory):
    t=trainer_factory()
    raw=RawLocalizerOutput(torch.tensor([[0.,1.,2.]],dtype=torch.float64,requires_grad=True),
        torch.tensor([[.8,.2,.4]],dtype=torch.float64,requires_grad=True),
        torch.tensor([[.4,.9,.6]],dtype=torch.float64,requires_grad=True))
    pred=decode_scientific(raw,2)
    target=CanonicalTargets(t.training.targets.rho[:1],t.training.targets.phi[:1])
    components=localization_components(pred,target)
    loc=reduce_s2_canonical(components)
    assert torch.equal(loc,components.uniform.sum(1).mean())
    assert torch.equal(loc,2*components.uniform.mean())
    fc=fc_fixture()
    target_fields=torch.zeros((1,4,30),dtype=torch.float64)
    k,ch=fc(pred,target_fields)
    assert torch.equal(fc.log_variances,torch.zeros(4,dtype=torch.float64))
    k.backward(retain_graph=True)
    for value in (raw.raw_radius,raw.cos_like,raw.sin_like):
        assert torch.any(value.grad[0,:2] != 0)
        assert value.grad[0,2] == 0
        value.grad = None
    fc.log_variances.grad = None
    (loc+.03*k).backward()
    for value in (raw.raw_radius,raw.cos_like,raw.sin_like):
        assert value.grad[0,2] == 0
        assert torch.any(value.grad[0,:2] != 0)
    assert fc.log_variances.grad is not None
    assert all(not p.requires_grad and p.grad is None for m in (fc.electric,fc.magnetic) for p in m.parameters())
    altered=RawLocalizerOutput(raw.raw_radius.detach().clone(),raw.cos_like.detach().clone(),raw.sin_like.detach().clone())
    altered.raw_radius[:,2]=900.
    altered.cos_like[:,2]=-900.
    other=decode_scientific(altered,2)
    assert other.rho.shape == (1,2)
    assert torch.equal(fc.reconstruct(pred),fc.reconstruct(other))
    assert canonical_metrics(pred,target) == canonical_metrics(other,target)
    assert torch.equal(loc,reduce_s2_canonical(localization_components(other,target)))
    matched=diagnostic_assignment(pred,target)
    with pytest.raises(TypeError):
        localization_components(pred,matched)
    with pytest.raises((SchemaValidationError,AttributeError)):
        s2.objective(t.model,torch.zeros((1,4,30),dtype=torch.float64),matched,target_fields,t.fc)


def test_diagnostic_swap_and_identity_tie():
    from inverse_em.localization.decoding import ScientificPrediction
    r=torch.tensor([[.2,.6]],dtype=torch.float64)
    p=torch.tensor([[.3,1.2]],dtype=torch.float64)
    target=CanonicalTargets(r,p)
    pred=ScientificPrediction(r.flip(1),torch.cos(p.flip(1)),torch.sin(p.flip(1)),p.flip(1))
    assert diagnostic_assignment(pred,target).chosen_indices.item() == 1
    assert canonical_metrics(pred,target)["cartesian"]["rmse"] > 0
    tied=CanonicalTargets(r[:,0:1].repeat(1,2),p[:,0:1].repeat(1,2))
    assert diagnostic_assignment(pred,tied).chosen_indices.item() == 0


def test_clean_validation_pooled_uneven_tail(trainer_factory,monkeypatch):
    t=trainer_factory()
    def forbidden(*args,**kwargs):
        raise AssertionError("Validation augmented fields")
    monkeypatch.setattr(s2,"noise_event",forbidden)
    # The production batch is 256; a two-case fixture exercises a retained
    # incomplete batch without relaxing the 16-case execution firewall.
    sizes=[]
    handle=t.model.register_forward_pre_hook(lambda m,args: sizes.append((len(args[0]),m.training,torch.is_grad_enabled())))
    result=s2.validate_clean(t.model,t.validation.fields,t.validation.targets,t.normalizer)
    handle.remove()
    with torch.no_grad():
        pred=decode_scientific(t.model(s2.clean_tensor(t.validation.fields,t.normalizer)),2)
        expected=canonical_metrics(pred,t.validation.targets)
    assert result == expected and result["source_count"] == 4
    assert sizes == [(2,False,False)]


def test_target_and_normalizer_guard(trainer_factory):
    from inverse_em.config.phase2 import InverseTask
    from inverse_em.normalization import FrozenChannelNormalizer
    t=trainer_factory()
    bad=CanonicalTargets(t.training.targets.rho.flip(1),t.training.targets.phi.flip(1))
    with pytest.raises(SchemaValidationError):
        s2.validate_targets(bad,2)
    wrong=FrozenChannelNormalizer(InverseTask.S1,np.zeros(4),np.ones(4),2,60)
    with pytest.raises(SchemaValidationError):
        s2.clean_tensor(t.training.fields,wrong)
