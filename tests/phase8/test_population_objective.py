import hashlib
import numpy as np
import pytest
import torch
from inverse_em.config.s3 import SeedRole
from inverse_em.localization import s3, CanonicalTargets
from inverse_em.localization.losses import localization_components, reduce_outer_s3
from inverse_em.localization.decoding import RawLocalizerOutput, decode_scientific
from inverse_em.localization.matching import diagnostic_assignment
from inverse_em.populations import generation, PopulationRole
from .conftest import fc_fixture


@pytest.mark.parametrize('stage', range(1,9))
def test_threshold_boundaries(stage):
    _, dr, dp = s3.stage_contract(stage)
    for threshold in (dr,float(np.deg2rad(dp))):
        edge = threshold
        for _ in range(8):
            edge = np.nextafter(edge,-np.inf)
        assert generation.accepts_threshold(threshold,threshold)
        assert generation.accepts_threshold(edge,threshold)
        assert not generation.accepts_threshold(np.nextafter(edge,-np.inf),threshold)


@pytest.mark.parametrize('count,seed,role', [(70000,20262201,PopulationRole.TRAINING),
    (15000,20262202,PopulationRole.VALIDATION), (2,20262206,PopulationRole.SEALED),
    (2,20262207,'ROBUSTNESS'), (17,81004,PopulationRole.REFERENCE_FIXTURE)])
def test_real_protected_surface_zero_reach(count,seed,role,monkeypatch):
    hits = []
    def stop(*args,**kwargs):
        hits.append(1)
        raise AssertionError('Phase1 reached')
    monkeypatch.setattr(generation,'generate_interleaved',stop)
    with pytest.raises(PermissionError):
        s3.tiny_population(count,seed,8,role)
    assert hits == []
    assert not any(v is generation.generate_s3 for v in vars(s3).values())
    assert not hasattr(s3,'generate_s3')


@pytest.mark.parametrize('seed', range(20262201,20262208))
def test_seed_cannot_be_disguised_as_fixture(seed):
    with pytest.raises(PermissionError):
        s3.tiny_population(1,seed,8)


def test_phase1_reuse_rejection_and_dedup(monkeypatch):
    draws = []
    bad = np.array([[.1,.1],[.1,.4],[.1,.8]])
    good = np.array([[.9,.8],[.1,.1],[.4,.4]])
    other = np.array([[.8,.85],[.2,.15],[.5,.45]])
    proposals = iter([bad,good,good,other])
    class Generator:
        def random(self,shape):
            draws.append(shape)
            return next(proposals)
    monkeypatch.setattr(generation.np.random,'Generator',lambda seed: Generator())
    p = s3.tiny_population(2,81004,8)
    assert draws == [(3,2)]*4 and p.proposal_ordinal.tolist() == [2,4]
    assert p.rho.shape == (2,3) and np.all(p.amplitude==1)
    assert np.all(np.diff(p.rho,axis=1)>0)
    r,ph = generation.canonicalize([.3,.3,.7],[2.,1.,0.])
    assert ph.tolist()==[1.,2.,0.]


def test_normalizer_plans_and_firewall(firewall):
    p = s3.production_normalizer_plan()
    assert p['scalar_count_per_channel']==16800000 and p['ddof']==0
    assert not any(callable(v) for v in p.values())
    with pytest.raises(TypeError):
        p['ddof']=1
    for fn in (s3.production_normalizer_entrypoint,s3.production_training_entrypoint,s3.protected_evaluation):
        with pytest.raises(PermissionError):
            fn()
    assert not any(firewall.values())


def test_tiny_analytical_superposition_reuses_forward_interface():
    p=s3.tiny_population(2,81004,8)
    calls=[]
    class Forward:
        def evaluate(self, source, theta):
            calls.append((source,theta))
            return s3.ComplexFields(np.full(30,1+2j,np.complex128),np.full(30,3+4j,np.complex128))
    fields=s3.analytical_fixture(p,Forward())
    assert len(calls)==6 and len(fields)==2
    np.testing.assert_array_equal(calls[0][1],2*np.pi*np.arange(30)/30)
    np.testing.assert_array_equal(s3.complex_fields_to_channels(fields[0]),np.tile(np.array([3.,6.,9.,12.])[:,None],(1,30)))


def test_duplicate_fixture_and_derived_seed_rejected(trainer_factory):
    from dataclasses import replace
    from inverse_em.errors import SchemaValidationError
    d=trainer_factory().stages[0]
    with pytest.raises(SchemaValidationError):
        replace(d,ids=(d.ids[0],d.ids[0]))
    with pytest.raises(PermissionError):
        s3.tiny_population(1,s3.stage_population_seed(20262201,1),1)


def test_configuration_ids_and_duplicates(trainer_factory):
    t = trainer_factory()
    d = t.stages[0]
    r,p = d.targets.rho[0].numpy(),d.targets.phi[0].numpy()
    h = hashlib.sha256(np.stack((r,p),axis=1).astype('<f8').tobytes()).hexdigest()
    assert d.ids[0] == 'S3ANF-train-ST01-'+h[:24]
    assert s3.configuration_id(SeedRole.TRAIN_POPULATION,1,r,p)==d.ids[0]


def test_independent_noise_draw_order_and_phase2_reuse(trainer_factory,monkeypatch):
    t=trainer_factory();d=t.stages[0];calls=[];streams=[]
    original=s3._rng;scale=s3.noise.scale_directions;add=s3.noise.add_scaled_noise
    def observe(*args):
        streams.append(args[-1]);return original(*args)
    def scale_spy(*args):
        calls.append('scale');return scale(*args)
    def add_spy(*args):
        calls.append('add');return add(*args)
    monkeypatch.setattr(s3,'_rng',observe)
    monkeypatch.setattr(s3.noise,'scale_directions',scale_spy)
    monkeypatch.setattr(s3.noise,'add_scaled_noise',add_spy)
    got,gammas=s3.augment_field(d.fields[0],d.ids[0],1,1)
    assert streams==['E_SNR','E_NOISE','H_SNR','H_NOISE']
    assert calls==['scale','add','scale','add']
    assert gammas[0]!=gammas[1] and all(30<=g<40 for g in gammas)
    expected=[]
    for label,F,gamma in zip(('E','H'),(d.fields[0].electric,d.fields[0].magnetic),gammas):
        z=original(SeedRole.TRAINING_AUGMENTATION,1,1,0,d.ids[0],label+'_NOISE').standard_normal((30,2))
        value=F+np.sqrt(np.mean(abs(F)**2)*10**(-gamma/10)/2)*(z[:,0]+1j*z[:,1])
        expected.extend((value.real,value.imag))
    np.testing.assert_allclose(got,np.stack(expected),rtol=1e-12,atol=1e-14)
    x,clean,receipt=s3.training_tensors(d.fields,d.ids,t.normalizer,1,1)
    np.testing.assert_array_equal(clean[0].numpy(),s3.complex_fields_to_channels(d.fields[0]))
    np.testing.assert_array_equal(x[0].numpy(),t.normalizer.transform(got))
    assert len(receipt['gammas_E_H'])==2


def test_outer_unweighted_circle_and_fc_gradients():
    raw=RawLocalizerOutput(torch.zeros((2,3),dtype=torch.float64,requires_grad=True),
        torch.full((2,3),.7,dtype=torch.float64,requires_grad=True),
        torch.full((2,3),.4,dtype=torch.float64,requires_grad=True))
    pred=decode_scientific(raw,3)
    r=torch.tensor([[.1,.4,.8],[.2,.5,.9]],dtype=torch.float64)
    target=CanonicalTargets(r,torch.ones_like(r))
    c=localization_components(pred,target);w=(1+r)/1.635
    expected=(w*c.rho).mean()+2*(w*c.phi).mean()+.1*c.circle.mean()
    got=reduce_outer_s3(c,target)
    assert torch.equal(got,expected) and c.circle.mean()>0
    assert not torch.isclose(got,(w*c.uniform).mean(),rtol=1e-12,atol=1e-14)
    fc=fc_fixture();k,ch=fc(pred,torch.zeros((2,4,30),dtype=torch.float64))
    assert torch.equal(k,.5*ch.sum())
    (got+.03*k).backward()
    assert all(torch.all(v.grad!=0) for v in (raw.raw_radius,raw.cos_like,raw.sin_like))
    assert fc.log_variances.grad is not None
    assert all(not p.requires_grad and p.grad is None for m in (fc.electric,fc.magnetic) for p in m.parameters())
    assignment=diagnostic_assignment(pred,target)
    with pytest.raises(TypeError):
        localization_components(pred,assignment)


def test_clean_validation_torch_score_no_matching(trainer_factory,monkeypatch):
    t=trainer_factory()
    def stop(*a,**k):
        raise AssertionError('Noise or matching in principal validation')
    import inverse_em.localization.matching as matching
    monkeypatch.setattr(s3,'augment_field',stop)
    monkeypatch.setattr(matching,'diagnostic_assignment',stop)
    score,metrics=s3.validate_clean(t.model,t.validation.fields,t.validation.targets,t.normalizer)
    with torch.no_grad():
        x=torch.from_numpy(t.normalizer.transform(np.stack([s3.complex_fields_to_channels(f) for f in t.validation.fields])))
        pred=s3.scientific_view(t.model(x),3)
        expected=float(torch.sqrt(s3.canonical_errors(pred,t.validation.targets)['cartesian'].square().mean()))
    assert score==expected and metrics['source_count']==6
    same=CanonicalTargets(torch.full((1,3),.4,dtype=torch.float64),torch.ones((1,3),dtype=torch.float64))
    p=decode_scientific(RawLocalizerOutput(torch.zeros((1,3),dtype=torch.float64),torch.ones((1,3),dtype=torch.float64),torch.zeros((1,3),dtype=torch.float64)),3)
    assert diagnostic_assignment(p,same).chosen_indices.item()==0
