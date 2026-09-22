import hashlib,math
from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from inverse_em.populations import *
from inverse_em.populations.generation import generate_interleaved
import inverse_em.populations.generation as population_generation

def digest(p):return hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in (p.rho,p.phi,p.amplitude,p.proposal_ordinal))).hexdigest()
def test_rng_identity_and_global_independence():
    assert RNGIdentity("NumPy PCG64DXSM",1).algorithm=="NumPy PCG64DXSM"
    np.random.seed(9);before=np.random.get_state();generate_s1(3,740001);after=np.random.get_state();assert np.array_equal(before[1],after[1]) and before[2:]==after[2:]
def test_area_uniform_transform_and_bounds():
    law=SourceLaw(.05,.95);u=np.array([0.,.25,.5,.75,np.nextafter(1.,0.)]);r=area_uniform_radius(u,law)
    assert r[0]==.05 and np.all(r>=.05) and np.all(r<.95)
    normalized=(r*r-.05**2)/(.95**2-.05**2);assert np.allclose(normalized,u,rtol=0,atol=2e-16)
def test_area_distribution_sanity():
    p=generate_s1(10000,740010);u=(p.rho[:,0]**2-.05**2)/(.95**2-.05**2);assert abs(float(u.mean())-.5)<.01
def test_s1_fixture_and_unit_amplitude():
    p=generate_s1(4,740001);assert digest(p)=="11e6721992ad967a93b0f7f843d39d242fb81b25a319680f358a9afa96c74429" and np.all(p.amplitude==1)
def test_s2_fixture_rejection_and_canonicalization():
    p=generate_s2(4,740002);assert digest(p)=="4066fdabed43cadf250b369c6a80a6ce11a00fde2c37af2901ad50f33641cdac"
    assert np.array_equal(p.proposal_ordinal,np.array([1,2,4,5],np.uint64)) and np.all(np.diff(p.rho,axis=1)>=0)
    assert all(admissible(r,phi,SeparationConstraint(.05,5)) for r,phi in zip(p.rho,p.phi))
def test_exact_tie_secondary_phi_order():
    r,p=canonicalize(np.array([.3,.3,.2]),np.array([2.,1.,4.]));assert np.array_equal(r,[.2,.3,.3]) and np.array_equal(p,[4.,1.,2.])
@pytest.mark.parametrize("threshold",[.05,.06,.07,.08,.10])
def test_radial_eight_ulp_boundaries(threshold):
    atol=eight_ulp_atol(threshold);assert accepts_threshold(threshold,threshold) and accepts_threshold(np.nextafter(threshold,np.inf),threshold)
    assert accepts_threshold(threshold-atol,threshold) and not accepts_threshold(np.nextafter(threshold-atol,-np.inf),threshold)
@pytest.mark.parametrize("degrees",[5.,6.,8.,10.,15.,20.,25.,40.])
def test_angular_eight_ulp_boundaries(degrees):
    t=float(np.deg2rad(np.float64(degrees)));atol=eight_ulp_atol(t);assert accepts_threshold(t,t) and accepts_threshold(t-atol,t) and not accepts_threshold(np.nextafter(t-atol,-np.inf),t)
def test_expected_s2_tolerances():
    assert eight_ulp_atol(.05)==5.551115123125783e-17
    assert eight_ulp_atol(float(np.deg2rad(np.float64(5))))==1.1102230246251565e-16
def test_wrapped_separation():assert wrapped_angular_separation(.01,2*np.pi-.01)==pytest.approx(.02)
def test_s3_curriculum_and_all_pairs():
    assert [(x.stage,x.separation.radial,x.separation.angular_degrees) for x in S3_CURRICULUM]==[(1,.10,40.),(2,.08,25.),(3,.08,20.),(4,.07,15.),(5,.06,10.),(6,.05,8.),(7,.05,6.),(8,.05,5.)]
    for stage in range(1,9):
        p=generate_s3(2,740020+stage,stage)
        assert all(admissible(r,phi,S3_CURRICULUM[stage-1].separation) for r,phi in zip(p.rho,p.phi))
def test_s3_fixture_whole_triple_and_stage_seed():
    p=generate_s3(4,740003,8);assert digest(p)=="6b1f85c2067be580d7437b0f898802af6a919762854348e61401a76e919ea3da" and np.array_equal(p.proposal_ordinal,[1,4,5,6])
    expected=(337762941316658746405563336259276709469,33041314116263920982267743306279215307,230536085149481312042159827949580218612,128771227006991193842988709461005461589,182724507872415681348635900261257145205,220274649742565993368580627610495601039,197586049340494163722503850417131911493,20895637259919448875859253557134305345)
    assert tuple(derived_s3_stage_seed(20262201,stage) for stage in range(1,9))==expected
def test_classifier_balance_no_sort_and_determinism():
    a=generate_classifier(3,740030);b=generate_classifier(3,740030);assert tuple(a)==(1,2,3,4,5)
    for s in range(1,6):assert np.array_equal(a[s].rho,b[s].rho) and a[s].rho.shape==(3,s)
    assert np.any(np.diff(a[5].rho[0])<0)
    assert [a[s].proposal_ordinal.tolist() for s in range(1,6)]==[[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]]
    h=hashlib.sha256()
    for s in range(1,6):
        for x in (a[s].rho,a[s].phi,a[s].amplitude):h.update(np.ascontiguousarray(x).tobytes())
    assert h.hexdigest()=="dda7175697977f8c345867790d09fd2061537aff0633769f0d0150ed5cf79980"
def test_surrogate_law_and_split():
    p=generate_surrogate(5,740040);assert p.definition.source_law==SourceLaw(.01,.99)
    ids=[f"src-{i:08d}" for i in range(10000)];s=split_surrogate_source_ids(ids,20260916)
    assert (len(s.training),len(s.validation),len(s.test))==(7000,1500,1500) and not(set(s.training)&set(s.validation)|set(s.training)&set(s.test)|set(s.validation)&set(s.test))
def test_sealed_firewall_and_immutability():
    with pytest.raises(PermissionError):generate_s1(1,20261816,PopulationRole.SEALED)
    p=generate_s2(1,740050)
    with pytest.raises(ValueError):p.rho[0,0]=.2
    for value in (p.rho,p.phi,p.amplitude,p.proposal_ordinal):
        with pytest.raises(ValueError):value.setflags(write=True)
    changed=p.rho.copy();changed[0,0]=.2;assert p.rho[0,0]!=.2
def test_validation_seed_is_not_derived():
    p=generate_s3(1,20262202,8,PopulationRole.VALIDATION);assert p.definition.rng.seed==20262202

@pytest.mark.parametrize("factory,seed",[(generate_classifier,20261906),(generate_s1,20261816),(generate_s2,20262006)])
@pytest.mark.parametrize("role",[PopulationRole.SEALED,PopulationRole.REFERENCE_FIXTURE,PopulationRole.VALIDATION,PopulationRole.TRAINING])
def test_protected_seed_role_matrix_rejects_before_rng(monkeypatch,factory,seed,role):
    monkeypatch.setattr(np.random,"PCG64DXSM",lambda *_:pytest.fail("RNG initialized before sealed firewall"))
    with pytest.raises(PermissionError):factory(1,seed,role)

@pytest.mark.parametrize("role",[PopulationRole.SEALED,PopulationRole.REFERENCE_FIXTURE,PopulationRole.VALIDATION,PopulationRole.TRAINING])
def test_s3_protected_seed_role_matrix_rejects_before_rng(monkeypatch,role):
    monkeypatch.setattr(np.random,"PCG64DXSM",lambda *_:pytest.fail("RNG initialized before sealed firewall"))
    with pytest.raises(PermissionError):generate_s3(1,20262206,8,role)

def test_protected_seed_cannot_be_mislabeled_even_with_authorization():
    with pytest.raises(PermissionError):generate_s1(1,20261816,PopulationRole.REFERENCE_FIXTURE,allow_sealed=True)

def test_direct_population_definition_cannot_bypass_guard(monkeypatch):
    d=PopulationDefinition("s2_final",PopulationRole.VALIDATION,1,2,SourceLaw(.05,.95),RNGIdentity("NumPy PCG64DXSM",20262006),separation=SeparationConstraint(.05,5.))
    monkeypatch.setattr(np.random,"PCG64DXSM",lambda *_:pytest.fail("RNG initialized before sealed firewall"))
    with pytest.raises(PermissionError):generate_interleaved(d,canonical=True)

@pytest.mark.parametrize("seed,source_count,law,separation",[
    (20261906,1,SourceLaw(.01,.99),None),
    (20261816,1,SourceLaw(.05,.95),None),
    (20262006,2,SourceLaw(.05,.95),SeparationConstraint(.05,5.)),
    (20262206,3,SourceLaw(.05,.95),SeparationConstraint(.05,5.)),
])
@pytest.mark.parametrize("population_id",["copied_identity","unknown_identity"])
@pytest.mark.parametrize("allow_sealed",[False,True])
def test_alternate_identity_with_any_protected_seed_rejects_before_rng(monkeypatch,seed,source_count,law,separation,population_id,allow_sealed):
    d=PopulationDefinition(population_id,PopulationRole.SEALED,1,source_count,law,RNGIdentity("NumPy PCG64DXSM",seed),8 if source_count==3 else None,separation)
    monkeypatch.setattr(np.random,"PCG64DXSM",lambda *_:pytest.fail("RNG initialized before seed-anchored firewall"))
    with pytest.raises(PermissionError):generate_interleaved(d,canonical=source_count>1,allow_sealed=allow_sealed)

@pytest.mark.parametrize("population_id,seed,source_count,law,separation",[
    ("classifier_final",20261906,1,SourceLaw(.01,.99),None),
    ("s1_final",20261816,1,SourceLaw(.05,.95),None),
    ("s2_final",20262006,2,SourceLaw(.05,.95),SeparationConstraint(.05,5.)),
    ("s3_final",20262206,3,SourceLaw(.05,.95),SeparationConstraint(.05,5.)),
])
@pytest.mark.parametrize("role",[PopulationRole.REFERENCE_FIXTURE,PopulationRole.VALIDATION,PopulationRole.TRAINING])
def test_copied_definition_wrong_role_rejects_even_with_authorization(monkeypatch,population_id,seed,source_count,law,separation,role):
    d=PopulationDefinition(population_id,role,1,source_count,law,RNGIdentity("NumPy PCG64DXSM",seed),8 if source_count==3 else None,separation)
    monkeypatch.setattr(np.random,"PCG64DXSM",lambda *_:pytest.fail("RNG initialized before seed-anchored firewall"))
    with pytest.raises(PermissionError):generate_interleaved(d,canonical=source_count>1,allow_sealed=True)

@pytest.mark.parametrize("stage",[None,1,7])
@pytest.mark.parametrize("population_id",["s3_final","copied_s3"])
def test_s3_protected_seed_missing_or_wrong_stage_rejects_before_rng(monkeypatch,stage,population_id):
    d=PopulationDefinition(population_id,PopulationRole.SEALED,1,3,SourceLaw(.05,.95),RNGIdentity("NumPy PCG64DXSM",20262206),stage,SeparationConstraint(.05,5.))
    monkeypatch.setattr(np.random,"PCG64DXSM",lambda *_:pytest.fail("RNG initialized before seed-anchored firewall"))
    with pytest.raises(PermissionError):generate_interleaved(d,canonical=True,allow_sealed=True)

@pytest.mark.parametrize("seed",[20261906,20261816,20262006,20262206])
def test_surrogate_path_rejects_every_protected_seed_before_rng(monkeypatch,seed):
    monkeypatch.setattr(np.random,"PCG64DXSM",lambda *_:pytest.fail("RNG initialized before seed-anchored firewall"))
    with pytest.raises(PermissionError):generate_surrogate(1,seed)

@pytest.mark.parametrize("factory,args",[
    (generate_classifier,(1,20261906,PopulationRole.SEALED)),
    (generate_s1,(1,20261816,PopulationRole.SEALED)),
    (generate_s2,(1,20262006,PopulationRole.SEALED)),
    (generate_s3,(1,20262206,8,PopulationRole.SEALED)),
])
def test_correct_label_but_wrong_final_size_rejects_before_rng(monkeypatch,factory,args):
    monkeypatch.setattr(np.random,"PCG64DXSM",lambda *_:pytest.fail("RNG initialized before seed-anchored firewall"))
    with pytest.raises(PermissionError):factory(*args,allow_sealed=True)

def test_correct_label_but_wrong_scientific_law_rejects_before_rng(monkeypatch):
    d=PopulationDefinition("s1_final",PopulationRole.SEALED,15000,1,SourceLaw(.04,.95),RNGIdentity("NumPy PCG64DXSM",20261816))
    monkeypatch.setattr(np.random,"PCG64DXSM",lambda *_:pytest.fail("RNG initialized before seed-anchored firewall"))
    with pytest.raises(PermissionError):generate_interleaved(d,canonical=False,allow_sealed=True)

def test_empty_population_identity_rejected_before_traversal():
    with pytest.raises(Exception):PopulationDefinition("",PopulationRole.SEALED,1,1,SourceLaw(.05,.95),RNGIdentity("NumPy PCG64DXSM",20261816))

def test_protected_seed_registry_supports_collisions_fail_closed(monkeypatch):
    identities=(ProtectedPopulationIdentity("alpha","alpha_final",PopulationRole.SEALED,99001,("synthetic",1)),ProtectedPopulationIdentity("beta","beta_final",PopulationRole.SEALED,99001,("synthetic",2)))
    monkeypatch.setattr(population_generation,"PROTECTED_SEALED_IDENTITIES",identities)
    population_generation._guard_identity("alpha_final",PopulationRole.SEALED,99001,None,("synthetic",1),True)
    population_generation._guard_identity("beta_final",PopulationRole.SEALED,99001,None,("synthetic",2),True)
    with pytest.raises(PermissionError):population_generation._guard_identity("gamma_final",PopulationRole.SEALED,99001,None,("synthetic",1),True)
    with pytest.raises(PermissionError):population_generation._guard_identity("alpha_final",PopulationRole.VALIDATION,99001,None,("synthetic",1),True)
    with pytest.raises(PermissionError):population_generation._guard_identity("alpha_final",PopulationRole.SEALED,99001,None,("synthetic",2),True)

def test_nonscientific_fixture_seeds_remain_allowed():
    assert generate_classifier(1,740060)[1].definition.role is PopulationRole.REFERENCE_FIXTURE
    assert generate_s1(1,740061).definition.role is PopulationRole.REFERENCE_FIXTURE
    assert generate_s2(1,740062).definition.role is PopulationRole.REFERENCE_FIXTURE
    assert generate_s3(1,740063,8).definition.role is PopulationRole.REFERENCE_FIXTURE

def test_frozen_final_population_identities():
    expected={
        "classifier":[("TRAINING",20261901,70000,14000,None,None,None),("VALIDATION",20261902,15000,3000,None,None,None),("SEALED",20261906,15000,3000,None,None,None)],
        "s1":[("TRAINING",20261811,70000,None,None,None,None),("VALIDATION",20261812,15000,None,None,None,None),("SEALED",20261816,15000,None,None,None,None)],
        "s2":[("TRAINING",20262001,70000,None,None,None,None),("VALIDATION",20262002,15000,None,None,None,None),("SEALED",20262006,15000,None,None,None,None)],
        "s3":[("TRAINING",20262201,560000,None,None,8,70000),("VALIDATION",20262202,15000,None,8,None,None),("SEALED",20262206,15000,None,8,None,None)],
    }
    actual={study:[(x.role.value,x.seed,x.count,x.per_class,x.stage,x.stages,x.count_per_stage) for x in FINAL_POPULATION_IDENTITIES if x.study==study] for study in expected}
    assert actual==expected
    assert (SURROGATE_FINAL.generation_seed,SURROGATE_FINAL.split_seed,SURROGATE_FINAL.total,SURROGATE_FINAL.training,SURROGATE_FINAL.validation,SURROGATE_FINAL.test)==(20260915,20260916,10000,7000,1500,1500)
    with pytest.raises(FrozenInstanceError):SURROGATE_FINAL.total=1

@pytest.mark.parametrize("rho,phi",[
    ([.20,.22,.80],[0.,1.,2.]),([.20,.80,.22],[0.,1.,2.]),([.20,.60,.62],[0.,1.,2.]),
    ([.10,.50,.90],[0.,.01,2.]),([.10,.50,.90],[0.,2.,.01]),([.10,.50,.90],[0.,2.,2.01]),
])
def test_s3_each_pair_can_independently_reject(rho,phi):
    assert admissible(np.asarray(rho),np.asarray(phi),SeparationConstraint(.05,5.)) is False

def test_s3_passes_only_when_every_pair_passes():
    rho=np.asarray([.10,.50,.90]);phi=np.asarray([0.,2.,4.])
    for i,j in ((0,1),(0,2),(1,2)):
        assert abs(rho[i]-rho[j])>=.05
        d=abs(phi[i]-phi[j])%(2*np.pi);assert min(d,2*np.pi-d)>=np.deg2rad(5.)
    assert admissible(rho,phi,SeparationConstraint(.05,5.)) is True
