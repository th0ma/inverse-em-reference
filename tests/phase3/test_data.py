import numpy as np
import pytest
from inverse_em.config.surrogate import HISTORICAL_ARRAY_SHA256,HISTORICAL_SPLIT_SHA256
from inverse_em.errors import SchemaValidationError
from inverse_em.surrogates.data import LazyPointwiseFieldDataset,array_sha256,generate_analytical_targets,generate_historical_sources,historical_split

def test_historical_coordinate_and_split_hashes():
    sources=generate_historical_sources();split=historical_split(sources.source_id)
    assert {"rho_s":array_sha256(sources.rho),"phi_s":array_sha256(sources.phi),"source_id":array_sha256(sources.source_id),"theta":array_sha256(sources.theta)}==dict(HISTORICAL_ARRAY_SHA256)
    assert {"training":array_sha256(np.asarray(split.training,dtype="U12")),"validation":array_sha256(np.asarray(split.validation,dtype="U12")),"test":array_sha256(np.asarray(split.test,dtype="U12"))}==dict(HISTORICAL_SPLIT_SHA256)
    assert not(set(split.training)&set(split.validation) or set(split.training)&set(split.test) or set(split.validation)&set(split.test))

def test_endpoint_excluded_grid_and_source_domain():
    x=generate_historical_sources();assert len(x.source_id)==10000 and x.theta.shape==(72,) and x.theta[0]==0 and x.theta[-1]<2*np.pi
    assert np.array_equal(x.theta,2*np.pi*np.arange(72,dtype=np.float64)/72) and np.all((x.rho>=.01)&(x.rho<.99)) and np.all((x.phi>=0)&(x.phi<2*np.pi))

def test_lazy_pointwise_mapping_raw_targets_and_firewall():
    sources=generate_historical_sources();split=historical_split(sources.source_id);lookup={value:i for i,value in enumerate(sources.source_id.tolist())};ids=np.asarray(split.training[:2],dtype="U12");indices=np.asarray([lookup[x] for x in ids]);target=np.arange(288,dtype=np.float64).reshape(2,72,2)
    data=LazyPointwiseFieldDataset(ids,sources.rho[indices],sources.phi[indices],sources.theta,target,partition="TRAINING",split=split,sources=sources)
    assert len(data)==144 and tuple(float(x) for x in data[73][:3])==(sources.rho[indices[1]],sources.phi[indices[1]],sources.theta[1]) and np.array_equal(data[73][3].numpy(),target[1,1]) and np.shares_memory(data.targets,target)
    with pytest.raises(PermissionError):LazyPointwiseFieldDataset(ids,sources.rho[indices],sources.phi[indices],sources.theta,target,partition="HELD_OUT_TEST",split=split,sources=sources)

def test_partition_identity_rejects_heldout_cross_partition_unknown_and_misalignment():
    sources=generate_historical_sources();split=historical_split(sources.source_id);lookup={value:i for i,value in enumerate(sources.source_id.tolist())};target=np.zeros((1,72,2),np.float64)
    def view(source_id,partition="TRAINING",rho_offset=0.):
        index=lookup.get(source_id,0)
        return LazyPointwiseFieldDataset([source_id],[sources.rho[index]+rho_offset],[sources.phi[index]],sources.theta,target,partition=partition,split=split,sources=sources)
    assert view(split.training[0],"TRAINING").partition=="TRAINING"
    assert view(split.validation[0],"VALIDATION").partition=="VALIDATION"
    for source_id,partition in ((split.test[0],"TRAINING"),(split.test[0],"VALIDATION"),(split.validation[0],"TRAINING"),(split.training[0],"VALIDATION"),("src-unknown", "TRAINING")):
        with pytest.raises(PermissionError): view(source_id,partition)
    with pytest.raises(SchemaValidationError): view(split.training[0],rho_offset=1e-12)
    with pytest.raises(SchemaValidationError): LazyPointwiseFieldDataset([split.training[0]],[],[],sources.theta,target,partition="TRAINING",split=split,sources=sources)
    index=lookup[split.training[0]]
    with pytest.raises(SchemaValidationError): LazyPointwiseFieldDataset([split.training[0],split.training[0]],[sources.rho[index]]*2,[sources.phi[index]]*2,sources.theta,np.zeros((2,72,2)),partition="TRAINING",split=split,sources=sources)

def test_small_physics_fixture_raw_component_semantics():
    sources=generate_historical_sources();e,h=generate_analytical_targets(sources,[0])
    assert e.shape==h.shape==(1,72,2) and e.dtype==h.dtype==np.float64 and np.isfinite(e).all() and np.isfinite(h).all()
    assert np.any(e[...,0]!=0) and np.any(e[...,1]!=0) and np.any(h[...,0]!=0) and np.any(h[...,1]!=0)
