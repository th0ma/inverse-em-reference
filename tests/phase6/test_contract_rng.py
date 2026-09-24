from dataclasses import replace
import hashlib
import numpy as np
import pytest
import torch
from inverse_em.config.s1 import S1Config, SeedRole, budget, execution_seed
from inverse_em.localization import s1, parameter_count
from inverse_em.training.s1 import initialize_localizer
from inverse_em.provenance.s1 import HISTORICAL_IDENTITIES, HISTORICAL_VALIDATION


def historical_rng(master, epoch, identity, stream):
    text = f"s1_analytical_noise_final_v1|{master}|{epoch}|{identity}|{stream}"
    seed = int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:16], 'little')
    return np.random.Generator(np.random.PCG64DXSM(seed))


def test_exact_config_and_budget():
    c = S1Config()
    assert (c.train_count,c.validation_count,c.source_count,c.active_slots,c.angles,c.batch_size,c.epochs)==(70000,15000,1,1,30,128,200)
    assert dict(c.seed_roles)==dict(zip([r.value for r in SeedRole],range(20261811,20261818)))
    assert (c.lr,c.betas,c.eps,c.weight_decay,c.T_0,c.T_mult,c.eta_min)==(.0005,(.9,.999),1e-8,1e-5,200,2,1e-6)
    assert c.loss_coefficients==(1.,2.,.1,.03) and c.early_stopping is False
    assert c.phase5_sha256=='8303dc42fce7f495bd882b07573948861bd7a35025a141b2224be6e21dd6c363'
    assert c.electric_sha256=='d3a274f0ab4d514cdbb6b5edb32f3976b9926bb82e838e0277ba0f69f5ffa80b'
    assert c.magnetic_sha256=='0285679d940bb20ecbea334e0815018bf7c8aa6b7210d5e2ce3a2a57a053173c'
    assert budget()==dict(full_batches=546,tail=112,updates_per_epoch=547,epochs=200,total_updates=109400)
    assert len(c.sha256)==64 and c.sha256==S1Config().sha256


@pytest.mark.parametrize('field,value',[('epochs',201),('lr',.001),('active_slots',2),('train_count',70000.),('early_stopping',True)])
def test_config_rejects_drift(field,value):
    with pytest.raises(ValueError): replace(S1Config(),**{field:value})


@pytest.mark.parametrize('role',list(SeedRole))
def test_seed_role_firewall(role):
    if role is SeedRole.TRAINING_AUGMENTATION:
        assert execution_seed(role,SeedRole.TRAINING_AUGMENTATION)==20261815
    else:
        with pytest.raises((PermissionError,ValueError)): execution_seed(role,SeedRole.TRAINING_AUGMENTATION)


def test_exact_configuration_id_encoding():
    triple=np.array([.2,.3,1.],dtype='<f8').tobytes()
    expected='S1FINAL-train-'+hashlib.sha256(b's1_analytical_noise_final_v1|20261811|train|0'+triple).hexdigest()[:32]
    actual=s1.configuration_id(SeedRole.TRAIN_POPULATION,0,.2,.3)
    assert actual==expected
    assert actual!=s1.configuration_id(SeedRole.VALIDATION_POPULATION,0,.2,.3)
    with pytest.raises(PermissionError): s1.configuration_id(SeedRole.SEALED_RESERVED,0,.2,.3)


def test_keyed_draws_match_independent_historical_oracle():
    identity='S1FINAL-train-fixture'; gamma,draw=s1.noise_event(identity,1)
    assert gamma==historical_rng(20261815,1,identity,'SNR').uniform(30,40)
    E=historical_rng(20261815,1,identity,'E').standard_normal((30,2))
    H=historical_rng(20261815,1,identity,'H').standard_normal((30,2))
    assert np.array_equal(draw.electric_real,E[:,0]) and np.array_equal(draw.electric_imag,E[:,1])
    assert np.array_equal(draw.magnetic_real,H[:,0]) and np.array_equal(draw.magnetic_imag,H[:,1])
    assert not np.array_equal(E,H)
    assert gamma!=s1.noise_event(identity,2)[0] and gamma!=s1.noise_event(identity+'x',1)[0]
    assert 30<=gamma<40


def test_frozen_keyed_noise_digests():
    identity='S1FINAL-train-fixture'
    assert [s1.keyed_seed(SeedRole.TRAINING_AUGMENTATION,1,identity,s) for s in ('SNR','E','H')]==[
        333194308650409525507816565739888156949,20981602655818402777535045569653631797,
        88741488032269837998513815869114988623]
    gamma,d=s1.noise_event(identity,1)
    arrays=[np.asarray([gamma],dtype='<f8'),np.stack((d.electric_real,d.electric_imag),axis=1).astype('<f8'),
        np.stack((d.magnetic_real,d.magnetic_imag),axis=1).astype('<f8')]
    assert [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]==[
        '6201b342fab11fa40f80c510ff65474df3d726b28dc1ee3173d30aed9642bbaa',
        '324215521d54f591c76448c08ec444319cd56e8ac0ba181cac3e57ca18e5d467',
        '70bc19b061c32a0fbfe24476b9e9d3a3e660c2761203cdf83420463a023cd489']


@pytest.mark.parametrize('epoch',[0,-1,201,True])
def test_noise_rejects_invalid_epoch(epoch):
    with pytest.raises(ValueError): s1.noise_event('fixture',epoch)


def test_historical_order_digest_and_retained_tail():
    batches=s1.epoch_minibatches(70000,1)
    assert len(batches)==547 and all(len(b)==128 for b in batches[:-1]) and len(batches[-1])==112
    order=np.concatenate(batches)
    assert hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()=='c9c2702ad5e84eec9838c3765d81da41a8ce1d29c50bd3fc57574b736945a84f'
    assert np.array_equal(order,historical_rng(20261814,1,'ALL','ORDER').permutation(70000))
    assert not np.array_equal(order,np.concatenate(s1.epoch_minibatches(70000,2)))
    assert sorted(np.concatenate(s1.epoch_minibatches(7,1)))==list(range(7))


def test_initialization_rng_dtype_and_parameter_identity():
    before=torch.get_rng_state().clone(); dtype=torch.get_default_dtype()
    model=initialize_localizer(); other=initialize_localizer()
    assert torch.equal(before,torch.get_rng_state()) and torch.get_default_dtype()==dtype
    assert parameter_count(model)==399433
    assert all(p.dtype is torch.float64 and p.device.type=='cpu' for p in model.parameters())
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in model.state_dict().items())
    with torch.random.fork_rng(devices=[]):
        torch.set_default_dtype(torch.float64)
        try:
            third=initialize_localizer()
            assert torch.get_default_dtype() is torch.float64
            assert all(torch.equal(v,third.state_dict()[k]) for k,v in model.state_dict().items())
        finally: torch.set_default_dtype(dtype)


def test_reference_evidence_not_acceptance_threshold():
    assert HISTORICAL_VALIDATION['epoch']==200 and HISTORICAL_VALIDATION['updates']==109400
    assert HISTORICAL_VALIDATION['cartesian_rmse']==.00022168686968818813
    assert HISTORICAL_IDENTITIES['checkpoint_file']=='d5a961fb83811cd09fa3b9b9993f1adbd10266ea10f83365993172ea7578f42d'
