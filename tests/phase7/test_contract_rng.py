import hashlib
import numpy as np
import pytest
import torch
from inverse_em.config.s2 import S2Config, SeedRole, execution_seed, budget
from inverse_em.errors import SchemaValidationError
from inverse_em.localization import make_localizer
from inverse_em.localization import s2
from inverse_em.provenance.canonical import canonical_sha256
from inverse_em.training.s2 import initialize_localizer


def test_frozen_config():
    c = S2Config()
    assert c.sha256 == canonical_sha256(S2Config())
    assert c.sha256 == "11893683e0eba9070808ae16e10074eb5033c2c7767aeca4bae523713161fbef"
    assert c.task == "S2" and c.active_slots == c.source_count == 2
    assert (c.train_count, c.validation_count, c.batch_size, c.validation_batch_size) == (70000,15000,128,256)
    assert (c.epochs, c.patience, c.material_test) == (600,200,"score < es_reference - 1e-5")
    assert c.loss_coefficients == (1.,2.,.1,.03)
    assert c.snr_interval == (30.,40.)
    assert tuple(dict(c.seed_roles).values()) == tuple(range(20262001,20262008))
    assert budget() == dict(full_batches=546,tail=112,updates_per_epoch=547,epochs=600,total_updates=328200)
    with pytest.raises(SchemaValidationError):
        S2Config(patience=199)


@pytest.mark.parametrize("role", [SeedRole.TRAIN_POPULATION, SeedRole.VALIDATION_POPULATION,
                                  SeedRole.SEALED_RESERVED, SeedRole.ROBUSTNESS_RESERVED,20262006,20262007])
def test_non_executable_roles(role):
    with pytest.raises(PermissionError):
        execution_seed(role, role)


def test_role_mismatch_and_epoch_bounds():
    with pytest.raises(SchemaValidationError):
        s2.keyed_seed(SeedRole.MINIBATCH_ORDER,1,"id","E")
    for epoch in (0,601,True):
        with pytest.raises(SchemaValidationError):
            s2.noise_event("id",epoch)
    with pytest.raises(SchemaValidationError):
        s2.keyed_seed(SeedRole.MINIBATCH_ORDER,1,"not-ALL","ORDER")


def test_exact_ids_keys_draws():
    identity = s2.configuration_id(SeedRole.TRAIN_POPULATION,0,[.2,.6],[.3,1.2],population_seed=81001)
    assert identity == "S2ANF-train-1857680c843fe2f72cc93ee38d451bdd"
    assert s2.configuration_id(SeedRole.VALIDATION_POPULATION,0,[.2,.6],[.3,1.2],population_seed=81001).endswith(identity.split("-")[-1])
    assert s2.configuration_id(SeedRole.TRAIN_POPULATION,1,[.2,.6],[.3,1.2],population_seed=81001) != identity
    assert s2.keyed_seed(SeedRole.TRAINING_AUGMENTATION,1,identity,"SNR") == 307731895960071355467774266570732807834
    gamma, z = s2.noise_event(identity,1)
    assert gamma == 34.40193028422125
    electric = np.stack((z.electric_real,z.electric_imag),axis=1)
    magnetic = np.stack((z.magnetic_real,z.magnetic_imag),axis=1)
    assert hashlib.sha256(electric.astype("<f8").tobytes()).hexdigest() == "a58a95efeaf34a069cff4fba3e8c981bcedab424b0c602e97addb58e70a991f4"
    assert hashlib.sha256(magnetic.astype("<f8").tobytes()).hexdigest() == "d9c7474a9ed3a3609340277b184a81db9264917026cd3c14cb3de8f0c8afa02c"
    assert not np.array_equal(electric,magnetic)
    with pytest.raises(SchemaValidationError):
        s2.configuration_id(SeedRole.TRAIN_POPULATION,0,[.6,.2],[1.2,.3])


def test_order_fixture_and_bound():
    assert s2.keyed_seed(SeedRole.MINIBATCH_ORDER,1,"ALL","ORDER") == 27042484487002488894058941777581428333
    assert s2.epoch_minibatches(9,1)[0].tolist() == [5,0,7,1,8,4,2,3,6]
    assert sum(map(len,s2.epoch_minibatches(16,2))) == 16
    with pytest.raises(SchemaValidationError):
        s2.epoch_minibatches(70000,1)


def test_initialization_isolation_and_draw_semantics():
    before = torch.get_rng_state().clone()
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        m = initialize_localizer()
        assert torch.get_default_dtype() is torch.float64
        assert torch.equal(before,torch.get_rng_state())
        with torch.random.fork_rng(devices=[]):
            expected_external = torch.rand(5)
        assert torch.equal(expected_external,torch.rand(5))
        with torch.random.fork_rng(devices=[]):
            torch.set_default_dtype(torch.float32)
            torch.manual_seed(20262003)
            expected = make_localizer().double()
        assert all(p.dtype is torch.float64 for p in m.parameters())
        assert all(torch.equal(v,expected.state_dict()[k]) for k,v in m.state_dict().items())
        assert all(torch.equal(v,initialize_localizer().state_dict()[k]) for k,v in m.state_dict().items())
    finally:
        torch.set_default_dtype(dtype)
        torch.set_rng_state(before)
