from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from inverse_em.config.s3 import S3Config, SeedRole, budget, execution_seed
from inverse_em.errors import SchemaValidationError
from inverse_em.localization import s3
from inverse_em.training.s3 import initialize_localizer, tensor_identity
from inverse_em.provenance.s3 import replay_history


def test_config_budget_and_frozen_values():
    c = S3Config()
    assert tuple(dict(c.seed_roles).values()) == tuple(range(20262201,20262208))
    assert c.curriculum == ((1,.1,40.),(2,.08,25.),(3,.08,20.),(4,.07,15.),(5,.06,10.),(6,.05,8.),(7,.05,6.),(8,.05,5.))
    assert budget() == dict(full_batches=546, tail=112, updates_per_epoch=547,
        updates_per_stage=27350, epochs=400, total_updates=218800, training_cases=560000)
    with pytest.raises(SchemaValidationError):
        replace(c, early_stopping=True)
    assert len(c.sha256) == 64 and c.sha256 == S3Config().sha256


def test_exact_stage_seed_identities():
    expected = [337762941316658746405563336259276709469,33041314116263920982267743306279215307,
        230536085149481312042159827949580218612,128771227006991193842988709461005461589,
        182724507872415681348635900261257145205,220274649742565993368580627610495601039,
        197586049340494163722503850417131911493,20895637259919448875859253557134305345]
    assert [s3.stage_population_seed(20262201,i) for i in range(1,9)] == expected


@pytest.mark.parametrize('role', [SeedRole.TRAIN_POPULATION, SeedRole.VALIDATION_POPULATION,
    SeedRole.SEALED_RESERVED, SeedRole.ROBUSTNESS_RESERVED, 'MODEL_INITIALIZATION'])
def test_execution_role_firewall(role):
    with pytest.raises(PermissionError):
        execution_seed(role, SeedRole.MODEL_INITIALIZATION)


def test_initialization_dtype_and_rng_isolation():
    state = torch.get_rng_state().clone()
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        a = initialize_localizer()
        assert torch.get_default_dtype() == torch.float64
        torch.set_default_dtype(torch.float32)
        b = initialize_localizer()
        assert tensor_identity(a) == tensor_identity(b)
        assert torch.equal(state, torch.get_rng_state())
        assert sum(p.numel() for p in a.parameters()) == 399433
        assert all(p.dtype == torch.float64 for p in a.parameters())
    finally:
        torch.set_default_dtype(dtype)


def test_exact_key_order_and_separations():
    role = SeedRole.TRAINING_AUGMENTATION
    key = b's3_analytical_noise_final_v1|20262205|2|51|0|id|E_NOISE'
    seed = int.from_bytes(hashlib.sha256(key).digest()[:16], 'little')
    assert s3.keyed_seed(role,2,51,0,'id','E_NOISE') == seed
    variants = [(2,51,0,'E_NOISE'),(1,51,0,'E_NOISE'),(2,52,0,'E_NOISE'),
                (2,51,1,'E_NOISE'),(2,51,0,'H_NOISE')]
    assert len({s3.keyed_seed(role,s,e,x,'id',stream) for s,e,x,stream in variants}) == 5
    expected = np.random.Generator(np.random.PCG64DXSM(s3.keyed_seed(SeedRole.MINIBATCH_ORDER,2,3,0,'ALL','ORDER'))).permutation(16)
    assert np.array_equal(np.concatenate(s3.epoch_minibatches(16,2,3)), expected)
    with pytest.raises(PermissionError):
        s3.epoch_minibatches(70000,1,1)


def test_historical_scalar_replay_is_pure(monkeypatch):
    def stop(*a, **k):
        raise AssertionError('Scientific construction during scalar replay')
    monkeypatch.setattr(torch.nn.Linear, '__init__', stop)
    monkeypatch.setattr(torch.optim.Adam, '__init__', stop)
    monkeypatch.setattr(np.random, 'Generator', stop)
    rows = [json.loads(x) for x in Path(__file__).with_name('history_scalars.jsonl').read_text().splitlines()]
    out = replay_history(rows)
    assert len(rows) == 400 and len(out['events']) == 78
    assert out['best'] == (394,8,44,215518,.01900846562333382)
    assert out['terminal'] == (400,218800,.020064019380502007)
    with pytest.raises(ValueError):
        replay_history(rows+[rows[-1]])
