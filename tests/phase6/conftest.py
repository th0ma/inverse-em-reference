import numpy as np
import pytest
import torch
from torch import nn
from inverse_em.config.phase2 import InverseTask
from inverse_em.config.s1 import S1Config, SeedRole
from inverse_em.localization import CanonicalTargets, FieldConsistency
from inverse_em.localization import s1
from inverse_em.normalization import FrozenChannelNormalizer
from inverse_em.observations import ComplexFields
from inverse_em.surrogates.inference import FrozenSurrogate
from inverse_em.training.s1 import S1Fixture, BoundedS1Trainer, initialize_localizer


@pytest.fixture(autouse=True)
def scientific_firewall(monkeypatch):
    import inverse_em.normalization.api as normalization
    import inverse_em.populations.generation as generation
    import inverse_em.physics.api as physics
    def forbidden(*args, **kwargs):
        pytest.fail("Forbidden scientific execution in bounded Phase-6 test")
    original = generation.generate_s1
    def bounded(count, seed, role, **kwargs):
        assert count <= 16 and role.value == "REFERENCE_FIXTURE"
        assert seed not in dict(S1Config().seed_roles).values()
        return original(count, seed, role, **kwargs)
    monkeypatch.setattr(s1, "generate_s1", bounded)
    monkeypatch.setattr(normalization, "fit_clean_training_normalizer", forbidden)
    monkeypatch.setattr(normalization, "_frozen_training_populations", forbidden)
    monkeypatch.setattr(physics.PhysicsTMForward, "evaluate", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


class TinyField(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float64))
    def forward(self, x):
        return torch.stack((self.scale*(x[..., 0]+.2*x[..., 1]),
                            self.scale*(x[..., 2]-.1*x[..., 0])), -1)


def fc_fixture():
    return FieldConsistency(FrozenSurrogate(TinyField(1.)), FrozenSurrogate(TinyField(2.)),
        2*torch.pi*torch.arange(30, dtype=torch.float64)/30)


def data_fixture(role):
    offset = 0 if role is SeedRole.TRAIN_POPULATION else .03
    rho = np.array([.2+offset, .6+offset]); phi = np.array([.3, 1.2])
    theta = 2*np.pi*np.arange(30)/30
    fields = tuple(ComplexFields(np.asarray(r + np.cos(theta-p)*.2 + 1j*np.sin(theta-p), np.complex128),
        np.asarray(2*r + .1*np.cos(theta-p) + .4j*np.sin(theta-p), np.complex128)) for r,p in zip(rho,phi))
    seed = 81001 if role is SeedRole.TRAIN_POPULATION else 81002
    ids = tuple(s1.configuration_id(role, i, r, p, population_seed=seed) for i,(r,p) in enumerate(zip(rho,phi)))
    return S1Fixture(fields, CanonicalTargets(torch.tensor(rho[:,None]),torch.tensor(phi[:,None])), ids, role)


@pytest.fixture
def trainer_factory():
    def build():
        normalizer = FrozenChannelNormalizer(InverseTask.S1, np.array([.1,.2,.3,.4]), np.array([2.,3.,4.,5.]),2,60)
        return BoundedS1Trainer(initialize_localizer(), fc_fixture(), data_fixture(SeedRole.TRAIN_POPULATION),
            data_fixture(SeedRole.VALIDATION_POPULATION), normalizer)
    return build
