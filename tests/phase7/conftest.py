from collections import Counter
import numpy as np
import pytest
import torch
from torch import nn
from inverse_em.config.phase2 import InverseTask
from inverse_em.config.s2 import S2Config, SeedRole
from inverse_em.localization import CanonicalTargets, FieldConsistency
from inverse_em.localization import s2
from inverse_em.normalization import FrozenChannelNormalizer
from inverse_em.observations import ComplexFields
from inverse_em.surrogates.inference import FrozenSurrogate
from inverse_em.training.s2 import S2Fixture, BoundedS2Trainer, initialize_localizer


@pytest.fixture(autouse=True)
def firewall(monkeypatch):
    import inverse_em.normalization as normalization
    import inverse_em.normalization.api as norm_api
    import inverse_em.populations.generation as generation
    import inverse_em.physics.api as physics
    counters = Counter({key: 0 for key in ("production_training", "train_population",
        "validation_population", "normalizer", "checkpoint", "sealed", "robustness")})
    def forbidden(key):
        def call(*args, **kwargs):
            counters[key] += 1
            raise AssertionError("Protected operation: " + key)
        return call
    original = generation.generate_interleaved
    def bounded(definition, **kwargs):
        count, seed, role = definition.count, definition.rng.seed, definition.role
        if count > 16:
            return forbidden("train_population" if count == 70000 else "validation_population")()
        if seed in (20262006, 20262007):
            return forbidden("sealed" if seed == 20262006 else "robustness")()
        assert seed not in dict(S2Config().seed_roles).values()
        assert role.value == "REFERENCE_FIXTURE"
        return original(definition, **kwargs)
    monkeypatch.setattr(generation, "generate_interleaved", bounded)
    monkeypatch.setattr(normalization, "fit_clean_training_normalizer", forbidden("normalizer"))
    monkeypatch.setattr(norm_api, "fit_clean_training_normalizer", forbidden("normalizer"))
    monkeypatch.setattr(norm_api, "_frozen_training_populations", forbidden("normalizer"))
    monkeypatch.setattr(physics.PhysicsTMForward, "evaluate", forbidden("production_training"))
    monkeypatch.setattr(torch, "load", forbidden("checkpoint"))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield counters
    torch.set_num_threads(threads)
    assert not any(counters.values()), counters


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
    rho = np.array([[.2+offset, .6+offset], [.3+offset, .7+offset]])
    phi = np.array([[.3, 1.2], [.5, 2.]])
    theta = 2*np.pi*np.arange(30)/30
    fields = tuple(ComplexFields(
        np.asarray(sum(r + np.cos(theta-p)*.2 + 1j*np.sin(theta-p) for r,p in zip(rs,ps)), np.complex128),
        np.asarray(sum(2*r + .1*np.cos(theta-p) + .4j*np.sin(theta-p) for r,p in zip(rs,ps)), np.complex128))
        for rs,ps in zip(rho,phi))
    seed = 81001 if role is SeedRole.TRAIN_POPULATION else 81002
    ids = tuple(s2.configuration_id(role, i, r, p, population_seed=seed)
                for i,(r,p) in enumerate(zip(rho,phi)))
    return S2Fixture(fields, CanonicalTargets(torch.tensor(rho),torch.tensor(phi)), ids, role)


@pytest.fixture
def trainer_factory():
    def build():
        normalizer = FrozenChannelNormalizer(InverseTask.S2, np.array([.1,.2,.3,.4]),
                                             np.array([2.,3.,4.,5.]),2,60)
        fc = fc_fixture()
        return BoundedS2Trainer(initialize_localizer(), fc, data_fixture(SeedRole.TRAIN_POPULATION),
                               data_fixture(SeedRole.VALIDATION_POPULATION), normalizer)
    return build
