from collections import Counter
import numpy as np
import pytest
import torch
from torch import nn
from inverse_em.config.s3 import SeedRole
from inverse_em.config.phase2 import InverseTask
from inverse_em.localization import s3, CanonicalTargets, FieldConsistency
from inverse_em.normalization import FrozenChannelNormalizer
from inverse_em.observations import ComplexFields
from inverse_em.surrogates.inference import FrozenSurrogate
from inverse_em.training.s3 import S3Fixture, BoundedS3Trainer, initialize_localizer


@pytest.fixture(autouse=True)
def firewall(monkeypatch):
    from inverse_em.populations import generation
    import inverse_em.normalization as normalization
    import inverse_em.normalization.api as api
    from inverse_em.physics import PhysicsTMForward
    counts = Counter({k: 0 for k in ('production_training', 'train_population', 'validation_population',
                                   'normalizer', 'checkpoint', 'sealed', 'robustness')})
    def stop(key):
        def reject(*args, **kwargs):
            counts[key] += 1
            raise AssertionError('Protected execution: '+key)
        return reject
    original = generation.generate_interleaved
    def guarded(d, **kwargs):
        if d.count > 16:
            return stop('train_population' if d.count == 70000 else 'validation_population')()
        assert d.role is generation.PopulationRole.REFERENCE_FIXTURE
        assert d.rng.seed not in range(20262201, 20262208)
        assert d.rng.seed not in [s3.stage_population_seed(20262201, i) for i in range(1, 9)]
        return original(d, **kwargs)
    monkeypatch.setattr(generation, 'generate_interleaved', guarded)
    monkeypatch.setattr(normalization, 'fit_clean_training_normalizer', stop('normalizer'))
    monkeypatch.setattr(api, 'fit_clean_training_normalizer', stop('normalizer'))
    monkeypatch.setattr(api, '_frozen_training_populations', stop('normalizer'))
    monkeypatch.setattr(api, '_fit_generated_training_populations', stop('normalizer'))
    monkeypatch.setattr(PhysicsTMForward, 'evaluate', stop('production_training'))
    monkeypatch.setattr(torch, 'load', stop('checkpoint'))
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield counts
    torch.set_num_threads(old)
    assert not any(counts.values()), counts


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


def data_fixture(stage, role):
    offset = stage*.001
    r = np.array([[.10+offset, .45+offset, .85+offset], [.12+offset, .47+offset, .87+offset]])
    p = np.array([[.1, 2., 4.], [.2, 2.1, 4.1]])
    theta = 2*np.pi*np.arange(30)/30
    fields = tuple(ComplexFields(
        np.asarray(sum(rh+.2*np.cos(theta-ph)+1j*np.sin(theta-ph) for rh, ph in zip(rs, ps)), np.complex128),
        np.asarray(sum(2*rh+.1*np.cos(theta-ph)+.4j*np.sin(theta-ph) for rh, ph in zip(rs, ps)), np.complex128))
        for rs, ps in zip(r, p))
    ids = tuple(s3.configuration_id(role, stage, rh, ph) for rh, ph in zip(r, p))
    return S3Fixture(fields, CanonicalTargets(torch.tensor(r), torch.tensor(p)), ids, role, stage)


@pytest.fixture
def trainer_factory():
    def build():
        normalizer = FrozenChannelNormalizer(InverseTask.S3, np.array([.1,.2,.3,.4]),
            np.array([2.,3.,4.,5.]), 4, 120, tuple(range(1,9)))
        return BoundedS3Trainer(initialize_localizer(), fc_fixture(),
            tuple(data_fixture(i, SeedRole.TRAIN_POPULATION) for i in (1,2)),
            data_fixture(8, SeedRole.VALIDATION_POPULATION), normalizer)
    return build
