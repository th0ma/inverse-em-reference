import numpy as np
import pytest
import torch

from inverse_em.localization.decoding import RawLocalizerOutput, decode_scientific
from inverse_em.localization.losses import (CanonicalTargets, LocalizationComponents,
    reduce_s2_canonical, reduce_outer_s3)
from inverse_em.localization.metrics import canonical_metrics


def t(values):
    return torch.tensor(values, dtype=torch.float64)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_active_slot_decoder_no_projection(count):
    raw = RawLocalizerOutput(t([[0., 1., -1.]]), t([[2., 3., 4.]]), t([[0., 1., -1.]]))
    p = decode_scientific(raw, count)
    assert p.rho.shape == (1, count)
    assert torch.equal(p.rho, (.05 + .90 * torch.sigmoid(raw.raw_radius))[:, :count])
    assert torch.equal(p.cos_like, raw.cos_like[:, :count])
    assert torch.equal(p.phi, torch.atan2(raw.sin_like, raw.cos_like)[:, :count])


def test_s2_sum_not_source_mean():
    c = LocalizationComponents(t([[1., 3.]]), t([[2., 4.]]), t([[0., 0.]]))
    assert reduce_s2_canonical(c).item() == 16.
    assert reduce_s2_canonical(c).item() != c.uniform.mean().item()


def test_s3_circle_is_not_outer_weighted():
    c = LocalizationComponents(t([[1., 2., 3.]]), t([[2., 3., 4.]]), t([[9., 2., 1.]]))
    target = CanonicalTargets(t([[.1, .4, .9]]), t([[0., 1., 2.]]))
    w = (1 + target.rho) / 1.635
    expected = (w * c.rho).mean() + 2 * (w * c.phi).mean() + .1 * c.circle.mean()
    assert torch.equal(reduce_outer_s3(c, target), expected)
    assert not torch.isclose(expected, (w * c.uniform).mean(), rtol=1e-12, atol=1e-14)


def test_pooled_metric_not_mean_of_batch_rmse():
    from inverse_em.localization.decoding import ScientificPrediction
    prediction = ScientificPrediction(t([[.2], [.5]]), t([[1.], [1.]]), t([[0.], [0.]]), t([[0.], [0.]]))
    target = CanonicalTargets(t([[.1], [.1]]), t([[0.], [0.]]))
    result = canonical_metrics(prediction, target)
    errors = np.array([.1, .4])
    assert result["source_count"] == 2
    np.testing.assert_allclose(result["cartesian"]["rmse"], np.sqrt((errors ** 2).mean()), rtol=1e-12, atol=1e-14)
    assert result["cartesian"]["rmse"] != result["cartesian"]["mean"]


def test_normalizer_is_explicit_synthetic_statistics_only():
    from inverse_em.config.phase2 import InverseTask
    from inverse_em.normalization.api import FrozenChannelNormalizer
    import inspect
    assert "task" in inspect.signature(FrozenChannelNormalizer).parameters
    # Exact policy is bound independently by the configuration ledger; no fitter.
    from inverse_em.config.phase2 import Phase2ScientificConfig
    assert Phase2ScientificConfig().normalization_ddof == 0
    value = FrozenChannelNormalizer(InverseTask.S1, np.array([1., 2., 3., 4.]),
                                    np.array([2., 4., 8., 16.]), 2, 60)
    raw = np.arange(120, dtype=np.float64).reshape(4, 30)
    expected = (raw - value.mean[:, None]) / value.std[:, None]
    assert np.array_equal(value.transform(raw), expected)
    from dataclasses import replace
    for change in ({"ddof": 1}, {"scalar_count_per_channel": 59}, {"curriculum_stages": (1,)}):
        with pytest.raises(ValueError):
            replace(value, **change)
