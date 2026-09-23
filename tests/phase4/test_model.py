import pytest
import hashlib
import torch
from torch import nn

from inverse_em.classifier import (OwnedDropoutRNG, SourceCountClassifier, historical_state_sha256,
                                   make_initialized_classifier, parameter_count)
from inverse_em.config.classifier import DROPOUT_DERIVATION_STRING, DROPOUT_DERIVATION_SHA256, DROPOUT_SEED, HISTORICAL_INITIAL_STATE_SHA256


def test_architecture_parameter_count_and_layout():
    model = SourceCountClassifier().double()
    assert parameter_count(model) == 99_973
    assert sum(isinstance(layer, nn.CircularPad1d) for layer in model.modules()) == 9
    assert sum(isinstance(layer, nn.Dropout) for layer in model.modules()) == 4
    assert all(conv.padding == (0,) and conv.stride == (1,) and conv.dilation == (1,) and conv.groups == 1
               and conv.bias is not None for conv in (layer for layer in model.modules() if isinstance(layer, nn.Conv1d)))


def test_residual_has_second_activation_before_addition_and_no_post_add_activation():
    block = SourceCountClassifier().double().blocks[0].eval(); value = torch.ones(1, 64, 30, dtype=torch.float64)
    transformed = block.act2(block.conv2(block.pad2(block.dropout(block.act1(block.conv1(block.pad1(value)))))))
    assert torch.equal(block(value), value + transformed)


def test_shape_dtype_and_rejections():
    model = make_initialized_classifier().eval()
    assert model(torch.zeros(2, 4, 30, dtype=torch.float64)).shape == (2, 5)
    with pytest.raises(Exception): model(torch.zeros(2, 4, 30))
    with pytest.raises(Exception): model(torch.zeros(2, 120, dtype=torch.float64))
    with pytest.raises(Exception): model(torch.zeros(2, 4, 29, dtype=torch.float64))


def test_cyclic_roll_equivariance_and_logit_invariance():
    model = make_initialized_classifier().eval(); x = torch.arange(240, dtype=torch.float64).reshape(2, 4, 30) / 100
    def features(value):
        value = model.act(model.conv(model.pad(value)))
        return model.blocks(value)
    assert torch.allclose(features(torch.roll(x, 7, -1)), torch.roll(features(x), 7, -1), atol=2e-14, rtol=2e-14)
    assert torch.allclose(model(torch.roll(x, 7, -1)), model(x), atol=2e-14, rtol=2e-14)


def test_initialization_exact_and_preserves_caller_rng():
    torch.manual_seed(77); before = torch.get_rng_state().clone(); first = make_initialized_classifier()
    assert torch.equal(before, torch.get_rng_state())
    second = make_initialized_classifier()
    assert all(torch.equal(first.state_dict()[key], second.state_dict()[key]) for key in first.state_dict())
    assert historical_state_sha256(first) == HISTORICAL_INITIAL_STATE_SHA256


def test_owned_dropout_isolated_stochastic_and_replayable():
    model = make_initialized_classifier().train(); x = torch.ones(2, 4, 30, dtype=torch.float64)
    a = OwnedDropoutRNG(DROPOUT_SEED); b = OwnedDropoutRNG(DROPOUT_SEED)
    torch.manual_seed(91); caller = torch.get_rng_state().clone()
    with a.activate(): first = model(x)
    with a.activate(): second = model(x)
    with b.activate(): replay = model(x)
    assert torch.equal(caller, torch.get_rng_state()) and torch.equal(first, replay) and not torch.equal(first, second)


def test_reference_dropout_seed_derivation():
    digest = hashlib.sha256(DROPOUT_DERIVATION_STRING.encode()).digest()
    assert digest.hex() == DROPOUT_DERIVATION_SHA256
    assert int.from_bytes(digest[:8], "little") == DROPOUT_SEED
