"""Narrower-than-owner Phase-11 execution: no RNG, fitting, training or physics.

Only Phase-11 tests receive these sentinels. Prior tests are never patched.
Aliases already imported into inverse_em modules are replaced by identity too.
"""
import importlib
import inspect
import sys

import pytest
from phase11_accounting import ACCOUNTING, DirectFirewall


MODULES = (
    "populations.generation", "surrogates.data", "surrogates.checkpoint",
    "surrogates.training", "classifier.checkpoint", "classifier.training",
    "classifier.model", "surrogates.model", "normalization.api", "physics.api", "noise.api",
    "localization.s1", "localization.s2", "localization.s3",
    "training.s1", "training.s2", "training.s3", "evaluation.execution",
)


def dangerous(module, name):
    return (name.startswith(("generate", "fit_", "_fit_", "load_", "save_", "initialize_", "make_initialized", "draw_"))
            or name in ("_frozen_training_populations", "keyed_rng", "noise_event", "epoch_minibatches", "tiny_population", "analytical_fixture", "train_epoch", "train_model"))


def _rng_targets(namespace):
    return tuple(value for name, value in vars(namespace).items()
                 if callable(value) and not name.startswith("_")
                 and name not in ("get_state", "test"))


@pytest.fixture(autouse=True)
def phase11_firewall(monkeypatch):
    import numpy as np
    import torch
    modules = [importlib.import_module("inverse_em." + name) for name in MODULES]

    def denied(*args, **kwargs):
        ACCOUNTING.rejected()
        raise PermissionError("Phase-11 cannot execute grandfathered/production mechanisms")

    targets = []
    for module in modules:
        for name, value in vars(module).items():
            if inspect.isfunction(value) and dangerous(module.__name__, name):
                targets.append(value)
    # Capture native identities too: module patching alone misses local aliases.
    targets.extend(_rng_targets(np.random))
    targets.extend(getattr(torch, name) for name in
                   ("load", "manual_seed", "seed", "set_rng_state", "rand", "randn", "randperm", "Generator"))
    targets.append(torch.random.default_generator.manual_seed)
    for module in tuple(sys.modules.values()):
        if module is None or not getattr(module, "__name__", "").startswith("inverse_em"):
            continue
        for name, value in tuple(vars(module).items()):
            if any(value is target for target in targets):
                monkeypatch.setattr(module, name, denied)
    from inverse_em.physics.api import PhysicsTMForward
    targets.append(PhysicsTMForward.evaluate)
    monkeypatch.setattr(PhysicsTMForward, "evaluate", denied)
    for module in modules:
        for value in tuple(vars(module).values()):
            if inspect.isclass(value) and value.__module__ == module.__name__:
                if "Trainer" in value.__name__ or value.__name__ in ("TrainingNoiseStream", "FixtureEngine", "OwnedDropoutRNG"):
                    targets.append(value.__init__)
                    monkeypatch.setattr(value, "__init__", denied)
    for name in ("RandomState", "default_rng", "Generator", "PCG64", "PCG64DXSM", "SeedSequence", "seed", "random", "uniform", "normal", "standard_normal", "permutation"):
        monkeypatch.setattr(np.random, name, denied)
    for name in ("load", "manual_seed", "seed", "rand", "randn", "randperm"):
        monkeypatch.setattr(torch, name, denied)
    targets.append(torch.nn.Module.__init__)
    monkeypatch.setattr(torch.nn.Module, "__init__", denied)
    with DirectFirewall(ACCOUNTING, targets):
        yield
