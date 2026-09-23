import importlib
import socket
import sys

import numpy as np

import inverse_em.physics as physics
import inverse_em.populations as populations


def test_phase2_import_has_no_scientific_or_external_side_effects(monkeypatch):
    events = {"rng": 0, "generation": 0, "physics": 0, "writes": 0, "network": 0}

    def forbidden_rng(*args, **kwargs):
        events["rng"] += 1
        raise AssertionError("RNG constructed during import")

    def forbidden_generation(*args, **kwargs):
        events["generation"] += 1
        raise AssertionError("population generated during import")

    class ForbiddenPhysics:
        def __init__(self, *args, **kwargs):
            events["physics"] += 1
            raise AssertionError("PhysicsTM constructed during import")

    def forbidden_connect(*args, **kwargs):
        events["network"] += 1
        raise AssertionError("network accessed during import")

    def audit(event, args):
        if event == "open" and len(args) > 1 and isinstance(args[1], str):
            if any(flag in args[1] for flag in "wax+"):
                events["writes"] += 1

    monkeypatch.setattr(np.random, "PCG64DXSM", forbidden_rng)
    for name in ("generate_classifier", "generate_s1", "generate_s2", "generate_s3", "generate_surrogate"):
        monkeypatch.setattr(populations, name, forbidden_generation)
    monkeypatch.setattr(physics, "PhysicsTMForward", ForbiddenPhysics)
    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)
    sys.addaudithook(audit)
    for name in tuple(sys.modules):
        if name == "inverse_em.normalization" or name.startswith("inverse_em.normalization."):
            del sys.modules[name]
    importlib.import_module("inverse_em.normalization")
    assert events == {"rng": 0, "generation": 0, "physics": 0, "writes": 0, "network": 0}
