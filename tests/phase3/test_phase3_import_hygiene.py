import importlib,socket,sys
import numpy as np
import torch
import inverse_em.physics as physics

def test_phase3_import_has_no_side_effects(monkeypatch):
    events={"rng":0,"physics":0,"checkpoint":0,"write":0,"network":0}
    def forbidden(name):
        def call(*a,**k):events[name]+=1;raise AssertionError(name)
        return call
    monkeypatch.setattr(np.random,"default_rng",forbidden("rng"));monkeypatch.setattr(torch,"Generator",forbidden("rng"));monkeypatch.setattr(torch,"load",forbidden("checkpoint"));monkeypatch.setattr(torch,"save",forbidden("write"));monkeypatch.setattr(physics,"PhysicsTMForward",forbidden("physics"));monkeypatch.setattr(socket.socket,"connect",forbidden("network"))
    for name in tuple(sys.modules):
        if name=="inverse_em.surrogates" or name.startswith("inverse_em.surrogates."):del sys.modules[name]
    importlib.import_module("inverse_em.surrogates");assert events=={"rng":0,"physics":0,"checkpoint":0,"write":0,"network":0}
