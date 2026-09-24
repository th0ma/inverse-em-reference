import os
import subprocess
import sys


def test_phase8_import_has_no_scientific_side_effects():
    code = '''
import builtins, socket
import numpy as np
import torch
from inverse_em.populations import generation
from inverse_em.normalization import api
from inverse_em.physics import PhysicsTMForward
def stop(*a, **k): raise AssertionError("Import side effect")
state=torch.get_rng_state().clone()
nstate=np.random.get_state()
generation.generate_interleaved=stop
api.fit_clean_training_normalizer=stop
PhysicsTMForward.evaluate=stop
torch.load=stop
torch.nn.Linear.__init__=stop
torch.optim.Adam.__init__=stop
np.random.Generator=stop
socket.socket=stop
original=builtins.open
def readonly(file, mode="r", *args, **kw):
    if any(c in mode for c in "wax+"): stop()
    return original(file, mode, *args, **kw)
builtins.open=readonly
import inverse_em.config.s3
import inverse_em.localization.s3
import inverse_em.provenance.s3
import inverse_em.training.s3
assert torch.equal(state,torch.get_rng_state())
after=np.random.get_state()
assert nstate[0]==after[0] and np.array_equal(nstate[1],after[1]) and nstate[2:]==after[2:]
'''
    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    out=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,env=env)
    assert out.returncode==0,out.stdout+out.stderr
