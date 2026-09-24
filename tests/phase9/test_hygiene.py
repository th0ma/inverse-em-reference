"""Fresh-process import probe; guards installed after framework import."""
import os
import subprocess
import sys


def test_fresh_import_hygiene():
    code = r'''
import builtins, pathlib, socket, numpy as np, torch
import inverse_em.noise.api as noise
import inverse_em.populations.generation as populations
import inverse_em.normalization.api as normalization
import inverse_em.physics as physics
before_np=np.random.get_state()
before_torch=torch.get_rng_state().clone()
def stop(*args, **kwargs): raise AssertionError("import scientific side effect")
np.random.SeedSequence=stop
np.random.PCG64DXSM=stop
torch.load=stop
torch.nn.Linear.__init__=stop
normalization.fit_clean_training_normalizer=stop
physics.PhysicsTMForward.__init__=stop
for name in dir(populations):
    if name.startswith("generate"): setattr(populations,name,stop)
builtins.open=stop
pathlib.Path.open=stop
pathlib.Path.mkdir=stop
socket.socket=stop
import inverse_em.config.evaluation
import inverse_em.evaluation.contracts
import inverse_em.evaluation.artifacts
import inverse_em.evaluation.reporting
import inverse_em.provenance.evaluation
import inverse_em.evaluation.execution
after=np.random.get_state()
assert after[0]==before_np[0] and np.array_equal(after[1],before_np[1]) and after[2:]==before_np[2:]
assert torch.equal(before_torch,torch.get_rng_state())
print("IMPORT HYGIENE PASS")
'''
    result=subprocess.run([sys.executable,"-B","-c",code],capture_output=True,text=True,env=os.environ.copy())
    assert result.returncode==0, result.stdout+result.stderr
    assert "IMPORT HYGIENE PASS" in result.stdout
