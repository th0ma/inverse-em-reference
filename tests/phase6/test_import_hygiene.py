import os
from pathlib import Path
import subprocess
import sys


def test_imports_have_no_execution_rng_or_io_side_effects(tmp_path):
    code = '''import pathlib, socket, torch, numpy as np
import inverse_em.physics.api as physics
import inverse_em.normalization.api as norm
import inverse_em.populations.generation as pop
def forbidden(*a,**k): raise AssertionError("Scientific or network operation on import")
physics.PhysicsTMForward.evaluate=forbidden
norm.fit_clean_training_normalizer=forbidden
pop.generate_s1=forbidden
torch.load=forbidden
socket.socket=forbidden
root=pathlib.Path.cwd(); before=set(root.rglob('*'))
t=torch.get_rng_state().clone(); n=np.random.get_state()
import inverse_em.config.s1, inverse_em.localization.s1, inverse_em.training.s1, inverse_em.provenance.s1
assert torch.equal(t,torch.get_rng_state())
after=np.random.get_state()
assert n[0]==after[0] and np.array_equal(n[1],after[1]) and n[2:]==after[2:]
assert before==set(root.rglob('*'))
'''
    env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]/'src'),PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable,'-B','-c',code],cwd=tmp_path,env=env,check=True,capture_output=True,text=True)
