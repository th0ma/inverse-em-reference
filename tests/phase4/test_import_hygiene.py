import os
from pathlib import Path
import subprocess
import sys


def test_import_is_side_effect_free(tmp_path):
    code = """import pathlib,torch,numpy as np
root=pathlib.Path.cwd(); before={str(p.relative_to(root)) for p in root.rglob('*')}; ts=torch.get_rng_state().clone(); ns=np.random.get_state()
import inverse_em.classifier
assert torch.equal(ts,torch.get_rng_state()); after={str(p.relative_to(root)) for p in root.rglob('*')}; assert before==after
assert ns[0]==np.random.get_state()[0] and np.array_equal(ns[1],np.random.get_state()[1])
"""
    environment = dict(os.environ); environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path, env=environment, check=True)
