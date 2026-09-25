"""Distinct module name avoids collisions with unchanged prior-phase tests."""
import os
from pathlib import Path
import subprocess
import sys


def test_fresh_import_has_no_activity(tmp_path):
    script = r'''
import numpy as np
import torch
import socket
import builtins
import pathlib
import sys
import copy
before_np = copy.deepcopy(np.random.get_state())
before_torch = torch.get_rng_state().clone()
def forbidden(*args, **kwargs):
    raise AssertionError("Import attempted protected activity")
torch.load = forbidden
torch.nn.Module.__init__ = forbidden
socket.socket = forbidden
original_open = builtins.open
def guarded(file, mode="r", *args, **kwargs):
    if any(flag in mode for flag in "wax+"):
        forbidden()
    return original_open(file, mode, *args, **kwargs)
builtins.open = guarded
pathlib.Path.write_bytes = forbidden
pathlib.Path.write_text = forbidden
import inverse_em.artifact_integration
import inverse_em.artifact_integration.contracts
import inverse_em.artifact_integration.validation
import inverse_em.artifact_integration.planning
import inverse_em.artifact_integration.integration
assert torch.equal(before_torch, torch.get_rng_state())
after = np.random.get_state()
assert before_np[0] == after[0] and np.array_equal(before_np[1], after[1]) and before_np[2:] == after[2:]
assert not any(name.startswith(("inverse_em.training", "inverse_em.data", "inverse_em.normalization", "inverse_em.evaluation")) for name in sys.modules)
'''
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert list(Path(tmp_path).iterdir()) == []
