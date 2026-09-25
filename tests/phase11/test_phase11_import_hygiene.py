import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("target", ["phase11_accounting", "test_phase11_identities", "test_phase11_reference_replay",
    "test_phase11_numerics", "test_phase11_integration", "test_phase11_contract_firewall", "conftest"])
def test_direct_import_without_scientific_activity(target, tmp_path):
    script = r'''
import builtins, copy, importlib, pathlib, socket, sys
import numpy as np
import torch
import pytest
import scipy.special
before_profile = sys.getprofile()
before_tools = tuple(sys.monitoring.get_tool(i) for i in range(6))
before_np = copy.deepcopy(np.random.get_state())
before_torch = torch.get_rng_state().clone()
def forbidden(*a, **k): raise AssertionError("import activity")
for name in ("RandomState", "default_rng", "Generator", "PCG64", "PCG64DXSM", "SeedSequence", "seed"):
    setattr(np.random, name, forbidden)
torch.load = forbidden
torch.manual_seed = forbidden
torch.nn.Module.__init__ = forbidden
socket.socket = forbidden
original = builtins.open
def guarded(path, mode="r", *a, **k):
    if any(x in mode for x in "wax+"): forbidden()
    return original(path, mode, *a, **k)
builtins.open = guarded
pathlib.Path.write_bytes = forbidden
pathlib.Path.write_text = forbidden
pathlib.Path.mkdir = forbidden
importlib.import_module(sys.argv[1])
after = np.random.get_state()
assert before_np[0] == after[0] and np.array_equal(before_np[1], after[1]) and before_np[2:] == after[2:]
assert torch.equal(before_torch, torch.get_rng_state())
assert sys.getprofile() is before_profile
assert tuple(sys.monitoring.get_tool(i) for i in range(6)) == before_tools
'''
    result = subprocess.run([sys.executable, "-B", "-c", script, target], cwd=tmp_path,
                            capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not list(Path(tmp_path).iterdir())
