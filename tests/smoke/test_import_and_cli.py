from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from inverse_em.cli.main import FUTURE_GROUPS, run


def snapshot(root: Path):
    return sorted((p.relative_to(root).as_posix(), p.stat().st_size) for p in root.rglob("*") if p.is_file())


def test_import_has_no_rng_side_effect(tmp_path):
    code = "import random; random.seed(9281); a=random.getstate(); import inverse_em; assert a==random.getstate()"
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_import_has_no_filesystem_writes(tmp_path):
    before = snapshot(tmp_path)
    result = subprocess.run([sys.executable, "-c", "import inverse_em"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert snapshot(tmp_path) == before


def test_import_has_no_numpy_rng_side_effect(tmp_path):
    code = "import numpy as np; np.random.seed(9281); a=np.random.get_state(); import inverse_em; b=np.random.get_state(); assert a[0]==b[0] and (a[1]==b[1]).all() and a[2:]==b[2:]"
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    if result.returncode and "No module named 'numpy'" in result.stderr:
        pytest.skip("NumPy is not installed")
    assert result.returncode == 0, result.stderr


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        run(["--help"])
    assert exc.value.code == 0
    assert "Phase 0" in capsys.readouterr().out


@pytest.mark.parametrize("group", FUTURE_GROUPS)
def test_future_commands_fail_closed(group):
    from inverse_em.errors import PhaseNotImplementedError
    with pytest.raises(PhaseNotImplementedError, match="Not implemented in Phase 0"):
        run([group])


@pytest.mark.parametrize("group", FUTURE_GROUPS)
def test_future_command_subprocess_is_fail_closed_and_write_free(group, tmp_path):
    before = snapshot(tmp_path)
    result = subprocess.run([sys.executable, "-m", "inverse_em.cli.main", group], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Not implemented in Phase 0" in result.stderr
    assert snapshot(tmp_path) == before
