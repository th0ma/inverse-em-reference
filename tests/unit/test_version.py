from pathlib import Path
import re
import tomllib

import inverse_em


def test_project_version_has_one_runtime_packaging_source():
    root = Path(__file__).parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["dynamic"] == ["version"]
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "inverse_em._version.__version__"
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    cited = re.search(r"^version:\s*([^\s]+)\s*$", citation, re.MULTILINE)
    assert cited is not None
    assert cited.group(1) == inverse_em.__version__
