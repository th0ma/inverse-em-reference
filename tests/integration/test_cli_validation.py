import json
import pytest

from inverse_em.cli.main import run
from inverse_em.errors import SchemaValidationError


def test_config_validate_and_digest(valid_config_mapping, write_json, capsys):
    path = write_json("config.json", valid_config_mapping)
    assert run(["config", "validate", str(path)]) == 0
    assert capsys.readouterr().out.strip() == "VALID"
    assert run(["config", "digest", str(path)]) == 0
    digest = capsys.readouterr().out.strip()
    assert len(digest) == 64


def test_state_validation_command(capsys):
    assert run(["state", "validate", "FROZEN", "SEALED_CLEAN"]) == 0
    assert capsys.readouterr().out.strip() == "VALID"


def test_manifest_and_registry_cli_validation(valid_manifest_mapping, write_json, capsys):
    manifest = write_json("manifest.json", valid_manifest_mapping)
    registry = write_json("registry.json", {"schema_version": "artifact-registry/1.0", "artifacts": []})
    assert run(["manifest", "validate", str(manifest)]) == 0
    assert capsys.readouterr().out.strip() == "VALID"
    assert run(["registry", "validate", str(registry)]) == 0
    assert capsys.readouterr().out.strip() == "VALID"


def test_malformed_manifest_and_registry_cli_input(write_json):
    manifest = write_json("manifest.json", {"schema_version": "bad"})
    registry = write_json("registry.json", {"schema_version": 1, "artifacts": []})
    with pytest.raises(SchemaValidationError):
        run(["manifest", "validate", str(manifest)])
    with pytest.raises(SchemaValidationError):
        run(["registry", "validate", str(registry)])
