from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def valid_config_mapping():
    return {
        "schema_version": "scientific-config/1.0",
        "study_id": "synthetic-phase0-test",
        "scientific_role": "REFERENCE_FIXTURE",
        "sections": {"example": {"integer": 1, "float": 1.25}},
    }


@pytest.fixture
def qualified_na():
    return {"status": "NOT_APPLICABLE", "value": None, "reason": "Synthetic Phase-0 fixture"}


@pytest.fixture
def valid_manifest_mapping(qualified_na):
    zero = "0" * 64
    return {
        "schema_version": "scientific-run-manifest/1.0",
        "study_id": "synthetic-phase0-test",
        "scientific_role": "REFERENCE_FIXTURE",
        "state_before": "DEVELOPMENT",
        "state_after": "TRAINING",
        "repository": {"commit": "uncommitted-fixture", "dirty": True, "git_status_sha256": zero},
        "configuration": {"path": "configs/fixture.json", "sha256": zero},
        "physics": dict(qualified_na),
        "populations": dict(qualified_na),
        "normalizer": dict(qualified_na),
        "seeds": dict(qualified_na),
        "model": dict(qualified_na),
        "training": dict(qualified_na),
        "artifacts": dict(qualified_na),
        "environment": dict(qualified_na),
        "timestamps": dict(qualified_na),
    }


@pytest.fixture
def write_json(tmp_path):
    def write(name: str, value) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path
    return write
