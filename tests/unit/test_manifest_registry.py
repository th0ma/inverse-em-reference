import copy
from dataclasses import FrozenInstanceError
import pytest

from inverse_em.errors import SchemaValidationError
from inverse_em.provenance.manifest import RunManifest, ValueStatus
from inverse_em.provenance.registry import ArtifactRegistry


ZERO = "0" * 64


def test_manifest_validates_and_preserves_not_applicable(valid_manifest_mapping):
    parsed = RunManifest.from_mapping(valid_manifest_mapping)
    assert parsed.physics.status is ValueStatus.NOT_APPLICABLE
    assert parsed.physics.reason


def test_manifest_unknown_field_rejected(valid_manifest_mapping):
    value = copy.deepcopy(valid_manifest_mapping); value["mystery"] = 1
    with pytest.raises(SchemaValidationError):
        RunManifest.from_mapping(value)


@pytest.mark.parametrize("version", ["scientific-run-manifest/2.0", "", 1, None])
def test_manifest_schema_version_rejected(valid_manifest_mapping, version):
    value = copy.deepcopy(valid_manifest_mapping); value["schema_version"] = version
    with pytest.raises(SchemaValidationError):
        RunManifest.from_mapping(value)


def test_manifest_critical_provenance_is_typed(valid_manifest_mapping):
    value = copy.deepcopy(valid_manifest_mapping); value["repository"] = {"x": 1}
    with pytest.raises(SchemaValidationError):
        RunManifest.from_mapping(value)


def test_manifest_is_immutable(valid_manifest_mapping):
    parsed = RunManifest.from_mapping(valid_manifest_mapping)
    with pytest.raises(FrozenInstanceError):
        parsed.study_id = "changed"
    with pytest.raises(FrozenInstanceError):
        parsed.repository.commit = "changed"


@pytest.mark.parametrize("status", list(ValueStatus))
def test_all_qualified_states_round_trip(valid_manifest_mapping, status):
    value = copy.deepcopy(valid_manifest_mapping)
    if status is ValueStatus.PRESENT:
        value["physics"] = {"status": status.value, "value": {"revision": "fixture", "sha256": ZERO}, "reason": None}
    else:
        value["physics"] = {"status": status.value, "value": None, "reason": "Explicit synthetic reason"}
    parsed = RunManifest.from_mapping(value)
    reparsed = RunManifest.from_mapping({**value, "physics": parsed.physics.to_mapping()})
    assert reparsed.physics.status is status


def test_present_typed_manifest_provenance(valid_manifest_mapping):
    value = copy.deepcopy(valid_manifest_mapping)
    value["populations"] = {"status": "PRESENT", "reason": None, "value": [{
        "population_id": "fixture", "scientific_role": "VALIDATION", "sha256": ZERO,
        "sample_count": 2, "seed": 7,
    }]}
    value["seeds"] = {"status": "PRESENT", "reason": None, "value": [{"role": "training", "value": 7}]}
    value["model"] = {"status": "PRESENT", "reason": None, "value": {
        "class_name": "SyntheticFixture", "parameter_count": 3, "dtype": "float64", "device": "cpu",
    }}
    value["training"] = {"status": "PRESENT", "reason": None, "value": {
        "epochs_completed": 1, "updates_completed": 2, "checkpoint_selection_criterion": "synthetic fixture",
        "best": {"status": "PRESENT", "reason": None, "value": {"epoch": 1, "update": 2, "sha256": ZERO}},
        "terminal": {"status": "PRESENT", "reason": None, "value": {"epoch": 1, "update": 2, "sha256": ZERO}},
    }}
    value["artifacts"] = {"status": "PRESENT", "reason": None, "value": [{"logical_id": "fixture", "sha256": ZERO}]}
    value["environment"] = {"status": "PRESENT", "reason": None, "value": {
        "python_version": "3.12", "platform": "synthetic", "packages": [{"name": "fixture", "version": "1"}],
    }}
    value["timestamps"] = {"status": "PRESENT", "reason": None, "value": {
        "started_at": "2026-01-01T00:00:00+00:00", "completed_at": "2026-01-01T00:01:00+00:00",
    }}
    parsed = RunManifest.from_mapping(value)
    assert parsed.training.value.best.value.epoch == 1
    assert parsed.training.value.terminal.value.update == 2


@pytest.mark.parametrize("document", ["manifest", "registry"])
def test_required_schema_version_cannot_be_missing(valid_manifest_mapping, document):
    if document == "manifest":
        value = copy.deepcopy(valid_manifest_mapping); del value["schema_version"]
        with pytest.raises(SchemaValidationError): RunManifest.from_mapping(value)
    else:
        with pytest.raises(SchemaValidationError): ArtifactRegistry.from_mapping({"artifacts": []})


def test_registry_validates():
    registry = ArtifactRegistry.from_mapping({
        "schema_version": "artifact-registry/1.0",
        "artifacts": [{
            "logical_id": "fixture.one", "scientific_role": "REFERENCE_FIXTURE",
            "study_id": "phase0-test", "sha256": ZERO, "byte_size": 3,
            "format": "json", "storage_class": "GIT", "evidence_class": "REFERENCE",
            "provenance_manifest_sha256": ZERO,
        }],
    })
    assert registry.artifacts[0].location is None


def test_duplicate_artifact_ids_rejected():
    item = {
        "logical_id": "same", "scientific_role": "REFERENCE_FIXTURE", "study_id": "x",
        "sha256": ZERO, "byte_size": 0, "format": "json", "storage_class": "GIT",
        "evidence_class": "REFERENCE", "provenance_manifest_sha256": ZERO,
    }
    with pytest.raises(SchemaValidationError):
        ArtifactRegistry.from_mapping({"schema_version": "artifact-registry/1.0", "artifacts": [item, item]})


def test_registry_allows_consistent_digest_aliases():
    item = {
        "logical_id": "a", "scientific_role": "REFERENCE_FIXTURE", "study_id": "x",
        "sha256": ZERO, "byte_size": 3, "format": "json", "schema_version": "fixture/1.0",
        "storage_class": "GIT", "evidence_class": "REFERENCE", "provenance_manifest_sha256": ZERO,
    }
    registry = ArtifactRegistry.from_mapping({"schema_version": "artifact-registry/1.0", "artifacts": [item, {**item, "logical_id": "alias"}]})
    assert len(registry.artifacts) == 2


@pytest.mark.parametrize("change", [{"byte_size": 4}, {"format": "yaml"}, {"schema_version": "fixture/2.0"}])
def test_registry_rejects_conflicting_digest_metadata(change):
    item = {
        "logical_id": "a", "scientific_role": "REFERENCE_FIXTURE", "study_id": "x",
        "sha256": ZERO, "byte_size": 3, "format": "json", "schema_version": "fixture/1.0",
        "storage_class": "GIT", "evidence_class": "REFERENCE", "provenance_manifest_sha256": ZERO,
    }
    with pytest.raises(SchemaValidationError, match="Conflicting"):
        ArtifactRegistry.from_mapping({"schema_version": "artifact-registry/1.0", "artifacts": [item, {**item, **change, "logical_id": "b"}]})


@pytest.mark.parametrize("version", ["artifact-registry/2.0", "", 1, None])
def test_registry_schema_version_rejected(version):
    with pytest.raises(SchemaValidationError):
        ArtifactRegistry.from_mapping({"schema_version": version, "artifacts": []})
