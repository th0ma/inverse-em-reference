import copy
import json
import math
import pytest

from inverse_em.config import ScientificConfig, load_config
from inverse_em.errors import SchemaValidationError


def test_config_is_validated_and_digest_deterministic(valid_config_mapping, write_json):
    path = write_json("config.json", valid_config_mapping)
    first = load_config(path)
    second = load_config(path)
    assert first.sha256 == second.sha256


def test_unknown_config_field_rejected(valid_config_mapping):
    data = copy.deepcopy(valid_config_mapping); data["unexpected"] = True
    with pytest.raises(SchemaValidationError):
        ScientificConfig.from_mapping(data)


def test_unresolved_scientific_value_rejected(valid_config_mapping):
    data = copy.deepcopy(valid_config_mapping); data["sections"]["example"]["value"] = "UNRESOLVED"
    with pytest.raises(SchemaValidationError):
        ScientificConfig.from_mapping(data)


def test_no_silent_string_to_number_coercion(valid_config_mapping):
    data = copy.deepcopy(valid_config_mapping)
    data["acceptance"] = {"class_b_rtol": "1e-12"}
    with pytest.raises(SchemaValidationError):
        ScientificConfig.from_mapping(data)


def test_unknown_scientific_role_rejected(valid_config_mapping):
    data = copy.deepcopy(valid_config_mapping)
    data["scientific_role"] = "TESTISH"
    with pytest.raises(SchemaValidationError):
        ScientificConfig.from_mapping(data)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_values_rejected_during_validation(valid_config_mapping, value):
    data = copy.deepcopy(valid_config_mapping)
    data["sections"] = {"value": value}
    with pytest.raises(SchemaValidationError):
        ScientificConfig.from_mapping(data)


@pytest.mark.parametrize("version", ["scientific-config/2.0", "", 1, None])
def test_unsupported_schema_versions_rejected(valid_config_mapping, version):
    data = copy.deepcopy(valid_config_mapping)
    data["schema_version"] = version
    with pytest.raises(SchemaValidationError):
        ScientificConfig.from_mapping(data)


def test_missing_schema_version_rejected(valid_config_mapping):
    data = copy.deepcopy(valid_config_mapping); del data["schema_version"]
    with pytest.raises(SchemaValidationError):
        ScientificConfig.from_mapping(data)


def test_non_string_nested_key_and_collision_attempt_rejected(valid_config_mapping):
    for sections in ({1: "numeric"}, {1: "numeric", "1": "string"}):
        data = copy.deepcopy(valid_config_mapping)
        data["sections"] = sections
        with pytest.raises(SchemaValidationError):
            ScientificConfig.from_mapping(data)


def test_recursive_immutability(valid_config_mapping):
    config = ScientificConfig.from_mapping(valid_config_mapping)
    with pytest.raises(TypeError):
        config.sections["example"]["integer"] = 2


@pytest.mark.parametrize("path", [r"C:\\machine\\config.json", "/etc/config.json", "../config.json", r"configs\\x.json"])
def test_portable_configuration_paths_reject_machine_specific_forms(valid_config_mapping, path):
    data = copy.deepcopy(valid_config_mapping)
    data["sections"] = {"input_path": path}
    with pytest.raises(SchemaValidationError):
        ScientificConfig.from_mapping(data)


def test_logical_configuration_path_is_accepted(valid_config_mapping):
    data = copy.deepcopy(valid_config_mapping)
    data["sections"] = {"input_path": "configs/example.json"}
    assert ScientificConfig.from_mapping(data).sections["input_path"] == "configs/example.json"


@pytest.mark.parametrize("sections", [
    {"dataset": r"C:\Users\someone\data.npz"},
    {"foo": "D:/research/file.pt"},
    {"source": "/home/user/input.npy"},
    {"x": r"\\server\share\file.dat"},
    {"nested": {"arbitrary": r"C:\data\input.npy"}},
    {"nested": ["label", "/tmp/data"]},
    {"nested": ({"items": ["ok", "//server/share/data.npz"]},)},
])
def test_machine_absolute_paths_rejected_under_arbitrary_nested_keys(valid_config_mapping, sections):
    data = copy.deepcopy(valid_config_mapping)
    data["sections"] = sections
    with pytest.raises(SchemaValidationError, match="machine-local absolute"):
        ScientificConfig.from_mapping(data)


@pytest.mark.parametrize("value", [
    "scientific-config/1.0",
    "canonical-json-v1",
    "surrogate_E_seed20260917",
    "study/s1/final",
    "https://example.org/research/specification",
    "doi:10.1000/example",
    "urn:example:inverse-em:study",
    "ordinary descriptive text",
])
def test_ordinary_scientific_and_uri_strings_remain_valid(valid_config_mapping, value):
    data = copy.deepcopy(valid_config_mapping)
    data["sections"] = {"arbitrary_label": {"values": [value]}}
    assert ScientificConfig.from_mapping(data).sections["arbitrary_label"]["values"][0] == value


def test_invalid_absolute_path_never_obtains_configuration_digest(valid_config_mapping):
    data = copy.deepcopy(valid_config_mapping)
    data["sections"] = {"dataset": r"C:\Users\someone\data.npz"}
    with pytest.raises(SchemaValidationError):
        ScientificConfig.from_mapping(data).sha256
