import dataclasses
import hashlib
import importlib
import json
from pathlib import Path

import pytest
from inverse_em.provenance.canonical import canonical_bytes, canonical_sha256
from phase11_accounting import ROOT


CONFIGS = {
    "phase1": "Phase1ScientificConfig", "phase2": "Phase2ScientificConfig",
    "surrogate": "SurrogateScientificConfig", "classifier": "ClassifierScientificConfig",
    "localization": "LocalizationScientificConfig", "s1": "S1Config", "s2": "S2Config",
    "s3": "S3Config", "evaluation": "EvaluationConfig",
}
LEDGER = json.loads(Path(__file__).with_name("contract_ledger.json").read_text())


@pytest.mark.parametrize("module,name", CONFIGS.items())
def test_configuration_is_pinned_not_self_attested(module, name):
    value = getattr(importlib.import_module("inverse_em.config." + module), name)()
    assert value.sha256 == LEDGER["configs"][name]
    changed = dataclasses.asdict(value)
    changed["phase11_unapproved_identity"] = True
    assert canonical_sha256(changed) != LEDGER["configs"][name]


@pytest.mark.parametrize("path", ["tests/phase7/history_scalars.jsonl", "tests/phase8/history_scalars.jsonl",
                                  "src/inverse_em/physics/vendor/physics_tm.py",
                                  "tests/reference/physics_tm_reference_v1.json"])
def test_committed_reference_bytes(path):
    # Checkout uses LF for these files. Do not deserialize or execute the vendor.
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == LEDGER["closed_files"][path]


def test_typed_canonical_and_state_identity_distinctions():
    assert len({canonical_sha256(1), canonical_sha256(1.0), canonical_sha256(True)}) == 3
    assert canonical_bytes(-0.0) == canonical_bytes(0.0)
    assert canonical_bytes({"b": 2, "a": 1}) == b'{"canonicalization":"canonical-json-v1","payload":{"a":1,"b":2}}'
    from inverse_em.artifact_integration.contracts import Domain, Identity
    a = Identity(Domain.SOURCE, "synthetic-source-bytes/1", "a" * 64)
    b = Identity(Domain.NATIVE, "synthetic-source-bytes/1", "a" * 64)
    assert a != b


def test_upstream_dependency_edges():
    from inverse_em.config.s3 import S3Config
    from inverse_em.config.s2 import S2Config
    from inverse_em.config.s1 import S1Config
    from inverse_em.config.localization import LocalizationScientificConfig
    from inverse_em.config.surrogate import HISTORICAL_CHECKPOINT_SHA256
    for value in (S1Config(), S2Config(), S3Config()):
        assert value.phase5_sha256 == LEDGER["configs"]["LocalizationScientificConfig"]
        assert value.electric_sha256 == HISTORICAL_CHECKPOINT_SHA256["E_z_seed20260917"]
        assert value.magnetic_sha256 == HISTORICAL_CHECKPOINT_SHA256["H_phi_seed20260917"]
    assert S3Config().phase1_sha256 == LEDGER["configs"]["Phase1ScientificConfig"]
    assert S3Config().phase2_sha256 == LEDGER["configs"]["Phase2ScientificConfig"]
    assert LocalizationScientificConfig().surrogates.representation == "relative_angle"


def test_integration_config_and_slots_bind_closed_owners():
    from inverse_em.artifact_integration.contracts import Family, configuration, active_slots, tensor_schema
    owners = {"E": "SurrogateScientificConfig", "H": "SurrogateScientificConfig",
              "CLASSIFIER": "ClassifierScientificConfig", "S1": "S1Config", "S2": "S2Config",
              "S3": "S3Config", "NORMALIZER": "Phase2ScientificConfig"}
    for name, owner in owners.items():
        family = getattr(Family, name)
        assert configuration(family).digest == LEDGER["configs"][owner]
    for name, count in (("S1", 1), ("S2", 2), ("S3", 3)):
        family = getattr(Family, name)
        assert active_slots(family) == tuple(range(count))
        assert tensor_schema(family) == tensor_schema(Family.S1)
    assert sum(__import__("math").prod(shape) for _, shape in tensor_schema(Family.S1)) == 399433


@pytest.mark.parametrize("task", ["s1", "s2", "s3"])
def test_checkpoint_boundaries_remain_distinct(task):
    from inverse_em.artifact_integration.contracts import Family, boundary
    module = importlib.import_module("inverse_em.training." + task)
    family = getattr(Family, task.upper())
    assert boundary(family, "CONTINUATION") == module.BOUNDARY
    if task != "s1":
        assert boundary(family, "BEST") != module.BOUNDARY


def test_normalizer_and_noise_descriptors_only():
    from inverse_em.config.phase2 import Phase2ScientificConfig, CHANNEL_ORDER
    from inverse_em.config.s3 import S3Config
    c = Phase2ScientificConfig()
    assert c.channel_order == CHANNEL_ORDER and len(CHANNEL_ORDER) == 4
    assert c.normalization_ddof == 0 and c.validation_is_clean
    assert c.robustness_realizations == 10 and c.paired_across_snr
    assert S3Config().independent_field_snr is True
    assert S3Config().normalizer == "closed_phase2_30_angle_block_moments_all8_clean_train_ddof0_no_epsilon"
