"""All artifacts are explicit synthetic JSON; no production resources are read."""
from dataclasses import asdict, replace
import copy
import hashlib
import math
import struct

import pytest

from inverse_em.artifact_integration.contracts import (Context, Domain, Family, Identity, Lineage, Mapping,
    EvidenceRegistry, InsufficientEvidence, Normalizer, Outcome, Transformation, Use, active_slots, architecture, boundary, configuration,
    continuation_keys, current_runtime, identity, scientific_dependency, tensor_schema)
from inverse_em.artifact_integration.integration import (ProtectedResult, ReleaseFailure, SyntheticStore,
    decode, encode, load_production)
from inverse_em.artifact_integration.planning import MigrationPlan, classify, validate_migration
from inverse_em.artifact_integration.validation import (component_identity, plain, synthetic_document,
    reference_controls, tensor, validate, validate_tensors)


EVIDENCE = {}


def token(domain=Domain.NATIVE, name="fixture"):
    return identity(domain, "synthetic-evidence/1", name)


def state_for(family):
    return [tensor(name, shape, (0.0,) * math.prod(shape)) for name, shape in tensor_schema(family)]


def normalizer(task, evidence=None):
    values = (task, (0.0,) * 4, (1.0,) * 4, 8, 240,
              tuple(range(1, 9)) if task == "S3" else (),
              ("Re(E_z)", "Im(E_z)", "Re(H_phi)", "Im(H_phi)"), "float64", 0)
    registry = EvidenceRegistry() if evidence is None else evidence
    training = registry.training_provenance(task, 8, values[5])
    return registry.normalizer(task, values[1], values[2], 8, values[5], training)


def continuation(family, bound):
    params = tensor_schema(family) + ((("kendall", (4,)),) if active_slots(family) else ())
    zeros = [tensor(name, shape, (0.0,) * math.prod(shape)) for name, shape in params]
    parts = {"optimizer": {"parameter_order": [p[0] for p in params],
             "moments": {"exp_avg": zeros, "exp_avg_sq": copy.deepcopy(zeros)},
             "steps": [1] * len(params), "hyperparameters": {"configuration": configuration(family).digest}},
             "scheduler": {"last_epoch": 1, "configuration": configuration(family).digest},
             "rng": {k: "synthetic-owned-state-receipt" for k in ("torch", "numpy", "data_order", "noise")},
             "history": [{"epoch": 1, "updates": 1, "metric": 0.5}],
             "selection": {"epoch": 1, "updates": 1, "metric": 0.5,
                           "criterion": "macro_f1_max" if family is Family.CLASSIFIER else "mse_min" if family in (Family.E, Family.H) else "cartesian_rmse_min"},
             "counters": {"epoch": 1, "updates": 1},
             "boundary_state": {"boundary": bound, "validation_complete": True, "scheduler_complete": True},
             "early_stop": {"configuration": configuration(family).digest, "counter": 0}}
    if active_slots(family):
        parts["kendall"] = [tensor("kendall", (4,), (0.0,) * 4)]
        parts["stage"] = {"index": 1, "epoch": 1, "transition_pending": False}
    best_role = "best_validation" if family is Family.CLASSIFIER else "BEST"
    best_state = state_for(family)
    parts["best_snapshot"] = {"role": best_role, "boundary": boundary(family, best_role),
                              "selection": copy.deepcopy(parts["selection"]), "state": best_state,
                              "state_identity": plain(asdict(validate_tensors(best_state, tensor_schema(family))))}
    if active_slots(family): parts["best_snapshot"]["kendall"] = copy.deepcopy(parts["kendall"])
    if family in (Family.S1, Family.S3):
        parts["early_stop"] = {"enabled": False}
    elif family is Family.S2:
        parts["early_stop"] = {"enabled": True, "patience": 200, "reference": .5, "bad_epochs": 0,
                               "stopped": False, "material_test": "score < es_reference - 1e-5"}
    else:
        parts["early_stop"] = {"reference": .5, "bad_epochs": 0, "patience": 20, "stopped": False}
        if family is Family.CLASSIFIER: parts["early_stop"]["threshold"] = 1e-4
    controls, scheduler = reference_controls(family, parts["history"])
    parts["optimizer"]["hyperparameters"] = controls
    parts["scheduler"] = scheduler
    parts["boundary_state"].update(schema="closed-task-synthetic-continuation/2", fixture_cases=8,
        batches_per_epoch=1, epochs_per_fixture_stage=2 if family is Family.S3 else None)
    import torch
    import numpy as np
    owned = torch.Generator(device="cpu")
    rng_bytes = {"dtype": "uint8", "device": "cpu", "data": bytes(owned.get_state().tolist()).hex()}
    if family in (Family.E, Family.H):
        parts["rng"] = {"torch_rng_state": rng_bytes, "loader_generator_state": rng_bytes}
    elif family is Family.CLASSIFIER:
        from inverse_em.config.classifier import ClassifierScientificConfig
        noise = {"augmentation_seed": ClassifierScientificConfig().training.augmentation_seed,
                 "event_count": 8, "bit_generator": "PCG64DXSM", "bit_generator_state": np.random.PCG64DXSM(17001).state}
        noise["sha256"] = component_identity(noise).digest
        parts["rng"] = {"dropout_rng_state": rng_bytes, "noise_state": noise}
    else:
        parts["rng"] = {"keyed_contract": plain(asdict(scientific_dependency(family, "rng_contract"))),
                        "next_global_epoch": 2, "exposure": 0}
        if family in (Family.S1, Family.S2): parts["rng"]["torch_rng_state"] = rng_bytes
    return parts


def fixture(family=Family.E, use=Use.INFERENCE):
    evidence_registry = EvidenceRegistry()
    task = "S1" if family is Family.NORMALIZER else family.value
    norm = None if family in (Family.E, Family.H) else normalizer(task, evidence_registry)
    state = state_for(family)
    if family is Family.NORMALIZER:
        state = [tensor("mean", (4,), norm.mean), tensor("std", (4,), norm.std)]
    lineages, evidence = [], []
    if active_slots(family):
        for field in (Family.E, Family.H):
            e_state = state_for(field)
            lineage = evidence_registry.lineage(field, e_state)
            source = lineage.source
            lineages.append(lineage)
            evidence.append({"field": field.value, "source": plain(asdict(source)), "state": e_state})
    role = "best_validation" if family is Family.CLASSIFIER else "NORMALIZER" if family is Family.NORMALIZER else "BEST"
    if use is Use.CONTINUATION:
        role = "resumable" if family in (Family.E, Family.H, Family.CLASSIFIER) else "CONTINUATION"
    bound = boundary(family, role)
    parts = continuation(family, bound) if use is Use.CONTINUATION else None
    deps = tuple((k, scientific_dependency(family, k) if k in ("physics", "rng_contract") else token(Domain.NATIVE, k))
                 for k in ("physics", "training_population", "validation_population", "rng_contract", "provenance"))
    if norm is not None:
        training = evidence_registry.training_provenance(task, 8, norm.stages)
        deps = tuple((k, training if k == "training_population" else val) for k, val in deps)
    context = Context(family, use, task, role, configuration(family), architecture(family),
                      validate_tensors(state, tensor_schema(family)), current_runtime(), deps, norm, tuple(lineages), bound,
                      tuple((k, component_identity(parts[k])) for k in continuation_keys(family)) if parts else ())
    EVIDENCE[context.sha256] = evidence_registry
    return context, synthetic_document(context, state, lineage_states=evidence, continuation=parts)


@pytest.mark.parametrize("family", list(Family))
@pytest.mark.parametrize("use", list(Use))
def test_every_family_and_use(tmp_path, family, use):
    if family is Family.NORMALIZER and use is Use.CONTINUATION:
        with pytest.raises(ValueError, match="Normalizer has no continuation"):
            fixture(family, use)
        return
    ctx, doc = fixture(family, use)
    store = SyntheticStore(tmp_path, evidence=EVIDENCE[ctx.sha256])
    source = store.create_source("positive", doc, ctx)
    before = (store.root / "positive.source.json").read_bytes()
    result = store.integrate(source, ctx)
    try:
        assert result.decision.outcome is Outcome.DIRECTLY_COMPATIBLE
        assert (result.accepted is None) == (use is Use.PROVENANCE_ONLY)
        assert before == (store.root / "positive.source.json").read_bytes()
        published = decode((store.root / "positive.accepted.json").read_bytes())
        assert published["receipt"]["decision"]["context_sha256"] == ctx.sha256
        assert published["receipt"]["decision"]["use"] == use.value
        assert published["receipt"]["source_size"] == len(before)
    finally:
        if isinstance(result, ProtectedResult): result.close()
    with pytest.raises(FileExistsError):
        store.integrate(source, ctx)


@pytest.mark.parametrize("case", ["key", "shape", "dtype", "nonfinite", "buffer", "order", "slots", "role", "task",
                                  "config", "provenance", "missing_context", "lineage", "normalizer"])
def test_adversarial_transaction(tmp_path, case):
    ctx, doc = fixture(Family.S1)
    if case == "key": doc["state"][0]["name"] = "wrong"
    if case == "shape": doc["state"][0]["shape"] = [32, 20]  # Same count, wrong shape.
    if case == "dtype": doc["state"][0]["dtype"] = "float32"
    if case == "nonfinite": doc["state"][0]["data"] = struct.pack("<d", float("nan")).hex() + doc["state"][0]["data"][16:]
    if case == "buffer": doc["buffers"] = ["unexpected"]
    if case == "order": doc["state"] = doc["state"][::-1]
    if case == "slots": doc["slots"] = [1]
    if case == "role": doc["context"]["role"] = "TERMINAL"
    if case == "task": doc["context"]["task"] = "S2"
    if case == "config": doc["context"]["configuration"]["digest"] = "0" * 64
    if case == "provenance": doc["context"]["dependencies"][-1][1]["digest"] = "0" * 64
    if case == "missing_context": del doc["context"]["dependencies"]
    if case == "lineage": doc["lineage_states"][0]["state"][0]["data"] = struct.pack("<d", 1.0).hex() + doc["lineage_states"][0]["state"][0]["data"][16:]
    if case == "normalizer": doc["context"]["normalizer"]["std"][0] = 2.0
    store = SyntheticStore(tmp_path)
    source = store.create_source("bad", doc, ctx)
    import numpy as np
    import torch
    numpy_before = copy.deepcopy(np.random.get_state())
    torch_before = torch.get_rng_state().clone()
    # A live destination object is deliberately never passed to integration.
    destination = {"model": torch.ones(2), "optimizer": {"step": 2}, "scheduler": {"lr": .1},
                   "kendall": torch.zeros(4), "mode": "train", "gradients": torch.ones(2)}
    frozen = copy.deepcopy(destination)
    with pytest.raises((ValueError, TypeError, KeyError)):
        store.integrate(source, ctx)
    assert not (store.root / "bad.accepted.json").exists()
    assert not list(store.root.glob(".pending-*"))
    assert torch.equal(torch_before, torch.get_rng_state())
    after = np.random.get_state()
    assert numpy_before[0] == after[0] and np.array_equal(numpy_before[1], after[1]) and numpy_before[2:] == after[2:]
    for key in destination:
        assert torch.equal(destination[key], frozen[key]) if torch.is_tensor(destination[key]) else destination[key] == frozen[key]


@pytest.mark.parametrize("family", [Family.S1, Family.S2, Family.S3, Family.CLASSIFIER, Family.E, Family.H])
def test_missing_continuation_and_bad_boundary(family):
    ctx, doc = fixture(family, Use.CONTINUATION)
    for key in continuation_keys(family):
        broken = copy.deepcopy(doc)
        del broken["continuation"][key]
        with pytest.raises(ValueError): validate(broken, ctx)
    for role in ("BEST", "TERMINAL"):
        with pytest.raises(ValueError): replace(ctx, role=role)
    with pytest.raises(ValueError): replace(ctx, boundary="IN_EPOCH")


def test_context_forgery_and_namespaces(tmp_path):
    ctx, doc = fixture()
    store = SyntheticStore(tmp_path)
    source = store.create_source("ctx", doc, ctx)
    for bad in (None, {}, object()):
        with pytest.raises(ValueError): store.integrate(source, bad)
    with pytest.raises(ValueError): replace(ctx, expected_state=replace(ctx.expected_state, domain=Domain.SOURCE))
    with pytest.raises(ValueError): replace(ctx, dependencies=())
    wrong_physics = (("physics", token(Domain.CONFIG)),) + ctx.dependencies[1:]
    with pytest.raises(ValueError): replace(ctx, dependencies=wrong_physics)
    with pytest.raises(ValueError): replace(ctx, task="S1")
    forged = copy.deepcopy(ctx)
    object.__setattr__(forged, "configuration", token(Domain.CONFIG))
    with pytest.raises(ValueError): store.integrate(source, forged)
    forged = replace(ctx, expected_state=token(Domain.STATE))
    with pytest.raises(ValueError): store.integrate(source, forged)
    assert Identity(Domain.SOURCE, "same", "0" * 64) != Identity(Domain.STATE, "same", "0" * 64)


def test_runtime_insufficient_and_no_production(tmp_path):
    ctx, doc = fixture()
    doc["context"]["runtime"]["torch"] = "unknown-historical"
    store = SyntheticStore(tmp_path)
    result = store.integrate(store.create_source("runtime", doc, ctx), ctx)
    assert result.decision.outcome is Outcome.INSUFFICIENT_EVIDENCE and result.accepted is None
    assert not list(store.root.glob("*.accepted.json"))
    with pytest.raises(PermissionError): load_production("ignored.pt", ctx)
    with pytest.raises(ValueError): store.integrate("external.pt", ctx)


@pytest.mark.parametrize("when", ["before", "during", "prepublish"])
def test_source_mutation(tmp_path, monkeypatch, when):
    import inverse_em.artifact_integration.integration as module
    ctx, doc = fixture()
    store = SyntheticStore(tmp_path)
    source = store.create_source("mutation", doc, ctx)
    path = store.root / "mutation.source.json"
    if when == "before": path.write_bytes(b"changed")
    else:
        original = module.validate
        calls = []
        def mutate(*args):
            value = original(*args)
            calls.append(1)
            if len(calls) == (1 if when == "during" else 2): path.write_bytes(b"changed")
            return value
        monkeypatch.setattr(module, "validate", mutate)
    with pytest.raises(ValueError, match="mutation"):
        store.integrate(source, ctx)
    assert not list(store.root.glob("*.accepted.json"))


def test_classification_rule():
    ctx, _ = fixture()
    source = token(Domain.SOURCE)
    assert classify(ctx, source).outcome is Outcome.INSUFFICIENT_EVIDENCE
    facts = dict(established=True, evidence=(token(),))
    assert classify(ctx, source, **facts).outcome is Outcome.INSUFFICIENT_EVIDENCE
    assert classify(ctx, source, neutral_migration=True, **facts).outcome is Outcome.CONTROLLED_MIGRATION_REQUIRED
    assert classify(ctx, source, normalizer_unrecoverable=True, **facts).outcome is Outcome.REGENERATION_REQUIRED
    assert classify(ctx, source, trained_state_unrecoverable=True, **facts).outcome is Outcome.INSUFFICIENT_EVIDENCE
    assert classify(ctx, source, trained_state_unrecoverable=True, approved_recovery_exhausted=True, **facts).outcome is Outcome.RETRAINING_REQUIRED
    assert classify(ctx, source, contract_violation=True, approved_recovery_exhausted=True, **facts).outcome is Outcome.RETRAINING_REQUIRED
    assert classify(ctx, source, continuation_only_missing=True, approved_recovery_exhausted=True, **facts).outcome is Outcome.INSUFFICIENT_EVIDENCE


@pytest.mark.parametrize("operation", ["container_conversion", "metadata_key_rename", "state_key_remap"])
def test_neutral_migration(operation):
    plan = MigrationPlan(operation, (("a", "b"),), (("a", "same_semantic_tensor"),), (("b", "same_semantic_tensor"),))
    assert validate_migration(plan, {"a": "0000000000000000"}, {"b": "0000000000000000"}) == plan.identity
    with pytest.raises(ValueError): validate_migration(plan, {"a": 1}, {"b": 2})
    with pytest.raises(ValueError): validate_migration(plan, {}, {"b": 1})
    with pytest.raises(ValueError): replace(plan, pairs=(("a", "b"), ("c", "b")))
    with pytest.raises(ValueError): replace(plan, destination_semantics=(("b", "other_slot"),))


@pytest.mark.parametrize("operation", ["dtype_conversion", "normalization", "decoder", "slots", "active_slot", "values",
                                     "surrogate_substitution", "optimizer_reset", "synthesize_continuation"])
def test_scientific_migration_forbidden(operation):
    with pytest.raises(ValueError): MigrationPlan(operation, (), (), ())


@pytest.mark.parametrize("field,value", [("std", (0.0,) * 4), ("mean", (float("inf"),) * 4),
    ("task", "S2"), ("scalar_count", 7), ("dtype", "float32"), ("ddof", 1), ("channels", ("bad",) * 4),
    ("stages", (1,)), ("mapping", None), ("provenance", token(Domain.SOURCE))])
def test_normalizer_rejection(field, value):
    with pytest.raises(ValueError): replace(normalizer("S1"), **{field: value})


def test_no_usable_exposure_and_duplicate_json(tmp_path):
    ctx, doc = fixture()
    store = SyntheticStore(tmp_path)
    source = store.create_source("source", doc, ctx)
    assert not hasattr(source, "state")
    assert not list(store.root.glob("*.accepted.json"))
    with pytest.raises(ValueError): decode(b'{"a":1,"a":2}')
    with pytest.raises(ValueError): decode(b'{"a":NaN}')
    with pytest.raises(ValueError): store.create_source("../source", doc, ctx)
    with pytest.raises(ValueError): store.create_source("production", {"scope": "PRODUCTION"}, ctx)


def test_exact_closed_model_schemas_without_protected_rng():
    import torch
    from inverse_em.surrogates.model import BoundaryFieldMLP
    from inverse_em.classifier.model import SourceCountClassifier
    from inverse_em.localization.model import CircularLocalizer
    before = torch.get_rng_state().clone()
    # Synthetic construction only, never frozen initialization/seed helpers.
    with torch.random.fork_rng():
        for family, constructor in ((Family.E, BoundaryFieldMLP), (Family.CLASSIFIER, SourceCountClassifier),
                                    (Family.S1, CircularLocalizer)):
            model = constructor()
            assert tuple((k, tuple(v.shape)) for k, v in model.state_dict().items()) == tensor_schema(family)
            assert not list(model.named_buffers())
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("component", ["optimizer", "scheduler", "kendall", "history", "best_snapshot", "early_stop", "rng", "stage"])
def test_forged_continuation_receipts_still_check_structure(component):
    ctx, doc = fixture(Family.S2, Use.CONTINUATION)
    parts = doc["continuation"]
    if component == "optimizer": parts[component]["parameter_order"].reverse()
    if component == "scheduler": parts[component]["state"]["last_epoch"] = 9
    if component == "kendall": parts[component][0]["shape"] = [2, 2]
    if component == "history": parts[component][0]["epoch"] = 2
    if component == "best_snapshot": parts[component]["role"] = "TERMINAL"
    if component == "early_stop": parts[component]["patience"] = 199
    if component == "rng": del parts[component]["torch_rng_state"]
    if component == "stage": parts[component]["index"] = 2
    ctx = replace(ctx, continuation_components=tuple((k, component_identity(parts[k])) for k in continuation_keys(ctx.family)))
    doc["context"] = plain(asdict(ctx))
    with pytest.raises(ValueError): validate(doc, ctx)


def test_atomic_publish_failure(tmp_path, monkeypatch):
    import inverse_em.artifact_integration.integration as module
    ctx, doc = fixture()
    store = SyntheticStore(tmp_path)
    source = store.create_source("atomic", doc, ctx)
    original = (store.root / "atomic.source.json").read_bytes()
    def fail(*args): raise OSError("synthetic publication failure")
    monkeypatch.setattr(module.os, "link", fail)
    with pytest.raises(OSError): store.integrate(source, ctx)
    assert not list(store.root.glob("*.accepted.json"))
    assert not list(store.root.glob(".pending-*"))
    assert (store.root / "atomic.source.json").read_bytes() == original


def test_source_and_context_cannot_be_transferred_between_stores(tmp_path):
    ctx, doc = fixture()
    first, second = SyntheticStore(tmp_path), SyntheticStore(tmp_path)
    source = first.create_source("registered", doc, ctx)
    with pytest.raises(ValueError): second.integrate(source, ctx)


def test_use_specific_role_and_boundary_contracts():
    for family in (Family.S1, Family.S2, Family.S3):
        ctx, doc = fixture(family)
        with pytest.raises(ValueError): replace(ctx, role="TERMINAL")
        with pytest.raises(ValueError): replace(ctx, use=Use.CONTINUATION)
        for other in (Family.S1, Family.S2, Family.S3):
            if boundary(family, "BEST") != boundary(other, "BEST"):
                with pytest.raises(ValueError): replace(ctx, boundary=boundary(other, "BEST"))


def test_lineage_metadata_cannot_substitute_other_actual_state():
    ctx, doc = fixture(Family.S3)
    original = doc["lineage_states"][0]["state"][0]
    original["data"] = struct.pack("<d", .125).hex() + original["data"][16:]
    # All historical/source metadata remains byte-for-byte unchanged.
    with pytest.raises(ValueError, match="Actual E/H"):
        validate(doc, ctx)


def test_missing_evidence_is_not_retraining(tmp_path):
    from inverse_em.artifact_integration.integration import RejectedArtifact
    ctx, doc = fixture()
    del doc["state"]
    store = SyntheticStore(tmp_path)
    source = store.create_source("missing", doc, ctx)
    with pytest.raises(RejectedArtifact) as caught:
        store.integrate(source, ctx)
    assert caught.value.decision.outcome is Outcome.INSUFFICIENT_EVIDENCE
    assert caught.value.decision.use is Use.INFERENCE
    assert caught.value.decision.artifact == source.identity


def test_signed_zero_is_not_neutral_value_change():
    plan = MigrationPlan("state_key_remap", (("a", "b"),), (("a", "same"),), (("b", "same"),))
    with pytest.raises(ValueError): validate_migration(plan, {"a": 0.0}, {"b": -0.0})


def test_nested_context_and_plan_immutability():
    ctx, _ = fixture()
    with pytest.raises(ValueError): replace(ctx, dependencies=tuple(list(p) for p in ctx.dependencies))
    ctx, _ = fixture(Family.E, Use.CONTINUATION)
    with pytest.raises(ValueError): replace(ctx, continuation_components=tuple(list(p) for p in ctx.continuation_components))
    with pytest.raises(ValueError): MigrationPlan("metadata_key_rename", (("a", "b"),), (["a", "same"],), (("b", "same"),))


def _rebind_parts(ctx, doc):
    ctx = replace(ctx, continuation_components=tuple((k, component_identity(doc["continuation"][k]))
                  for k in continuation_keys(ctx.family)))
    doc["context"] = plain(asdict(ctx))
    return ctx


def _rejected_without_side_effects(tmp_path, ctx, doc, evidence):
    import numpy as np
    import torch
    numpy_before, torch_before = copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone()
    records_before = evidence._records.copy()
    store = SyntheticStore(tmp_path, evidence=evidence)
    untouched = store.root / "existing.accepted.json"
    untouched.write_bytes(b"existing immutable evidence")
    source = store.create_source("rejection", doc, ctx)
    original = (store.root / "rejection.source.json").read_bytes()
    try:
        result = store.integrate(source, ctx)
    except (ValueError, RuntimeError, TypeError):
        pass
    else:
        try:
            assert result.decision.outcome is Outcome.INSUFFICIENT_EVIDENCE and result.accepted is None
        finally:
            if isinstance(result, ProtectedResult): result.close()
    assert not (store.root / "rejection.accepted.json").exists()
    assert untouched.read_bytes() == b"existing immutable evidence"
    assert original == (store.root / "rejection.source.json").read_bytes()
    assert evidence._records == records_before
    assert torch.equal(torch_before, torch.get_rng_state())
    after = np.random.get_state()
    assert numpy_before[0] == after[0] and np.array_equal(numpy_before[1], after[1]) and numpy_before[2:] == after[2:]


@pytest.mark.parametrize("target", ["source", "stage"])
@pytest.mark.parametrize("mutation", ["different_size", "same_size", "recomputed_metadata"])
def test_blocker1_final_commit_mutations(tmp_path, monkeypatch, target, mutation):
    import numpy as np
    import torch
    import inverse_em.artifact_integration.integration as module
    ctx, doc = fixture()
    book = EVIDENCE[ctx.sha256]
    store = SyntheticStore(tmp_path, evidence=book)
    source = store.create_source("race", doc, ctx)
    immutable = store.root / "previous.accepted.json"
    immutable.write_bytes(b"existing immutable evidence")
    original_source = (store.root / "race.source.json").read_bytes()
    records = book._records.copy()
    caller_np, caller_torch = copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone()
    link = module.os.link
    attempted = []
    def synchronized(staged, destination):
        from pathlib import Path
        target_path = store.root / "race.source.json" if target == "source" else Path(staged)
        raw = target_path.read_bytes()
        if mutation == "different_size": changed = b"CORRUPT"
        elif mutation == "same_size": changed = b"!" + raw[1:]
        else:
            value = decode(raw)
            value["unrelated_recomputed_digest"] = hashlib.sha256(b"mutant").hexdigest()
            changed = encode(value)
        attempted.append(True)
        target_path.write_bytes(changed)  # Must be denied by the held OS handle.
        return link(staged, destination)
    monkeypatch.setattr(module.os, "link", synchronized)
    returned = None
    try:
        with pytest.raises((OSError, ValueError)): returned = store.integrate(source, ctx)
    finally:
        if isinstance(returned, ProtectedResult): returned.close()
    assert attempted and not (store.root / "race.accepted.json").exists()
    assert (store.root / "race.source.json").read_bytes() == original_source
    assert immutable.read_bytes() == b"existing immutable evidence"
    assert book._records == records
    assert torch.equal(caller_torch, torch.get_rng_state())
    after = np.random.get_state()
    assert caller_np[0] == after[0] and np.array_equal(caller_np[1], after[1]) and caller_np[2:] == after[2:]


@pytest.mark.parametrize("family", [Family.E, Family.H, Family.CLASSIFIER, Family.S1, Family.S2, Family.S3])
@pytest.mark.parametrize("case", ["lr999", "weight_decay", "optimizer_missing", "moments", "scheduler_missing",
    "scheduler_epoch", "lr_disagreement", "rng", "early_stop", "best", "updates", "legacy"])
def test_blocker2_correctly_hashed_invalid_continuation(tmp_path, family, case):
    ctx, doc = fixture(family, Use.CONTINUATION)
    book = EVIDENCE[ctx.sha256]
    parts = doc["continuation"]
    groups = parts["optimizer"]["hyperparameters"]["param_groups"]
    if case == "lr999": groups[0]["lr"] = 999.
    if case == "weight_decay": groups[0]["weight_decay"] = 999.
    if case == "optimizer_missing": del groups[0]["betas"]
    if case == "moments": parts["optimizer"]["moments"]["exp_avg"][0]["dtype"] = "float32"
    if case == "scheduler_missing": parts["scheduler"] = {"last_epoch": 1}
    if case == "scheduler_epoch": parts["scheduler"]["state"]["last_epoch"] = 999
    if case == "lr_disagreement": parts["scheduler"]["state"]["_last_lr"] = [999.]
    if case == "rng": parts["rng"] = {"torch": "NOT AN RNG STATE"}
    if case == "early_stop": parts["early_stop"] = {"nonsense": True}
    if case == "best": parts["selection"]["updates"] = 999
    if case == "updates": parts["counters"]["updates"] = 2
    if case == "legacy": del parts["boundary_state"]["schema"]
    ctx = _rebind_parts(ctx, doc)
    _rejected_without_side_effects(tmp_path, ctx, doc, book)


@pytest.mark.parametrize("case", ["impossible_stage", "missing_transition", "wrong_transition", "wrong_stage_clock"])
def test_blocker2_s3_stage_contract(tmp_path, case):
    ctx, doc = fixture(Family.S3, Use.CONTINUATION)
    book = EVIDENCE[ctx.sha256]
    stage = doc["continuation"]["stage"]
    if case == "impossible_stage": stage["index"] = 8
    if case == "missing_transition": del stage["transition_pending"]
    if case == "wrong_transition": stage["transition_pending"] = True
    if case == "wrong_stage_clock": stage["epoch"] = 2
    _rejected_without_side_effects(tmp_path, _rebind_parts(ctx, doc), doc, book)


@pytest.mark.parametrize("family,epoch", [(Family.E, 3), (Family.H, 2), (Family.CLASSIFIER, 2),
    (Family.S1, 4), (Family.S2, 4), (Family.S3, 2), (Family.S3, 3), (Family.S3, 4)])
def test_blocker2_evolved_positive_continuation(tmp_path, family, epoch):
    ctx, doc = fixture(family, Use.CONTINUATION)
    book = EVIDENCE[ctx.sha256]
    parts = doc["continuation"]
    history = [{"epoch": e, "updates": e, "metric": e / 10. if family is Family.CLASSIFIER else 1. / (e + 1)}
               for e in range(1, epoch + 1)]
    parts["history"] = history
    parts["counters"] = {"epoch": epoch, "updates": epoch}
    parts["selection"].update(history[-1])
    parts["best_snapshot"]["selection"] = copy.deepcopy(parts["selection"])
    controls, scheduler = reference_controls(family, history)
    parts["optimizer"]["hyperparameters"], parts["scheduler"] = controls, scheduler
    local = (epoch - 1) % 2 + 1 if family is Family.S3 else epoch
    parts["optimizer"]["steps"] = [local] * len(parts["optimizer"]["steps"])
    if active_slots(family):
        parts["stage"] = {"index": (epoch - 1) // 2 + 1 if family is Family.S3 else 1,
                          "epoch": local, "transition_pending": family is Family.S3 and local == 2}
        parts["rng"]["next_global_epoch"] = epoch + 1
    elif family is Family.CLASSIFIER:
        noise = parts["rng"]["noise_state"]
        noise["event_count"] = epoch * 8
        noise["sha256"] = component_identity({k: val for k, val in noise.items() if k != "sha256"}).digest
    if "reference" in parts["early_stop"]: parts["early_stop"]["reference"] = history[-1]["metric"]
    ctx = _rebind_parts(ctx, doc)
    store = SyntheticStore(tmp_path, evidence=book)
    with store.integrate(store.create_source("evolved", doc, ctx), ctx) as result:
        assert result.decision.outcome is Outcome.DIRECTLY_COMPATIBLE and result.accepted is not None


@pytest.mark.parametrize("case", ["unsupported", "zero", "circular", "unresolved", "wrong_training", "altered_stats", "wrong_task"])
def test_blocker3_provenance(tmp_path, case):
    ctx, doc = fixture(Family.NORMALIZER)
    book = EVIDENCE[ctx.sha256]
    n = ctx.normalizer
    if case in ("unsupported", "zero"):
        source = replace(n.provenance, schema="UNSUPPORTED") if case == "unsupported" else replace(n.provenance, digest="0" * 64)
        n = replace(n, provenance=source, mapping=replace(n.mapping, source=source))
    if case == "circular": n = replace(n, mapping=replace(n.mapping, evidence=n.provenance))
    if case == "unresolved": book = EvidenceRegistry()
    if case == "wrong_training":
        deps = tuple((k, token(name="wrong training") if k == "training_population" else val) for k, val in ctx.dependencies)
        ctx = replace(ctx, dependencies=deps)
    if case == "altered_stats":
        with pytest.raises(ValueError): replace(n, mean=(1.,) * 4)
        return
    if case == "wrong_task":
        with pytest.raises(ValueError): replace(n, task="S2")
        return
    ctx = replace(ctx, normalizer=n)
    doc["context"] = plain(asdict(ctx))
    _rejected_without_side_effects(tmp_path, ctx, doc, book)


@pytest.mark.parametrize("case", ["unsupported_transform", "unsupported_evidence", "missing", "mismatch", "unresolved",
    "wrong_version", "reversal", "semantic_change", "source_schema", "destination_schema"])
def test_blocker4_mapping_resolution(case):
    book = EvidenceRegistry()
    mapping, facts = book.identity_copy(b"tiny independent no-op observation")
    assert book.resolve(mapping, "identity_copy", facts) == mapping.evidence
    before = book._records.copy()
    if case == "unsupported_transform": mapping = replace(mapping, transformation=replace(mapping.transformation, name="unknown"))
    if case == "unsupported_evidence": mapping = replace(mapping, evidence=replace(mapping.evidence, schema="unknown"))
    if case == "missing": mapping = replace(mapping, transformation=None)
    if case == "mismatch": facts = {**facts, "sha256": "0" * 64}
    if case == "unresolved": book = EvidenceRegistry(); before = {}
    if case == "wrong_version": mapping = replace(mapping, transformation=replace(mapping.transformation, version=2))
    if case == "reversal": mapping = replace(mapping, source=mapping.destination, destination=mapping.source)
    if case == "semantic_change": mapping = replace(mapping, transformation=replace(mapping.transformation, semantics=("swap_slots",)))
    if case == "source_schema": mapping = replace(mapping, source=replace(mapping.source, schema="unknown"))
    if case == "destination_schema": mapping = replace(mapping, destination=replace(mapping.destination, schema="unknown"))
    with pytest.raises(InsufficientEvidence): book.resolve(mapping, "identity_copy", facts)
    assert book._records == before


def test_blocker3_arbitrary_training_digest_cannot_mint_provenance():
    book = EvidenceRegistry()
    with pytest.raises(ValueError): book.normalizer("S1", (0.,)*4, (1.,)*4, 8, (), token())
    assert book._records == {}


@pytest.mark.parametrize("family", [Family.S1, Family.S2, Family.S3])
def test_blocker2_best_kendall_complete(tmp_path, family):
    ctx, doc = fixture(family, Use.CONTINUATION)
    book = EVIDENCE[ctx.sha256]
    del doc["continuation"]["best_snapshot"]["kendall"]
    _rejected_without_side_effects(tmp_path, _rebind_parts(ctx, doc), doc, book)


@pytest.mark.parametrize("case", ["source_commit", "stage_commit", "optimizer", "rng", "provenance", "mapping"])
def test_blocker_regressions_detect_in_memory_weakening(tmp_path, case):
    from contextlib import contextmanager
    import inverse_em.artifact_integration.integration as module
    import inverse_em.artifact_integration.validation as validation
    with pytest.MonkeyPatch.context() as patch:
        if case in ("source_commit", "stage_commit"):
            def unprotected(path):
                return open(path, "rb")
            patch.setattr(module, "_commit_read_lock", unprotected)
            def unchecked_cleanup(resources, *args):
                resources.handles[1].close()
                resources.paths[-1].unlink()
            patch.setattr(module, "_protected_handoff", unchecked_cleanup)
            original_require = module.require
            patch.setattr(module, "require", lambda condition, message: None if message == "Committed bytes differ from validated bytes"
                          else original_require(condition, message))
            with pytest.raises(pytest.fail.Exception):
                test_blocker1_final_commit_mutations(tmp_path, patch, "source" if case == "source_commit" else "stage", "same_size")
        elif case == "optimizer":
            original_reference = validation.reference_controls
            def unchecked(family, history):
                controls, schedule = original_reference(family, history)
                controls["param_groups"][0]["lr"] = 999.
                return controls, schedule
            patch.setattr(validation, "reference_controls", unchecked)
            with pytest.raises(AssertionError):
                test_blocker2_correctly_hashed_invalid_continuation(tmp_path, Family.E, "lr999")
        elif case == "rng":
            patch.setattr(validation, "validate_rng", lambda *args: None)
            with pytest.raises(AssertionError):
                test_blocker2_correctly_hashed_invalid_continuation(tmp_path, Family.E, "rng")
        elif case == "provenance":
            patch.setattr(EvidenceRegistry, "resolve", lambda self, mapping, *args: mapping.evidence)
            with pytest.raises(AssertionError): test_blocker3_provenance(tmp_path, "unresolved")
        else:
            patch.setattr(EvidenceRegistry, "resolve", lambda self, mapping, *args: mapping.evidence)
            with pytest.raises(pytest.fail.Exception): test_blocker4_mapping_resolution("missing")


@pytest.fixture
def lease_setup(tmp_path):
    import numpy as np
    import torch
    ctx, doc = fixture()
    book = EVIDENCE[ctx.sha256]
    before_np, before_torch = copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone()
    store = SyntheticStore(tmp_path, evidence=book)
    source = store.create_source("lease", doc, ctx)
    (store.root / "immutable").write_bytes(b"prior evidence")
    yield store, source, ctx, doc, book
    assert (store.root / "immutable").read_bytes() == b"prior evidence"
    assert torch.equal(before_torch, torch.get_rng_state())
    after = np.random.get_state()
    assert before_np[0] == after[0] and np.array_equal(before_np[1], after[1]) and before_np[2:] == after[2:]


def _attempt_lease_mutation(path, operation):
    import os
    raw = path.read_bytes()
    if operation == "write": path.write_bytes(b"changed")
    elif operation == "equal": path.write_bytes(b"!" + raw[1:])
    elif operation == "delete": path.unlink()
    elif operation == "rename": path.rename(path.with_suffix(".renamed"))
    elif operation == "replace":
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(b"replacement")
        try: os.replace(replacement, path)
        finally: replacement.unlink(missing_ok=True)
    else:
        value = decode(raw)
        value["recomputed_digest"] = hashlib.sha256(b"changed").hexdigest()
        path.write_bytes(encode(value))


@pytest.mark.parametrize("target", ["source", "accepted"])
@pytest.mark.parametrize("operation", ["write", "equal", "delete", "rename", "replace", "metadata"])
def test_blocker_lease_lifetime(lease_setup, monkeypatch, target, operation):
    import inverse_em.artifact_integration.integration as module
    store, source, ctx, doc, book = lease_setup
    path = store.root / ("lease." + target + ".json")
    original = module._protected_handoff
    observed = []
    def hook(resources, raw, envelope, destination):
        original(resources, raw, envelope, destination)
        receipt = decode(destination.read_bytes())["receipt"]
        assert receipt["mapping"]["evidence"]["digest"] in book._records
        with pytest.raises(OSError): _attempt_lease_mutation(path, operation)
        observed.append(True)
    monkeypatch.setattr(module, "_protected_handoff", hook)
    lease = store.integrate(source, ctx)
    try:
        assert observed and type(lease) is ProtectedResult and lease.state == "OPEN"
        with pytest.raises(AttributeError): lease.state = "CLOSED"
        with pytest.raises(OSError): _attempt_lease_mutation(path, operation)
        published = decode((store.root / "lease.accepted.json").read_bytes())
        assert published["document"] == decode(lease.accepted.document) == doc
        assert published["receipt"] == decode(lease.accepted.receipt)
        assert published["receipt"]["destination"] == plain(asdict(identity(Domain.NATIVE, "validated-synthetic-document/1", doc)))
        assert hashlib.sha256((store.root / "lease.source.json").read_bytes()).hexdigest() == source.identity.digest
    finally: lease.close()
    assert lease.state == "CLOSED" and all(h.closed for h in lease._resources.handles)
    assert not list(store.root.glob(".pending-*"))
    lease.close()  # Idempotent.
    with pytest.raises(RuntimeError): _ = lease.accepted
    path.write_bytes(b"after explicit completion")  # No post-close immutability claim.


@pytest.mark.parametrize("phase", ["lock", "publication", "verification", "evidence", "handoff"])
def test_blocker_lease_pre_return_exception(lease_setup, monkeypatch, phase):
    import inverse_em.artifact_integration.integration as module
    store, source, ctx, doc, book = lease_setup
    before = book._records.copy()
    def fail(*args, **kwargs): raise OSError("injected " + phase)
    if phase == "lock": monkeypatch.setattr(module, "_commit_read_lock", fail)
    if phase == "publication": monkeypatch.setattr(module.os, "link", fail)
    if phase == "verification": monkeypatch.setattr(module.os.path, "samefile", lambda *args: False)
    if phase == "evidence":
        class PartialUpdate(dict):
            def update(self, other):
                super().update(other)
                raise OSError("evidence publication failed after partial update")
        book._records = PartialUpdate(book._records)
    if phase == "handoff": monkeypatch.setattr(module, "_protected_handoff", fail)
    with pytest.raises((OSError, ValueError)): store.integrate(source, ctx)
    assert book._records == before
    assert not (store.root / "lease.accepted.json").exists()
    assert not list(store.root.glob(".pending-*"))
    (store.root / "lease.source.json").write_bytes(encode(doc))


@pytest.mark.parametrize("mode", ["explicit", "context", "body_exception"])
def test_blocker_lease_completion(lease_setup, mode):
    store, source, ctx, doc, book = lease_setup
    lease = store.integrate(source, ctx)
    if mode == "explicit": lease.close()
    elif mode == "context":
        with lease as result: assert result.accepted is not None
    else:
        with pytest.raises(LookupError):
            with lease: raise LookupError("caller body failed")
    assert lease.state == "CLOSED"
    lease.close()
    assert not list(store.root.glob(".pending-*"))
    (store.root / "lease.source.json").write_bytes(b"released source")
    (store.root / "lease.accepted.json").write_bytes(b"released destination")


@pytest.mark.parametrize("mode", ["handle", "closed_handle_error", "context_handle"])
def test_blocker_lease_release_failure(lease_setup, monkeypatch, mode):
    from pathlib import Path
    store, source, ctx, doc, book = lease_setup
    lease = store.integrate(source, ctx)
    committed = (store.root / "lease.accepted.json").read_bytes()
    records = book._records.copy()
    if mode in ("handle", "context_handle", "closed_handle_error"):
        actual = lease._resources.handles[2]
        class FailOnce:
            failed = False
            @property
            def closed(self): return actual.closed
            def close(self):
                if not self.failed:
                    self.failed = True
                    if mode == "closed_handle_error": actual.close()
                    raise OSError("injected handle close failure")
                actual.close()
        lease._resources.handles[2] = FailOnce()
    try:
        with pytest.raises(ReleaseFailure) as caught:
            if mode == "context_handle":
                with lease: pass
            else: lease.close()
        assert caught.value.committed and caught.value.cleanup is lease._resources
        assert lease.state == "RELEASE_FAILED" and book._records == records
        assert (store.root / "lease.accepted.json").read_bytes() == committed
        with pytest.raises(RuntimeError): _ = lease.accepted
        lease.close()
        assert lease.state == "CLOSED" and not list(store.root.glob(".pending-*"))
    finally: lease.close()


def test_blocker_lease_rejection_cleanup_failure(lease_setup, monkeypatch):
    import inverse_em.artifact_integration.integration as module
    from pathlib import Path
    store, source, ctx, doc, book = lease_setup
    records = book._records.copy()
    def fail(*args): raise ValueError("pre-return failure")
    original = Path.unlink
    def blocked(path, *args, **kwargs): raise OSError("injected cleanup failure")
    with monkeypatch.context() as patch:
        patch.setattr(module, "_protected_handoff", fail)
        patch.setattr(Path, "unlink", blocked)
        with pytest.raises(ReleaseFailure) as caught: store.integrate(source, ctx)
        assert not caught.value.committed and book._records == records
    caught.value.cleanup.close()
    assert not (store.root / "lease.accepted.json").exists()
    assert not list(store.root.glob(".pending-*"))


def test_blocker_lease_nonusable_needs_no_completion(tmp_path):
    ctx, doc = fixture(use=Use.PROVENANCE_ONLY)
    store = SyntheticStore(tmp_path, evidence=EVIDENCE[ctx.sha256])
    source = store.create_source("provenance", doc, ctx)
    result = store.integrate(source, ctx)
    assert not isinstance(result, ProtectedResult) and result.accepted is None
    assert not list(store.root.glob(".pending-*"))
    (store.root / "provenance.source.json").write_bytes(b"released")


@pytest.mark.parametrize("failure", ["second_lock", "destination_lock", "staging_unlink", "staging_close"])
def test_blocker_lease_protected_cleanup_failure(lease_setup, monkeypatch, failure):
    import inverse_em.artifact_integration.integration as module
    from pathlib import Path
    store, source, ctx, doc, book = lease_setup
    before = book._records.copy()
    acquire = module._commit_read_lock
    handles = []
    class FailCloseOnce:
        def __init__(self, stream): self.stream, self.failed = stream, False
        @property
        def closed(self): return self.stream.closed
        def read(self, *args): return self.stream.read(*args)
        def seek(self, *args): return self.stream.seek(*args)
        def close(self):
            if not self.failed:
                self.failed = True
                raise OSError("stage close failed")
            self.stream.close()
    def acquire_hook(path):
        number = len(handles) + 1
        if (failure == "second_lock" and number == 2) or (failure == "destination_lock" and number == 3):
            raise OSError("lock acquisition failed")
        stream = acquire(path)
        if failure == "staging_close" and number == 2: stream = FailCloseOnce(stream)
        handles.append(stream)
        return stream
    monkeypatch.setattr(module, "_commit_read_lock", acquire_hook)
    if failure == "staging_unlink":
        unlink = Path.unlink
        calls = []
        def fail_once(path, *args, **kwargs):
            if path.name.startswith(".pending-") and not calls:
                calls.append(True)
                with pytest.raises(OSError): (store.root / "lease.source.json").write_bytes(b"bad")
                with pytest.raises(OSError): (store.root / "lease.accepted.json").write_bytes(b"bad")
                raise OSError("protected cleanup failed")
            return unlink(path, *args, **kwargs)
        monkeypatch.setattr(Path, "unlink", fail_once)
    with pytest.raises(OSError): store.integrate(source, ctx)
    assert all(h.closed for h in handles) and book._records == before
    assert not (store.root / "lease.accepted.json").exists()
    assert not list(store.root.glob(".pending-*"))


@pytest.mark.parametrize("released", [(0,), (2,), (0, 2)])
def test_blocker_lease_detects_secret_prereturn_release(lease_setup, monkeypatch, released):
    import inverse_em.artifact_integration.integration as module
    original = module._protected_handoff
    resources_seen = []
    def premature(resources, *args):
        original(resources, *args)
        resources_seen.append(resources)
        for index in released: resources.handles[index].close()
    monkeypatch.setattr(module, "_protected_handoff", premature)
    target = "source" if 0 in released else "accepted"
    try:
        with pytest.raises(pytest.fail.Exception):
            test_blocker_lease_lifetime(lease_setup, monkeypatch, target, "equal")
    finally:
        for resources in resources_seen: resources.close()
