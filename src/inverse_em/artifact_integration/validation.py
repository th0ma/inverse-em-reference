"""Pure validation of a bounded JSON interchange, not a production checkpoint format."""
from dataclasses import asdict
import json
import math
import struct

from .contracts import (Context, Domain, Family, Identity, Outcome, Use, active_slots,
                        EvidenceRegistry, InsufficientEvidence, boundary, continuation_keys,
                        configuration, current_runtime, identity, require, tensor_schema)


def plain(value):
    """JSON projection; no pickle or executable deserialization."""
    return json.loads(json.dumps(value, allow_nan=False))


def tensor(name, shape, values):
    """Encode explicit synthetic values without RNG, model construction or fitting."""
    require(type(shape) is tuple and len(values) == math.prod(shape), "Tensor size")
    return {"name": name, "shape": list(shape), "dtype": "float64", "device": "cpu",
            "data": struct.pack("<" + "d" * len(values), *values).hex()}


def validate_tensors(state, schema):
    require(type(state) is list and len(state) == len(schema), "Exact tensor keys required")
    for item, (name, shape) in zip(state, schema):
        require(type(item) is dict and set(item) == {"name", "shape", "dtype", "device", "data"}, "Tensor record schema")
        require(item["name"] == name, "Tensor key/parameter ordering")
        require(type(item["shape"]) is list and all(type(v) is int for v in item["shape"]) and
                item["shape"] == list(shape), "Tensor shape")
        require(item["dtype"] == "float64" and item["device"] == "cpu", "Tensor dtype/device")
        data = item["data"]
        require(type(data) is str and len(data) == math.prod(shape) * 16, "Tensor bytes size")
        raw = bytes.fromhex(data)
        require(raw.hex() == data, "Noncanonical tensor bytes")
        require(all(math.isfinite(v[0]) for v in struct.iter_unpack("<d", raw)), "Nonfinite tensor")
    return identity(Domain.STATE, "ordered-f64le-tensors/1", state)


def component_identity(value):
    return identity(Domain.STATE, "synthetic-continuation-component/1", value)


def reference_controls(family, history):
    """Detached optimizer/scheduler control replay only; no steps or RNG draws.

    Settings come from closed configs. Scalar parameters stand in for the exact
    ordered parameter IDs; incoming moment shapes are independently checked.
    """
    import torch
    if str(torch.__version__).split("+")[0] != "2.8.0":
        raise InsufficientEvidence("Closed continuation controls require the established PyTorch 2.8.0 runtime")
    from inverse_em.config.surrogate import SurrogateTrainingConfig
    from inverse_em.config.classifier import ClassifierScientificConfig
    from inverse_em.config.s1 import S1Config
    from inverse_em.config.s2 import S2Config
    from inverse_em.config.s3 import S3Config
    local = family in (Family.S1, Family.S2, Family.S3)
    count = len(tensor_schema(family)) + int(local)
    parameters = [torch.nn.Parameter(torch.zeros((), dtype=torch.float64)) for _ in range(count)]
    if local:
        cfg = {Family.S1: S1Config, Family.S2: S2Config, Family.S3: S3Config}[family]()
        optimizer = torch.optim.Adam(parameters, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps,
            weight_decay=cfg.weight_decay, amsgrad=False, maximize=False, foreach=False,
            fused=False, capturable=False, differentiable=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
            T_0=cfg.T_0, T_mult=cfg.T_mult, eta_min=cfg.eta_min)
        clock = (len(history) - 1) % 2 + 1 if family is Family.S3 else len(history)
        for epoch in range(1, clock + 1):
            scheduler.step(epoch)
    else:
        cfg = ClassifierScientificConfig().training if family is Family.CLASSIFIER else SurrogateTrainingConfig()
        optimizer = torch.optim.Adam(parameters, lr=cfg.learning_rate, betas=cfg.betas,
                                     eps=cfg.epsilon, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=cfg.scheduler_mode,
            factor=cfg.scheduler_factor, patience=cfg.scheduler_patience, threshold=cfg.scheduler_threshold,
            threshold_mode=cfg.scheduler_threshold_mode, cooldown=cfg.scheduler_cooldown,
            min_lr=cfg.scheduler_min_lr, eps=cfg.scheduler_epsilon)
        for row in history:
            scheduler.step(row["metric"])
    scheduler_state = scheduler.state_dict()
    if "mode_worse" in scheduler_state:
        # Exact symbolic encoding of the frozen sentinel, not a trained value.
        scheduler_state["mode_worse"] = "POSITIVE_INFINITY" if cfg.scheduler_mode == "min" else "NEGATIVE_INFINITY"
    return ({"type": "Adam", "param_groups": plain(optimizer.state_dict()["param_groups"])},
            {"type": type(scheduler).__name__, "state": plain(scheduler_state)})


def validate_rng(rng, context, epoch):
    import torch
    import numpy as np
    def torch_state(value):
        require(type(value) is dict and set(value) == {"dtype", "device", "data"}
                and value["dtype"] == "uint8" and value["device"] == "cpu", "Torch RNG schema")
        raw = bytes.fromhex(value["data"])
        require(raw.hex() == value["data"] and 0 < len(raw) <= 16384, "Torch RNG bytes")
        state = torch.tensor(list(raw), dtype=torch.uint8)
        owned = torch.Generator(device="cpu")
        try:
            owned.set_state(state)
        except RuntimeError as error:
            raise ValueError("Malformed Torch RNG state") from error
        require(torch.equal(owned.get_state(), state), "Torch RNG state round trip")
    require(type(rng) is dict, "RNG evidence schema")
    if context.family in (Family.E, Family.H):
        require(set(rng) == {"torch_rng_state", "loader_generator_state"}, "E/H owned RNG states")
        for value in rng.values(): torch_state(value)
    elif context.family is Family.CLASSIFIER:
        from inverse_em.config.classifier import ClassifierScientificConfig
        require(set(rng) == {"dropout_rng_state", "noise_state"}, "Classifier RNG boundary")
        torch_state(rng["dropout_rng_state"])
        noise = rng["noise_state"]
        require(type(noise) is dict and set(noise) == {"augmentation_seed", "event_count", "bit_generator", "bit_generator_state", "sha256"}, "Noise receipt schema")
        require(noise["augmentation_seed"] == ClassifierScientificConfig().training.augmentation_seed
                and type(noise["event_count"]) is int and noise["event_count"] == 8 * epoch
                and noise["bit_generator"] == "PCG64DXSM", "Noise receipt counters/identity")
        payload = {k: noise[k] for k in noise if k != "sha256"}
        require(noise["sha256"] == component_identity(payload).digest, "Noise evidence digest")
        # An isolated non-scientific generator validates the serialized state;
        # no frozen seed is executed and no random values are requested.
        owned = np.random.PCG64DXSM(0)
        owned.state = noise["bit_generator_state"]
        require(component_identity(plain(owned.state)) == component_identity(noise["bit_generator_state"]), "PCG64DXSM state round trip")
    else:
        expected = {"keyed_contract", "next_global_epoch", "exposure"}
        if context.family in (Family.S1, Family.S2): expected.add("torch_rng_state")
        require(set(rng) == expected and rng["keyed_contract"] == plain(asdict(dict(context.dependencies)["rng_contract"]))
                and type(rng["next_global_epoch"]) is int and rng["next_global_epoch"] == epoch + 1
                and type(rng["exposure"]) is int and rng["exposure"] == 0, "Keyed RNG contract/counters")
        if "torch_rng_state" in rng: torch_state(rng["torch_rng_state"])


def validate_continuation(parts, context):
    require(type(parts) is dict and set(parts) == set(continuation_keys(context.family)), "Missing continuation components")
    for key, expected in context.continuation_components:
        require(type(parts[key]) in (dict, list) and bool(parts[key]), "Empty continuation component")
        require(component_identity(parts[key]) == expected, "Continuation component differs from independently bound state: " + key)
    require(type(parts["boundary_state"]) is dict, "Continuation boundary schema")
    if parts["boundary_state"].get("schema") != "closed-task-synthetic-continuation/2":
        raise InsufficientEvidence("Legacy/opaque continuation evidence cannot establish scientific validity")
    # A bounded eight-case, one-update-per-epoch fixture matches the closed
    # bounded engines. Production-size or unknown continuation codecs fail closed.
    counters = parts["counters"]
    limit = 2 if context.family is Family.CLASSIFIER else 4
    require(set(counters) == {"epoch", "updates"} and all(type(v) is int and 1 <= v <= limit for v in counters.values())
            and counters["updates"] == counters["epoch"], "Bounded epoch/update counters")
    history = parts["history"]
    require(type(history) is list and len(history) == counters["epoch"], "History length")
    require([v["epoch"] for v in history] == list(range(1, counters["epoch"] + 1)), "History epochs")
    require(all(type(row) is dict and set(row) == {"epoch", "updates", "metric"}
                and type(row["epoch"]) is int and type(row["updates"]) is int and row["updates"] == row["epoch"] for row in history), "History updates")
    require(all(type(v["metric"]) in (float, int) and math.isfinite(v["metric"]) and v["metric"] >= 0
                and (context.family is not Family.CLASSIFIER or v["metric"] <= 1) for v in history), "History finite/range")
    selection = parts["selection"]
    require(set(selection) == {"epoch", "updates", "metric", "criterion"}, "Selection schema")
    require(type(selection["epoch"]) is int and type(selection["updates"]) is int, "Selection counters")
    expected_criterion = "macro_f1_max" if context.family is Family.CLASSIFIER else "mse_min" if context.family in (Family.E, Family.H) else "cartesian_rmse_min"
    require(selection["criterion"] == expected_criterion, "Selection criterion")
    chosen = (max if context.family is Family.CLASSIFIER else min)(history, key=lambda v: v["metric"])
    require(all(selection[k] == chosen[k] for k in ("epoch", "updates", "metric")), "Selection/history disagreement")
    best = parts["best_snapshot"]
    best_role = "best_validation" if context.family is Family.CLASSIFIER else "BEST"
    best_keys = {"role", "boundary", "selection", "state", "state_identity"}
    if context.family in (Family.S1, Family.S2, Family.S3): best_keys.add("kendall")
    require(set(best) == best_keys
            and best["role"] == best_role and best["boundary"] == boundary(context.family, best_role)
            and best["selection"] == selection, "BEST snapshot metadata")
    best_identity = validate_tensors(best["state"], tensor_schema(context.family))
    require(best["state_identity"] == plain(asdict(best_identity)), "BEST snapshot content identity")
    if "kendall" in best:
        validate_tensors(best["kendall"], (("kendall", (4,)),))
    if selection["epoch"] == counters["epoch"]:
        require(best_identity == context.expected_state, "Current BEST/current state disagreement")
        if "kendall" in best:
            require(best["kendall"] == parts["kendall"], "Current BEST/current Kendall disagreement")
    require(parts["boundary_state"] == {"schema": "closed-task-synthetic-continuation/2", "boundary": context.boundary,
            "validation_complete": True, "scheduler_complete": True, "fixture_cases": 8,
            "batches_per_epoch": 1, "epochs_per_fixture_stage": 2 if context.family is Family.S3 else None}, "Incomplete continuation boundary")
    optimizer = parts["optimizer"]
    params = list(tensor_schema(context.family))
    if context.family in (Family.S1, Family.S2, Family.S3):
        params += [("kendall", (4,))]
        validate_tensors(parts["kendall"], (("kendall", (4,)),))
        stage = parts["stage"]
        require(set(stage) == {"index", "epoch", "transition_pending"} and type(stage["index"]) is int
                and type(stage["epoch"]) is int and type(stage["transition_pending"]) is bool, "Stage schema")
        epoch = counters["epoch"]
        expected_stage = {"index": (epoch - 1) // 2 + 1, "epoch": (epoch - 1) % 2 + 1,
                          "transition_pending": epoch % 2 == 0} if context.family is Family.S3 else {
                              "index": 1, "epoch": epoch, "transition_pending": False}
        require(stage == expected_stage, "Stage/global epoch/transition mismatch")
    require(set(optimizer) == {"parameter_order", "moments", "steps", "hyperparameters"}, "Optimizer schema")
    require(optimizer["parameter_order"] == [p[0] for p in params], "Optimizer parameter order")
    require(set(optimizer["moments"]) == {"exp_avg", "exp_avg_sq"}, "Adam moments")
    for name in ("exp_avg", "exp_avg_sq"):
        validate_tensors(optimizer["moments"][name], params)
    require(all(v[0] >= 0 for item in optimizer["moments"]["exp_avg_sq"]
                for v in struct.iter_unpack("<d", bytes.fromhex(item["data"]))), "Negative Adam second moment")
    clock = parts["stage"]["epoch"] if context.family is Family.S3 else counters["epoch"]
    require(type(optimizer["steps"]) is list and len(optimizer["steps"]) == len(params)
            and all(type(n) is int and n == clock for n in optimizer["steps"]), "Adam clocks")
    expected_optimizer, expected_scheduler = reference_controls(context.family, history)
    require(component_identity(optimizer["hyperparameters"]) == component_identity(expected_optimizer), "Frozen complete Adam controls/LR mismatch")
    require(component_identity(parts["scheduler"]) == component_identity(expected_scheduler), "Frozen scheduler policy/evolved state/LR mismatch")
    validate_rng(parts["rng"], context, counters["epoch"])
    early = parts["early_stop"]
    if context.family in (Family.S1, Family.S3):
        require(early == {"enabled": False}, "Early stopping forbidden")
    elif context.family is Family.S2:
        require(set(early) == {"enabled", "patience", "reference", "bad_epochs", "stopped", "material_test"}
                and early["enabled"] is True and early["patience"] == 200
                and early["material_test"] == "score < es_reference - 1e-5", "S2 material-stop contract")
        reference, bad = float("inf"), 0
        for row in history:
            if row["metric"] < reference - 1e-5:
                reference, bad = row["metric"], 0
            else:
                bad += 1
        require(early["reference"] == reference and early["bad_epochs"] == bad
                and early["stopped"] is False and bad < 200, "S2 material-stop history")
    else:
        if context.family is Family.CLASSIFIER:
            from inverse_em.classifier.selection import SelectionState
            tracker = SelectionState()
            for row in history: tracker.update(row["metric"], row["epoch"], row["updates"])
            expected_early = {"reference": tracker.early_reference, "bad_epochs": tracker.bad_epochs,
                              "patience": 20, "threshold": 1e-4, "stopped": False}
        else:
            from inverse_em.surrogates.training import EarlyStopping
            tracker = EarlyStopping()
            for row in history: tracker.update(row["metric"])
            expected_early = {"reference": tracker.best, "bad_epochs": tracker.bad_epochs,
                              "patience": tracker.patience, "stopped": False}
        require(early == expected_early, "Frozen early-stop state/history mismatch")


def validate(document, context, evidence=None):
    require(type(context) is Context, "Mandatory immutable compatibility context")
    context.__post_init__()
    require(type(document) is dict and set(document) == {"schema", "scope", "context", "state", "buffers", "slots",
            "lineage_states", "continuation"}, "Artifact fields")
    require(document["schema"] == "phase10-synthetic/1" and document["scope"] == "SYNTHETIC_ONLY", "Production acceptance disabled")
    metadata = plain(asdict(context))
    observed = document["context"]
    require(type(observed) is dict and set(observed) == set(metadata), "Incomplete artifact context")
    # Runtime equivalence is not inferred from successful parsing.
    without_runtime = {k: v for k, v in observed.items() if k != "runtime"}
    require(without_runtime == {k: v for k, v in metadata.items() if k != "runtime"}, "Artifact/context mismatch")
    if observed["runtime"] != metadata["runtime"] or context.runtime != current_runtime():
        return Outcome.INSUFFICIENT_EVIDENCE, "Runtime equivalence unestablished"
    require(document["buffers"] == [], "Unexpected buffers")
    require(document["slots"] == list(active_slots(context.family)), "Active-slot semantics")
    require(validate_tensors(document["state"], tensor_schema(context.family)) == context.expected_state, "State-content mismatch")
    if context.family is Family.NORMALIZER:
        n = context.normalizer
        require(document["state"] == [tensor("mean", (4,), n.mean), tensor("std", (4,), n.std)], "Normalizer exact statistics")
    lineage = document["lineage_states"]
    require(type(lineage) is list and len(lineage) == len(context.lineage), "Lineage evidence missing")
    for item, bound in zip(lineage, context.lineage):
        require(type(item) is dict and set(item) == {"field", "source", "state"}, "Lineage evidence schema")
        require(item["field"] == bound.field.value and item["source"] == plain(asdict(bound.source)), "Lineage source binding")
        require(validate_tensors(item["state"], tensor_schema(bound.field)) == bound.state, "Actual E/H state differs from lineage")
    if context.use is Use.CONTINUATION:
        try:
            validate_continuation(document["continuation"], context)
        except InsufficientEvidence as error:
            return Outcome.INSUFFICIENT_EVIDENCE, str(error)
    else:
        require(document["continuation"] is None, "Unexpected continuation state")
    try:
        if context.normalizer is not None or context.lineage:
            if type(evidence) is not EvidenceRegistry:
                raise InsufficientEvidence("Independent supported evidence registry required")
        if context.normalizer is not None:
            n = context.normalizer
            stats = (n.task, n.mean, n.std, n.observation_count, n.scalar_count, n.stages, n.channels, n.dtype, n.ddof)
            facts = {"origin": "explicit-synthetic-clean-training-statistics", "task": n.task,
                     "configuration": asdict(configuration(Family.NORMALIZER)), "statistics": stats,
                     "training_population": asdict(dict(context.dependencies)["training_population"])}
            if n.provenance != identity(Domain.NATIVE, "explicit-synthetic-normalizer-source/1", facts):
                raise InsufficientEvidence("Normalizer provenance does not bind task/training/statistics")
            evidence.resolve(n.mapping, "explicit_statistics", facts)
        for lineage in context.lineage:
            evidence.resolve(lineage.mapping, "tensor_observation", {"field": lineage.field.value,
                "configuration": asdict(lineage.configuration), "representation": lineage.representation,
                "state": asdict(lineage.state)})
    except InsufficientEvidence as error:
        return Outcome.INSUFFICIENT_EVIDENCE, str(error)
    return Outcome.DIRECTLY_COMPATIBLE, "Exact synthetic schema, identities, dependencies and use validated"


def synthetic_document(context, state, *, lineage_states=(), continuation=None):
    """Build untrusted fixture bytes; construction does NOT grant compatibility."""
    return {"schema": "phase10-synthetic/1", "scope": "SYNTHETIC_ONLY", "context": plain(asdict(context)),
            "state": state, "buffers": [], "slots": list(active_slots(context.family)),
            "lineage_states": list(lineage_states), "continuation": continuation}
