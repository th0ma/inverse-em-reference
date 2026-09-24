"""Bounded S2 mechanics. No production runner or protected-resource loader."""
import copy
from dataclasses import dataclass
import hashlib
import math
import platform
import numpy as np
import torch

from inverse_em.config.s2 import S2Config, SeedRole, execution_seed
from inverse_em.errors import SchemaValidationError
from inverse_em.localization import make_localizer, FieldConsistency, CanonicalTargets
from inverse_em.localization.s2 import (MAX_FIXTURE_CASES, epoch_minibatches,
    objective, training_tensors, validate_clean, validate_targets, _inputs)
from inverse_em.observations import ComplexFields
from inverse_em.provenance.canonical import canonical_sha256
from inverse_em.provenance.s2 import S2Receipt

BOUNDARY = "EPOCH_COMPLETE_VALIDATION_AND_SCHEDULER_COMPLETE"
MAX_FIXTURE_EPOCHS = 4


def initialize_localizer(role=SeedRole.MODEL_INITIALIZATION):
    seed = execution_seed(role, SeedRole.MODEL_INITIALIZATION)
    dtype = torch.get_default_dtype()
    with torch.random.fork_rng(devices=[]):
        try:
            torch.random.default_generator.manual_seed(seed)
            torch.set_default_dtype(torch.float32)
            with torch.device("cpu"):
                model = make_localizer().double()
        finally:
            torch.set_default_dtype(dtype)
    return model


def optimizer_scheduler(model, fc, config=S2Config()):
    if not isinstance(fc, FieldConsistency):
        raise SchemaValidationError("S2 requires Phase-5 field consistency")
    if any(p.requires_grad for field in (fc.electric, fc.magnetic) for p in field.parameters()):
        raise SchemaValidationError("Surrogates must be frozen")
    if fc.electric.training or fc.magnetic.training:
        raise SchemaValidationError("Surrogates must be in eval mode")
    params = [p for p in model.parameters() if p.requires_grad] + [fc.log_variances]
    if sum(p.numel() for p in params) != 399437:
        raise SchemaValidationError("S2 optimized parameter count mismatch")
    optimizer = torch.optim.Adam(params, lr=config.lr, betas=config.betas, eps=config.eps,
        weight_decay=config.weight_decay, amsgrad=False, maximize=False, foreach=False,
        fused=False, capturable=False, differentiable=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.T_0, T_mult=config.T_mult, eta_min=config.eta_min)
    return optimizer, scheduler


@dataclass
class BestState:
    metric: float = math.inf
    epoch: int = 0
    update: int = 0

    def consider(self, metric, epoch, update):
        if not math.isfinite(metric):
            raise FloatingPointError("Nonfinite validation RMSE")
        if metric < self.metric:
            self.metric, self.epoch, self.update = float(metric), epoch, update
            return True
        return False


@dataclass(frozen=True)
class S2Fixture:
    fields: tuple
    targets: CanonicalTargets
    ids: tuple
    role: SeedRole

    def __post_init__(self):
        if not isinstance(self.role, SeedRole) or self.role not in (SeedRole.TRAIN_POPULATION, SeedRole.VALIDATION_POPULATION):
            raise PermissionError("No sealed/robustness fixture execution")
        n = len(self.fields)
        if not 1 <= n <= MAX_FIXTURE_CASES:
            raise PermissionError("Only tiny S2 fixtures are executable")
        if any(type(f) is not ComplexFields for f in self.fields):
            raise SchemaValidationError("Fixture requires clean complex fields")
        if self.targets.active_count != 2 or self.targets.rho.shape[0] != n:
            raise SchemaValidationError("Fixture targets must have shape (N,2)")
        validate_targets(self.targets, n)
        if len(self.ids) != n or len(set(self.ids)) != n or any(not isinstance(x, str) or not x.strip() for x in self.ids):
            raise SchemaValidationError("Unique nonempty fixture IDs required")

    @property
    def identity(self):
        h = hashlib.sha256()
        h.update(canonical_sha256({"role": self.role, "ids": self.ids}).encode())
        for f in self.fields:
            h.update(f.electric.tobytes()); h.update(f.magnetic.tobytes())
        for t in (self.targets.rho, self.targets.phi):
            h.update(t.detach().cpu().numpy().astype("<f8").tobytes())
        return h.hexdigest()


def _module_state(module):
    return {name: t.detach().cpu().clone() for name, t in module.state_dict().items()}


def _tensor_identity(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class BoundedS2Trainer:
    """At most 16 cases/role and four cumulative fixture epochs; no scientific run API."""
    def __init__(self, model, fc, training, validation, normalizer, config=S2Config()):
        if training.role is not SeedRole.TRAIN_POPULATION or validation.role is not SeedRole.VALIDATION_POPULATION:
            raise PermissionError("Training/validation roles cannot be interchanged")
        if set(training.ids) & set(validation.ids):
            raise SchemaValidationError("Training/validation identities overlap")
        if type(config) is not S2Config:
            raise SchemaValidationError("Frozen S2 config required")
        training.__post_init__()
        validation.__post_init__()
        _inputs(training.fields, normalizer)
        _inputs(validation.fields, normalizer)
        self.model, self.fc, self.training, self.validation = model, fc, training, validation
        self.normalizer, self.config = normalizer, config
        self.optimizer, self.scheduler = optimizer_scheduler(model, fc, config)
        self.completed_epoch = self.updates = 0
        self.best = BestState()
        self.material = MaterialState()
        self.termination_reason = None
        self.history = []
        self.best_checkpoint = None
        self.terminal_checkpoint = None
        self.boundary = BOUNDARY
        self.bindings = {
            "configuration": config.sha256, "populations": (training.identity, validation.identity),
            "normalizer": canonical_sha256({"task": normalizer.task, "mean": normalizer.mean.tolist(),
                "std": normalizer.std.tolist(), "count": normalizer.observation_count, "ddof": normalizer.ddof}),
            "phase5": config.phase5_sha256,
            "deployment_surrogates": (config.electric_sha256, config.magnetic_sha256),
            "fixture_surrogate_tensors": (_tensor_identity(fc.electric), _tensor_identity(fc.magnetic)),
            "fixture_angles": hashlib.sha256(fc.observation_angles.numpy().tobytes()).hexdigest(),
            "rng_convention": config.rng_convention, "seed_roles": config.seed_roles,
            "order_identity": "historical_ALL_ORDER_v1", "noise_identity": "historical_SNR_E_H_30x2_v1",
            "runtime_versions": (("python", platform.python_version()), ("numpy", np.__version__), ("torch", str(torch.__version__))),
        }

    def run(self, epochs=1, *, on_best=None):
        if self.boundary != BOUNDARY:
            raise RuntimeError("Only post-scheduler epoch boundaries can continue")
        self.training.__post_init__()
        self.validation.__post_init__()
        if (self.termination_reason == "early_stopping" or type(epochs) is not int
                or epochs < 1 or self.completed_epoch + epochs > MAX_FIXTURE_EPOCHS):
            raise PermissionError("Four cumulative fixture epochs maximum; no stopped-run continuation")
        self.termination_reason = None
        for epoch in range(self.completed_epoch + 1, self.completed_epoch + epochs + 1):
            self.boundary = "IN_EPOCH"
            self.model.train()
            x, clean = training_tensors(self.training.fields, self.training.ids, self.normalizer, epoch)
            orders = epoch_minibatches(len(self.training.fields), epoch)
            sums = np.zeros(7, np.float64)
            lr = self.optimizer.param_groups[0]["lr"]
            for indices in orders:
                ix = torch.as_tensor(indices, dtype=torch.int64)
                self.optimizer.zero_grad(set_to_none=True)
                targets = CanonicalTargets(self.training.targets.rho[ix], self.training.targets.phi[ix])
                loss, supervised, kendall, channels = objective(self.model, x[ix], targets, clean[ix], self.fc)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite S2 loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for group in self.optimizer.param_groups for p in group["params"]):
                    raise FloatingPointError("Nonfinite S2 gradient")
                self.optimizer.step()
                self.updates += 1
                sums += np.array([float(supervised.detach()), *channels.detach().tolist(),
                                  float(kendall.detach()), float(loss.detach())]) * len(ix)
            metrics = validate_clean(self.model, self.validation.fields, self.validation.targets, self.normalizer)
            score = metrics["cartesian"]["rmse"]
            self.completed_epoch = epoch
            new_best = self.best.consider(score, epoch, self.updates)
            self.material.observe(score)
            values = sums / len(x)
            record = {"epoch": epoch, "updates": self.updates, "lr": lr, "validation": metrics,
                "supervised": values[0], "fc_channels": values[1:5].tolist(), "kendall": values[5],
                "total": values[6], "kendall_s": self.fc.log_variances.detach().tolist(),
                "effective_weights": (.015 * torch.exp(-self.fc.log_variances.detach())).tolist(),
                "best": self.best.metric, "best_epoch": self.best.epoch,
                "es_reference": self.material.es_reference, "stale": self.material.stale,
                "order_sha256": hashlib.sha256(np.concatenate(orders).astype("<i8").tobytes()).hexdigest()}
            self.boundary = "VALIDATION_AND_ACCOUNTING_COMPLETE_PRE_SCHEDULER"
            if new_best:
                self.best_checkpoint = self._capture(self.history + [record])
                self.best_checkpoint["checkpoint_role"] = "BEST"
                if on_best is not None:
                    on_best(copy.deepcopy(self.best_checkpoint))
            self.history.append(record)
            self.scheduler.step(epoch)
            self.boundary = BOUNDARY
            if self.material.should_stop:
                self.termination_reason = "early_stopping"
                break
        if self.termination_reason != "early_stopping":
            self.termination_reason = "fixture_budget" if self.completed_epoch == MAX_FIXTURE_EPOCHS else "bounded_call_complete"
        self.terminal_checkpoint = self.state_dict()
        self.terminal_checkpoint["checkpoint_role"] = "TERMINAL"
        return copy.deepcopy(self.history)

    def _capture(self, history):
        return copy.deepcopy({"checkpoint_role": "CONTINUATION", "model": _module_state(self.model),
            "kendall": self.fc.log_variances.detach().clone(), "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(), "completed_epoch": self.completed_epoch,
            "next_epoch": self.completed_epoch + 1, "updates": self.updates,
            "best": vars(self.best).copy(), "material": vars(self.material).copy(),
            "patience": 200, "material_expression": self.config.material_test,
            "termination_reason": self.termination_reason, "boundary": self.boundary,
            "bindings": self.bindings, "torch_rng": torch.get_rng_state().clone(),
            "history": history, "history_position": len(history)})

    def state_dict(self):
        if self.boundary != BOUNDARY:
            raise RuntimeError("No mid-epoch or pre-scheduler continuation snapshots")
        state = self._capture(self.history)
        state["best_checkpoint"] = copy.deepcopy(self.best_checkpoint)
        return state

    def load_state_dict(self, state):
        # Validate before touching any live tensor/optimizer. This is a consistency
        # check for in-memory fixture snapshots, not an authority credential.
        if (state["bindings"] != self.bindings or state["boundary"] != BOUNDARY
                or state["checkpoint_role"] != "CONTINUATION"):
            raise SchemaValidationError("S2 continuation identities/role/boundary differ")
        epoch = state["completed_epoch"]
        if (type(epoch) is not int or not 0 <= epoch <= MAX_FIXTURE_EPOCHS
                or state["next_epoch"] != epoch + 1 or state["history_position"] != epoch
                or len(state["history"]) != epoch
                or state["updates"] != epoch * math.ceil(len(self.training.fields)/128)
                or state["patience"] != 200 or state["material_expression"] != self.config.material_test
                or state["scheduler"]["last_epoch"] != epoch):
            raise SchemaValidationError("Inconsistent or out-of-bounds S2 continuation")
        expected_best, expected_material = BestState(), MaterialState()
        for number, row in enumerate(state["history"], 1):
            if row["epoch"] != number or row["updates"] != number * math.ceil(len(self.training.fields)/128):
                raise SchemaValidationError("Inconsistent S2 history counters")
            expected_best.consider(row["validation"]["cartesian"]["rmse"], number, row["updates"])
            expected_material.observe(row["validation"]["cartesian"]["rmse"])
            if (row["best"] != expected_best.metric or row["best_epoch"] != expected_best.epoch
                    or row["es_reference"] != expected_material.es_reference or row["stale"] != expected_material.stale):
                raise SchemaValidationError("S2 selection history mismatch")
        if state["best"] != vars(expected_best) or state["material"] != vars(expected_material):
            raise SchemaValidationError("S2 selection state mismatch")
        best = state["best_checkpoint"]
        if epoch and (best is None or best["checkpoint_role"] != "BEST"
                or best["completed_epoch"] != expected_best.epoch
                or best["updates"] != expected_best.update or best["best"] != state["best"]
                or best["boundary"] != "VALIDATION_AND_ACCOUNTING_COMPLETE_PRE_SCHEDULER"):
            raise SchemaValidationError("Missing/inconsistent selected BEST")
        reason = state["termination_reason"]
        if reason not in (None, "early_stopping", "fixture_budget", "bounded_call_complete"):
            raise SchemaValidationError("Invalid termination reason")
        if (reason == "early_stopping") != expected_material.should_stop:
            raise SchemaValidationError("Inconsistent termination state")
        self.model.load_state_dict(state["model"], strict=True)
        with torch.no_grad():
            self.fc.log_variances.copy_(state["kendall"])
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        self.scheduler.load_state_dict(copy.deepcopy(state["scheduler"]))
        self.completed_epoch, self.updates = epoch, state["updates"]
        self.best, self.material = expected_best, expected_material
        self.history = copy.deepcopy(state["history"])
        self.best_checkpoint = copy.deepcopy(best)
        self.termination_reason = reason
        self.boundary = BOUNDARY
        self.model.eval()
        self.terminal_checkpoint = None
        # No stochastic forward operation: keyed NumPy streams and deterministic
        # localizer forward preserve caller RNG without restoring external state.

    def receipt(self):
        return S2Receipt(self.bindings, self.updates, self.best.epoch,
                         self.best.update, self.completed_epoch, self.termination_reason)


@dataclass
class MaterialState:
    es_reference: float | None = None
    stale: int = 0

    def observe(self, score):
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite validation RMSE")
        if self.es_reference is None or score < self.es_reference - 1e-5:
            self.es_reference = score
            self.stale = 0
        else:
            self.stale += 1

    @property
    def should_stop(self):
        return self.stale >= 200


def replay_history(rows):
    """Replay <=600 scalar history records; no inference/optimization/I/O."""
    if not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= 600:
        raise PermissionError("Scalar replay requires 1..600 records")
    best, material = BestState(), MaterialState()
    last_reset = None
    first_stop = None
    for epoch, row in enumerate(rows, 1):
        if row["epoch"] != epoch or row["updates"] != epoch * 547:
            raise SchemaValidationError("Historical epoch/update mismatch")
        score = row["validation"]["canonical_cartesian_rmse"]
        best.consider(score, epoch, row["updates"])
        material.observe(score)
        if material.stale == 0:
            last_reset = epoch
        expected = {"best": best.metric, "best_epoch": best.epoch,
                    "es_reference": material.es_reference,
                    "epochs_without_material_improvement": material.stale}
        if any(row[key] != value for key, value in expected.items()):
            raise SchemaValidationError("Historical selection mismatch")
        if material.should_stop and first_stop is None:
            first_stop = epoch
    return {"rows": len(rows), "best_epoch": best.epoch, "best_update": best.update,
            "last_material_reset": last_reset, "first_stop": first_stop,
            "stale": material.stale, "terminal_updates": rows[-1]["updates"]}
