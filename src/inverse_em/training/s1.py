"""Bounded S1 mechanics. No production runner or protected-resource loader."""
import copy
from dataclasses import dataclass
import hashlib
import math
import platform
import numpy as np
import torch

from inverse_em.config.s1 import S1Config, SeedRole, execution_seed
from inverse_em.errors import SchemaValidationError
from inverse_em.localization import make_localizer, FieldConsistency, CanonicalTargets
from inverse_em.localization.s1 import (MAX_FIXTURE_CASES, epoch_minibatches,
    objective, training_tensors, validate_clean)
from inverse_em.observations import ComplexFields
from inverse_em.provenance.canonical import canonical_sha256
from inverse_em.provenance.s1 import S1Receipt

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


def optimizer_scheduler(model, fc, config=S1Config()):
    if not isinstance(fc, FieldConsistency):
        raise SchemaValidationError("S1 requires Phase-5 field consistency")
    if any(p.requires_grad for field in (fc.electric, fc.magnetic) for p in field.parameters()):
        raise SchemaValidationError("Surrogates must be frozen")
    if fc.electric.training or fc.magnetic.training:
        raise SchemaValidationError("Surrogates must be in eval mode")
    params = [p for p in model.parameters() if p.requires_grad] + [fc.log_variances]
    optimizer = torch.optim.Adam(params, lr=config.lr, betas=config.betas, eps=config.eps,
        weight_decay=config.weight_decay, amsgrad=False, maximize=False, foreach=False,
        fused=False, capturable=False, differentiable=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.T_0, T_mult=config.T_mult, eta_min=config.eta_min)
    return optimizer, scheduler


@dataclass
class BestState:
    metric: float | None = None
    epoch: int | None = None
    update: int | None = None

    def consider(self, metric, epoch, update):
        if not math.isfinite(metric):
            raise FloatingPointError("Nonfinite validation RMSE")
        if self.metric is None or metric < self.metric:
            self.metric, self.epoch, self.update = float(metric), epoch, update
            return True
        return False


def validate_then_schedule(scheduler, epoch, validation_call):
    """Single shared epoch-end ordering seam, also testable at terminal epoch 200."""
    if type(epoch) is not int or not 1 <= epoch <= S1Config().epochs:
        raise SchemaValidationError("Outside the fixed S1 budget")
    metrics = validation_call()
    if not math.isfinite(metrics["cartesian"]["rmse"]):
        raise FloatingPointError("Nonfinite validation RMSE")
    scheduler.step(epoch)
    return metrics


@dataclass(frozen=True)
class S1Fixture:
    fields: tuple
    targets: CanonicalTargets
    ids: tuple
    role: SeedRole

    def __post_init__(self):
        if not isinstance(self.role, SeedRole) or self.role not in (SeedRole.TRAIN_POPULATION, SeedRole.VALIDATION_POPULATION):
            raise PermissionError("No sealed/robustness fixture execution")
        n = len(self.fields)
        if not 1 <= n <= MAX_FIXTURE_CASES:
            raise PermissionError("Only tiny S1 fixtures are executable")
        if any(type(f) is not ComplexFields for f in self.fields):
            raise SchemaValidationError("Fixture requires clean complex fields")
        if self.targets.active_count != 1 or self.targets.rho.shape[0] != n:
            raise SchemaValidationError("Fixture targets must have shape (N,1)")
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


class BoundedS1Trainer:
    """At most 16 cases/role and four cumulative fixture epochs; no scientific run API."""
    def __init__(self, model, fc, training, validation, normalizer, config=S1Config()):
        if training.role is not SeedRole.TRAIN_POPULATION or validation.role is not SeedRole.VALIDATION_POPULATION:
            raise PermissionError("Training/validation roles cannot be interchanged")
        if set(training.ids) & set(validation.ids):
            raise SchemaValidationError("Training/validation identities overlap")
        self.model, self.fc, self.training, self.validation = model, fc, training, validation
        self.normalizer, self.config = normalizer, config
        self.optimizer, self.scheduler = optimizer_scheduler(model, fc, config)
        self.completed_epoch = self.updates = 0
        self.best = BestState()
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
            raise RuntimeError("Only completed epoch boundaries can continue")
        if type(epochs) is not int or epochs < 1 or self.completed_epoch + epochs > MAX_FIXTURE_EPOCHS:
            raise PermissionError("Production training is unavailable; four fixture epochs maximum")
        for epoch in range(self.completed_epoch + 1, self.completed_epoch + epochs + 1):
            self.boundary = "IN_EPOCH"
            self.model.train()
            x, clean = training_tensors(self.training.fields, self.training.ids, self.normalizer, epoch)
            orders = epoch_minibatches(len(self.training.fields), epoch)
            generator = torch.Generator(device="cpu").manual_seed(self.config.seed(SeedRole.MINIBATCH_ORDER) + epoch)
            loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(len(x))),
                batch_sampler=[ix.tolist() for ix in orders], num_workers=0, generator=generator)
            sums = np.zeros(7, np.float64)
            lr = self.optimizer.param_groups[0]["lr"]
            for (ix,) in loader:
                self.optimizer.zero_grad(set_to_none=True)
                targets = CanonicalTargets(self.training.targets.rho[ix], self.training.targets.phi[ix])
                loss, supervised, kendall, channels = objective(self.model, x[ix], targets, clean[ix], self.fc)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite S1 loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for g in self.optimizer.param_groups for p in g["params"]):
                    raise FloatingPointError("Nonfinite S1 gradient")
                self.optimizer.step(); self.updates += 1
                sums += np.array([float(supervised.detach()), *channels.detach().tolist(),
                                  float(kendall.detach()), float(loss.detach())]) * len(ix)
            metrics = validate_then_schedule(self.scheduler, epoch,
                lambda: validate_clean(self.model, self.validation.fields, self.validation.targets, self.normalizer))
            self.completed_epoch = epoch
            self.boundary = BOUNDARY
            values = sums / len(x)
            record = {"epoch": epoch, "updates": self.updates, "lr": lr,
                "lr_next": self.optimizer.param_groups[0]["lr"], "validation": metrics,
                "supervised": values[0], "fc_channels": values[1:5].tolist(), "kendall": values[5],
                "total": values[6], "kendall_s": self.fc.log_variances.detach().tolist(),
                "effective_weights": (.015 * torch.exp(-self.fc.log_variances.detach())).tolist(),
                "order_sha256": hashlib.sha256(np.concatenate(orders).astype("<i8").tobytes()).hexdigest()}
            if self.best.consider(metrics["cartesian"]["rmse"], epoch, self.updates):
                self.best_checkpoint = self._capture(self.history + [record])
                self.best_checkpoint["checkpoint_role"] = "BEST"
                if on_best is not None:
                    on_best(copy.deepcopy(self.best_checkpoint))
            self.history.append(record)
        self.terminal_checkpoint = self.state_dict()
        self.terminal_checkpoint["checkpoint_role"] = "TERMINAL"
        return copy.deepcopy(self.history)

    def _capture(self, history):
        return copy.deepcopy({"checkpoint_role": "CONTINUATION", "model": _module_state(self.model), "kendall": self.fc.log_variances.detach().clone(),
            "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
            "completed_epoch": self.completed_epoch, "next_epoch": self.completed_epoch + 1,
            "updates": self.updates, "best": vars(self.best).copy(), "boundary": self.boundary,
            "bindings": self.bindings, "torch_rng": torch.get_rng_state().clone(),
            "history": history, "history_position": len(history)})

    def state_dict(self):
        if self.boundary != BOUNDARY:
            raise RuntimeError("No mid-epoch snapshots")
        state = self._capture(self.history)
        state["best_checkpoint"] = copy.deepcopy(self.best_checkpoint)
        return state

    def load_state_dict(self, state):
        if state["bindings"] != self.bindings or state["boundary"] != BOUNDARY:
            raise SchemaValidationError("S1 continuation identities/boundary differ")
        epoch = state["completed_epoch"]
        if (type(epoch) is not int or not 0 <= epoch <= MAX_FIXTURE_EPOCHS or state["next_epoch"] != epoch + 1
                or state["history_position"] != epoch or len(state["history"]) != epoch
                or state["updates"] != epoch * math.ceil(len(self.training.fields)/128)):
            raise SchemaValidationError("Inconsistent S1 epoch-boundary counters")
        self.model.load_state_dict(state["model"], strict=True)
        with torch.no_grad():
            self.fc.log_variances.copy_(state["kendall"])
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        self.scheduler.load_state_dict(copy.deepcopy(state["scheduler"]))
        self.completed_epoch, self.updates = epoch, state["updates"]
        self.best = BestState(**state["best"])
        self.history = copy.deepcopy(state["history"])
        self.best_checkpoint = copy.deepcopy(state.get("best_checkpoint"))
        if self.best_checkpoint is None and self.best.epoch == epoch:
            self.best_checkpoint = copy.deepcopy(state)
            self.best_checkpoint["checkpoint_role"] = "BEST"
        self.boundary = BOUNDARY
        self.model.eval()
        self.terminal_checkpoint = None
        # No stochastic forward operation exists; owned loader generators and keyed
        # NumPy draws make global RNG restoration unnecessary and preserve the caller.

    def receipt(self):
        b = self.bindings
        return S1Receipt(self.config.sha256, b["populations"], b["normalizer"], b["phase5"],
            b["deployment_surrogates"], b["rng_convention"], self.updates, self.best.epoch,
            self.best.update, self.completed_epoch, b["runtime_versions"])
