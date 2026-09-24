"""Two-stage/four-epoch bounded S3 engine; no production or checkpoint-file API."""
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import platform
import numpy as np
import torch
from inverse_em.config.s3 import S3Config, SeedRole, execution_seed
from inverse_em.config.phase1 import Phase1ScientificConfig
from inverse_em.config.phase2 import Phase2ScientificConfig
from inverse_em.config.localization import LocalizationScientificConfig
from inverse_em.errors import SchemaValidationError
from inverse_em.localization import make_localizer, FieldConsistency, CanonicalTargets
from inverse_em.localization.model import CircularLocalizer
from inverse_em.localization import s3
from inverse_em.provenance.canonical import canonical_sha256
from inverse_em.provenance.s3 import S3Receipt

BOUNDARY = "S3_V1_EPOCH_VALIDATION_BEST_HISTORY_SCHEDULER_COMPLETE"
BEST_BOUNDARY = "S3_V1_VALIDATION_COMPLETE_PRE_SCHEDULER"
MAX_STAGES = 2
EPOCHS_PER_FIXTURE_STAGE = 2
MAX_EPOCHS = 4


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


def optimizer_scheduler(model, fc, config):
    if type(model) is not CircularLocalizer or type(fc) is not FieldConsistency:
        raise SchemaValidationError("Phase-5 model and FC required")
    if any(m.training or any(p.requires_grad for p in m.parameters()) for m in (fc.electric, fc.magnetic)):
        raise SchemaValidationError("FC surrogates must be frozen in eval mode")
    params = list(model.parameters()) + [fc.log_variances]
    if sum(p.numel() for p in params) != 399437:
        raise SchemaValidationError("S3 parameter count mismatch")
    opt = torch.optim.Adam(params, lr=config.lr, betas=config.betas, eps=config.eps,
        weight_decay=config.weight_decay, amsgrad=False, maximize=False, foreach=False,
        fused=False, capturable=False, differentiable=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,
        T_0=config.T_0, T_mult=config.T_mult, eta_min=config.eta_min)
    return opt, sched


@dataclass
class GlobalBest:
    value: float = math.inf
    epoch: int = 0
    stage: int = 0
    stage_epoch: int = 0
    updates: int = 0

    def observe(self, score, epoch, stage, local, updates):
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite S3 validation RMSE")
        if score < self.value:
            self.value, self.epoch, self.stage, self.stage_epoch, self.updates = float(score), epoch, stage, local, updates
            return True
        return False


@dataclass(frozen=True)
class S3Fixture:
    fields: tuple
    targets: CanonicalTargets
    ids: tuple
    role: SeedRole
    stage: int

    def __post_init__(self):
        if self.role not in (SeedRole.TRAIN_POPULATION, SeedRole.VALIDATION_POPULATION) or not isinstance(self.role, SeedRole):
            raise PermissionError("Only bounded training/validation fixtures")
        if self.role is SeedRole.VALIDATION_POPULATION and self.stage != 8:
            raise SchemaValidationError("Fixed Stage-8 validation required")
        n = len(self.fields)
        s3.validate_targets(self.targets, n, self.stage)
        if any(type(f) is not s3.ComplexFields for f in self.fields):
            raise SchemaValidationError("Clean fixture fields required")
        expected = tuple(s3.configuration_id(self.role, self.stage, r, p)
            for r, p in zip(self.targets.rho.detach().numpy(), self.targets.phi.detach().numpy()))
        if self.ids != expected or len(set(self.ids)) != n:
            raise SchemaValidationError("Unique canonical fixture identities required")

    @property
    def identity(self):
        h = hashlib.sha256(canonical_sha256((self.role, self.stage, self.ids)).encode())
        for f in self.fields:
            h.update(f.electric.tobytes())
            h.update(f.magnetic.tobytes())
        return h.hexdigest()


def tensor_identity(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _module_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


class BoundedS3Trainer:
    def __init__(self, model, fc, stages, validation, normalizer, config=S3Config()):
        if type(config) is not S3Config:
            raise SchemaValidationError("Frozen S3 config required")
        config.__post_init__()
        if not isinstance(stages, tuple) or not 1 <= len(stages) <= MAX_STAGES:
            raise PermissionError("At most two fixture stages")
        for i, data in enumerate(stages, 1):
            if type(data) is not S3Fixture or data.role is not SeedRole.TRAIN_POPULATION or data.stage != i:
                raise SchemaValidationError("Sequential fixture stages 1..2 required")
            data.__post_init__()
        if type(validation) is not S3Fixture or validation.role is not SeedRole.VALIDATION_POPULATION:
            raise SchemaValidationError("Validation fixture required")
        validation.__post_init__()
        physical = [set(x.rsplit("-", 1)[-1] for x in d.ids) for d in (*stages, validation)]
        if any(physical[i] & physical[j] for i in range(len(physical)) for j in range(i)):
            raise SchemaValidationError("Fixture geometry overlap")
        for data in (*stages, validation):
            s3._inputs(data.fields, normalizer)
        if (Phase1ScientificConfig().sha256, Phase2ScientificConfig().sha256, LocalizationScientificConfig().sha256) != (
                config.phase1_sha256, config.phase2_sha256, config.phase5_sha256):
            raise SchemaValidationError("Closed dependency configuration changed")
        self.model, self.fc, self.stages, self.validation = model, fc, stages, validation
        self.normalizer, self.config = normalizer, config
        self.optimizer, self.scheduler = optimizer_scheduler(model, fc, config)
        self.global_epoch = self.updates = self.stage_epoch = 0
        self.stage = 1
        self.transition_pending = False
        self.history = []
        self.best = GlobalBest()
        self.best_checkpoint = self.terminal_checkpoint = None
        self.boundary = BOUNDARY
        self.bindings = self._bindings()

    def _bindings(self):
        c, n = self.config, self.normalizer
        return {"config": c.sha256, "populations": tuple(d.identity for d in (*self.stages, self.validation)),
            "normalizer": canonical_sha256({"task": n.task, "mean": n.mean.tolist(), "std": n.std.tolist(),
                "count": n.observation_count, "scalar_count": n.scalar_count_per_channel, "stages": n.curriculum_stages, "ddof": n.ddof}),
            "dependencies": (c.phase1_sha256, c.phase2_sha256, c.phase5_sha256),
            "physics": "b0d8200ef30d05720f0b697822dc0e037935243beea9084edf410c9725c32005",
            "surrogates": (c.electric_sha256, c.magnetic_sha256),
            "fixture_surrogates": (tensor_identity(self.fc.electric), tensor_identity(self.fc.magnetic)),
            "angles": hashlib.sha256(self.fc.observation_angles.detach().numpy().tobytes()).hexdigest(),
            "rng": c.rng_convention, "boundary": BOUNDARY, "fixture_budget": (len(self.stages), 2, 4),
            "runtime": (platform.python_version(), np.__version__, str(torch.__version__), "cpu", "float64")}

    def _validate_inputs(self):
        for d in (*self.stages, self.validation):
            d.__post_init__()
        if self._bindings() != self.bindings:
            raise SchemaValidationError("Bounded input identities changed")

    def _advance_stage(self):
        if self.transition_pending:
            if self.stage >= len(self.stages):
                raise PermissionError("Fixture stages exhausted")
            self.stage += 1
            self.stage_epoch = 0
            self.optimizer, self.scheduler = optimizer_scheduler(self.model, self.fc, self.config)
            self.transition_pending = False

    def run(self, epochs=1, *, on_best=None):
        if self.boundary != BOUNDARY:
            raise RuntimeError("Only completed epoch continuation allowed")
        if type(epochs) is not int or epochs < 1 or self.global_epoch+epochs > min(MAX_EPOCHS, len(self.stages)*2):
            raise PermissionError("At most four cumulative fixture epochs")
        self._validate_inputs()
        for _ in range(epochs):
            self._advance_stage()
            self.boundary = "IN_EPOCH"
            epoch, local = self.global_epoch+1, self.stage_epoch+1
            data = self.stages[self.stage-1]
            x, clean, receipt = s3.training_tensors(data.fields, data.ids, self.normalizer, self.stage, epoch)
            batches = s3.epoch_minibatches(len(x), self.stage, epoch)
            self.model.train()
            sums = np.zeros(7)
            lr = self.optimizer.param_groups[0]["lr"]
            for indices in batches:
                ix = torch.as_tensor(indices, dtype=torch.int64)
                self.optimizer.zero_grad(set_to_none=True)
                targets = CanonicalTargets(data.targets.rho[ix], data.targets.phi[ix])
                loss, loc, k, channels = s3.objective(self.model, x[ix], targets, clean[ix], self.fc)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite S3 loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for group in self.optimizer.param_groups for p in group["params"]):
                    raise FloatingPointError("Nonfinite S3 gradient")
                self.optimizer.step()
                self.updates += 1
                sums += np.array([float(loc.detach()), *channels.detach().tolist(), float(k.detach()), float(loss.detach())])*len(ix)
            score, metrics = s3.validate_clean(self.model, self.validation.fields, self.validation.targets, self.normalizer)
            self.global_epoch, self.stage_epoch = epoch, local
            improved = self.best.observe(score, epoch, self.stage, local, self.updates)
            record = {"global_epoch": epoch, "stage": self.stage, "stage_epoch": local, "updates": self.updates,
                "lr": lr, "validation_rmse": score, "validation": metrics, "losses": (sums/len(x)).tolist(),
                "kendall_s": self.fc.log_variances.detach().tolist(),
                "effective_weights": (.015*torch.exp(-self.fc.log_variances.detach())).tolist(),
                "strict_best": improved, "best_rmse": self.best.value, **receipt,
                "order_sha256": hashlib.sha256(np.concatenate(batches).astype("<i8").tobytes()).hexdigest()}
            self.boundary = BEST_BOUNDARY
            if improved:
                self.best_checkpoint = {"checkpoint_role": "BEST", "boundary": BEST_BOUNDARY,
                    "model": _module_state(self.model), "kendall": self.fc.log_variances.detach().clone(),
                    "metadata": copy.deepcopy(record)}
                if on_best is not None:
                    on_best(copy.deepcopy(self.best_checkpoint))
            self.history.append(record)
            self.scheduler.step(local)
            self.transition_pending = local == EPOCHS_PER_FIXTURE_STAGE
            self.boundary = BOUNDARY
        self.terminal_checkpoint = self.state_dict()
        self.terminal_checkpoint["checkpoint_role"] = "TERMINAL"
        return copy.deepcopy(self.history)

    def state_dict(self):
        if self.boundary != BOUNDARY:
            raise RuntimeError("Cannot capture mid-epoch continuation")
        return copy.deepcopy({"checkpoint_role": "CONTINUATION", "boundary": self.boundary,
            "model": _module_state(self.model), "kendall": self.fc.log_variances.detach().clone(),
            "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
            "global_epoch": self.global_epoch, "stage": self.stage, "stage_epoch": self.stage_epoch,
            "updates": self.updates, "transition_pending": self.transition_pending, "best": vars(self.best),
            "best_checkpoint": self.best_checkpoint, "history": self.history,
            "history_position": len(self.history), "bindings": self.bindings})

    def load_state_dict(self, state):
        self._validate_inputs()
        if state["checkpoint_role"] != "CONTINUATION" or state["boundary"] != BOUNDARY or state["bindings"] != self.bindings:
            raise SchemaValidationError("Incompatible continuation role/boundary/identities")
        epoch = state["global_epoch"]
        if type(epoch) is not int or not self.global_epoch <= epoch <= min(4, len(self.stages)*2):
            raise PermissionError("Continuation exceeds budget or rewinds live engine")
        stage, local = ((epoch-1)//2+1, (epoch-1)%2+1) if epoch else (1, 0)
        if (state["stage"], state["stage_epoch"], state["updates"], state["history_position"], len(state["history"])) != (stage, local, epoch, epoch, epoch):
            raise SchemaValidationError("Inconsistent fixture counters")
        if state["transition_pending"] != (local == 2) or state["scheduler"]["last_epoch"] != local:
            raise SchemaValidationError("Invalid stage transition/scheduler state")
        best = GlobalBest()
        for i, row in enumerate(state["history"], 1):
            st, lo = (i-1)//2+1, (i-1)%2+1
            if (row["global_epoch"], row["stage"], row["stage_epoch"], row["updates"]) != (i, st, lo, i):
                raise SchemaValidationError("Invalid continuation history")
            changed = best.observe(row["validation_rmse"], i, st, lo, i)
            if row["strict_best"] != changed or row["best_rmse"] != best.value:
                raise SchemaValidationError("Global BEST/history mismatch")
        if state["best"] != vars(best):
            raise SchemaValidationError("Global BEST metadata mismatch")
        snap = state["best_checkpoint"]
        if epoch and (snap is None or snap["checkpoint_role"] != "BEST" or snap["boundary"] != BEST_BOUNDARY
                or snap["metadata"] != state["history"][best.epoch-1]):
            raise SchemaValidationError("BEST snapshot mismatch")
        model_state = self.model.state_dict()
        if state["model"].keys() != model_state.keys() or any(
                v.shape != model_state[k].shape or v.dtype != torch.float64 or not torch.isfinite(v).all()
                for k, v in state["model"].items()):
            raise SchemaValidationError("Invalid continuation model state")
        if state["kendall"].shape != (4,) or state["kendall"].dtype != torch.float64 or not torch.isfinite(state["kendall"]).all():
            raise SchemaValidationError("Invalid Kendall state")
        optimizer, scheduler = self._validated_continuation_optimizer(state, local)
        self.model.load_state_dict(state["model"])
        with torch.no_grad():
            self.fc.log_variances.copy_(state["kendall"])
        self.optimizer, self.scheduler = optimizer, scheduler
        self.global_epoch, self.stage, self.stage_epoch = epoch, stage, local
        self.updates = state["updates"]
        self.transition_pending = state["transition_pending"]
        self.best, self.history, self.best_checkpoint = best, copy.deepcopy(state["history"]), copy.deepcopy(snap)
        self.model.eval()
        self.terminal_checkpoint = None

    def _validated_continuation_optimizer(self, state, local):
        """Prepare restoration without touching live tensors or training objects.

        The frozen constructor is the authority for every serialized option,
        including runtime Adam flags. Replaying only the scheduler clock admits
        evolved LR while rejecting altered future scheduling. No optimizer step,
        model initialization, RNG draw or scientific evaluation occurs here.
        """
        optimizer, scheduler = optimizer_scheduler(self.model, self.fc, self.config)
        if local:
            scheduler.step(local)
        incoming = state["optimizer"]
        if (type(incoming) is not dict or incoming.keys() != {"state", "param_groups"}
                or canonical_sha256(incoming["param_groups"]) != canonical_sha256(optimizer.state_dict()["param_groups"])
                or canonical_sha256(state["scheduler"]) != canonical_sha256(scheduler.state_dict())):
            raise SchemaValidationError("Incompatible frozen Adam/scheduler continuation")
        moments = incoming["state"]
        params = optimizer.param_groups[0]["params"]
        expected_ids = set(range(len(params))) if local else set()
        if type(moments) is not dict or set(moments) != expected_ids:
            raise SchemaValidationError("Invalid Adam parameter-state mapping")
        for index, moment in moments.items():
            if type(index) is not int or type(moment) is not dict or moment.keys() != {"step", "exp_avg", "exp_avg_sq"}:
                raise SchemaValidationError("Invalid Adam moment schema")
            step = moment["step"]
            if (not isinstance(step, torch.Tensor) or step.device.type != "cpu"
                    or step.shape != () or step.dtype not in (torch.float32, torch.float64)
                    or not torch.isfinite(step) or step.item() != local):
                raise SchemaValidationError("Invalid stage-local Adam step")
            for key in ("exp_avg", "exp_avg_sq"):
                value = moment[key]
                if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                        or value.dtype != params[index].dtype or value.shape != params[index].shape
                        or not torch.isfinite(value).all()
                        or (key == "exp_avg_sq" and (value < 0).any())):
                    raise SchemaValidationError("Invalid Adam moment tensor")
        # Loading into these detached optimizer/scheduler objects cannot partially
        # restore the destination if deserialization raises.
        optimizer.load_state_dict(copy.deepcopy(incoming))
        scheduler.load_state_dict(copy.deepcopy(state["scheduler"]))
        return optimizer, scheduler

    def receipt(self):
        return S3Receipt(json.dumps(self.bindings, sort_keys=True), self.global_epoch,
            self.updates, (self.best.epoch, self.best.stage, self.best.stage_epoch, self.best.updates), self.boundary)
