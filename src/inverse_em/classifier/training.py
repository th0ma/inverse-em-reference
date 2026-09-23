from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict
import math
import torch
from torch.nn import functional as F

from inverse_em.config.classifier import ClassifierScientificConfig
from inverse_em.errors import SchemaValidationError
from inverse_em.noise import TrainingNoiseStream
from inverse_em.normalization import FrozenChannelNormalizer
from inverse_em.observations import ComplexFields
from inverse_em.provenance.canonical import canonical_sha256
from .data import clean_observation_tensor, epoch_minibatches, noisy_observation_tensor
from .metrics import classification_metrics
from .selection import SelectionState


def copy_model_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _plain(value):
    if isinstance(value, dict): return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list): return [_plain(item) for item in value]
    if isinstance(value, tuple): return tuple(_plain(item) for item in value)
    if hasattr(value, "item"): return value.item()
    return value


def capture_noise_state(stream: TrainingNoiseStream, event_count: int):
    state = _plain(copy.deepcopy(stream._rng.bit_generator.state))
    payload = {"augmentation_seed": stream.augmentation_seed, "event_count": int(event_count),
               "bit_generator": "PCG64DXSM", "bit_generator_state": state}
    return {**payload, "sha256": canonical_sha256(payload)}


def restore_noise_state(receipt):
    required = {"augmentation_seed", "event_count", "bit_generator", "bit_generator_state", "sha256"}
    if required - receipt.keys() or receipt["bit_generator"] != "PCG64DXSM":
        raise SchemaValidationError("Invalid classifier noise-state receipt")
    payload = {key: receipt[key] for key in ("augmentation_seed", "event_count", "bit_generator", "bit_generator_state")}
    if canonical_sha256(payload) != receipt["sha256"]:
        raise SchemaValidationError("Classifier noise-state receipt digest mismatch")
    stream = TrainingNoiseStream(int(receipt["augmentation_seed"]))
    stream._rng.bit_generator.state = copy.deepcopy(receipt["bit_generator_state"])
    return stream, int(receipt["event_count"])


class OwnedDropoutRNG:
    def __init__(self, seed: int):
        caller = torch.get_rng_state().clone()
        try:
            torch.manual_seed(seed); self.state = torch.get_rng_state().clone()
        finally: torch.set_rng_state(caller)

    @contextmanager
    def activate(self):
        caller = torch.get_rng_state().clone()
        try:
            torch.set_rng_state(self.state)
            yield
            self.state = torch.get_rng_state().clone()
        finally: torch.set_rng_state(caller)


class BoundedClassifierTrainer:
    """Mechanism-only Phase-4 trainer, hard-bounded against scientific-scale execution."""
    def __init__(self, model, config: ClassifierScientificConfig = ClassifierScientificConfig()):
        self.model = model; self.config = config
        c = config.training
        self.optimizer = torch.optim.Adam(model.parameters(), lr=c.learning_rate, betas=c.betas, eps=c.epsilon,
                                          weight_decay=c.weight_decay, amsgrad=c.amsgrad)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode=c.scheduler_mode, factor=c.scheduler_factor, patience=c.scheduler_patience,
            threshold=c.scheduler_threshold, threshold_mode=c.scheduler_threshold_mode,
            cooldown=c.scheduler_cooldown, min_lr=c.scheduler_min_lr, eps=c.scheduler_epsilon)
        self.selection = SelectionState(); self.dropout_rng = OwnedDropoutRNG(c.dropout_seed)
        self.noise_stream = TrainingNoiseStream(c.augmentation_seed); self.noise_event_count = 0
        self.history = []; self.updates = 0; self.next_epoch = 1; self.best_model_state = None

    def run(self, train_fields, train_labels, validation_fields, validation_labels,
            normalizer: FrozenChannelNormalizer, epochs: int = 1):
        if type(epochs) is not int or epochs < 1 or self.next_epoch + epochs - 1 > 2:
            raise PermissionError("Bounded Phase-4 implementation permits at most two smoke epochs")
        if len(train_fields) > 100 or len(validation_fields) > 100:
            raise PermissionError("Scientific-scale classifier training is not authorized in Phase-4 implementation")
        for labels, fields in ((train_labels, train_fields), (validation_labels, validation_fields)):
            if (not isinstance(labels, torch.Tensor) or labels.dtype is not torch.int64 or labels.device.type != "cpu"
                    or labels.shape != (len(fields),) or torch.any((labels < 0) | (labels > 4))):
                raise SchemaValidationError("Classifier labels must be CPU int64 values 0..4")
            if any(type(field) is not ComplexFields for field in fields):
                raise SchemaValidationError("Classifier training requires clean ComplexFields inputs")
        validation_tensor = clean_observation_tensor(validation_fields, normalizer)
        for epoch in range(self.next_epoch, self.next_epoch + epochs):
            noisy, _ = noisy_observation_tensor(train_fields, normalizer, self.noise_stream, self.noise_event_count)
            self.noise_event_count += len(train_fields); self.model.train(); loss_sum = 0.0
            for indices in epoch_minibatches(len(train_fields), epoch, self.config.training.minibatch_seed,
                                             self.config.training.batch_size):
                index = torch.from_numpy(indices.astype("int64")); self.optimizer.zero_grad(set_to_none=True)
                with self.dropout_rng.activate():
                    loss = F.cross_entropy(self.model(noisy[index]), train_labels[index], reduction="mean",
                                           label_smoothing=0.0)
                    if not torch.isfinite(loss): raise FloatingPointError("Nonfinite classifier training loss")
                    loss.backward()
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in self.model.parameters()):
                    raise FloatingPointError("Missing or nonfinite classifier gradient")
                self.optimizer.step(); self.updates += 1; loss_sum += float(loss.detach()) * len(indices)
            self.model.eval()
            with torch.no_grad(): metrics = classification_metrics(self.model(validation_tensor), validation_labels)
            improved, stop = self.selection.update(metrics.macro_f1, epoch, self.updates,
                threshold=self.config.training.early_stopping_threshold,
                patience=self.config.training.early_stopping_patience)
            if improved: self.best_model_state = copy_model_state(self.model)
            self.scheduler.step(metrics.macro_f1)
            self.history.append({"epoch": epoch, "updates": self.updates,
                "training_cross_entropy": loss_sum / len(train_fields), "validation": asdict(metrics),
                "learning_rate": self.optimizer.param_groups[0]["lr"], "best": improved})
            self.next_epoch = epoch + 1
            if stop: break
        return tuple(self.history)

    def state_dict(self):
        return {"model_state": copy_model_state(self.model), "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(), "selection_state": asdict(self.selection),
            "dropout_rng_state": self.dropout_rng.state.clone(),
            "noise_state": capture_noise_state(self.noise_stream, self.noise_event_count),
            "history": copy.deepcopy(self.history), "updates": self.updates, "next_epoch": self.next_epoch,
            "best_model_state": copy.deepcopy(self.best_model_state)}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model_state"], strict=True); self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"]); self.selection = SelectionState(**state["selection_state"])
        self.dropout_rng.state = state["dropout_rng_state"].clone(); self.noise_stream, self.noise_event_count = restore_noise_state(state["noise_state"])
        self.history = copy.deepcopy(state["history"]); self.updates = int(state["updates"]); self.next_epoch = int(state["next_epoch"])
        self.best_model_state = copy.deepcopy(state["best_model_state"])
