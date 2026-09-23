from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
import torch

from inverse_em.config.classifier import ClassifierScientificConfig
from inverse_em.provenance.canonical import file_sha256


REQUIRED_METADATA = frozenset({"repository_revision", "normalizer_sha256", "training_population_identity",
                               "validation_population_identity", "physics_identity", "software"})
RNG_CONTRACT = "phase2-sequential-noise+phase4-keyed-order+owned-torch-dropout-v1"


@dataclass(frozen=True)
class ClassifierCheckpointContext:
    normalizer_sha256: str
    physics_identity: str
    training_population_identity: str
    validation_population_identity: str
    rng_contract: str = RNG_CONTRACT
    torch_version: str = str(torch.__version__)

    def __post_init__(self):
        if not all(type(value) is str and value for value in (
            self.normalizer_sha256, self.physics_identity, self.training_population_identity,
            self.validation_population_identity, self.rng_contract, self.torch_version)):
            raise ValueError("Checkpoint compatibility context requires nonempty string identities")
        if not self.training_population_identity.startswith("TRAINING:"):
            raise ValueError("Training population identity must declare TRAINING role")
        if not self.validation_population_identity.startswith("VALIDATION:"):
            raise ValueError("Validation population identity must declare VALIDATION role")
        if self.rng_contract != RNG_CONTRACT:
            raise ValueError("Classifier RNG-contract identity mismatch")


def _base(metadata, artifact_type):
    missing = REQUIRED_METADATA - metadata.keys()
    if missing: raise ValueError(f"Missing classifier checkpoint metadata: {sorted(missing)}")
    ClassifierCheckpointContext(
        normalizer_sha256=metadata["normalizer_sha256"], physics_identity=metadata["physics_identity"],
        training_population_identity=metadata["training_population_identity"],
        validation_population_identity=metadata["validation_population_identity"],
        torch_version=str(metadata["software"].get("torch", "")),
    )
    normalized = dict(metadata); normalized["software"] = {str(key): str(value) for key, value in metadata["software"].items()}
    return {"schema_version": "phase4-classifier-checkpoint/1.0", "artifact_type": artifact_type,
            "scientific_config_sha256": ClassifierScientificConfig().sha256, "parameter_count": 99_973,
            "dtype": "float64", "device": "cpu", "class_count": 5, "input_shape": (4, 30),
            "rng_contract": RNG_CONTRACT,
            "selection_criterion": "strict clean-validation macro-F1 increase",
            "timestamp": datetime.now(timezone.utc).isoformat(), **normalized}


def _save(path, payload):
    destination = Path(path)
    if destination.exists(): raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, destination)
    return file_sha256(destination)


def save_best_checkpoint(path, trainer, metadata):
    if trainer.best_model_state is None: raise ValueError("No BEST classifier state is available")
    payload = _base(metadata, "best_validation")
    best_record = next(item for item in trainer.history if item["epoch"] == trainer.selection.best_epoch)
    payload.update(model_state=trainer.best_model_state, epoch=trainer.selection.best_epoch,
                   update_count=trainer.selection.best_update, best_validation_macro_f1=trainer.selection.best_macro_f1,
                   validation_metrics=best_record["validation"])
    return _save(path, payload)


def save_resumable_checkpoint(path, trainer, metadata):
    payload = _base(metadata, "resumable"); payload["training_state"] = trainer.state_dict()
    return _save(path, payload)


def load_classifier_checkpoint(path, context: ClassifierCheckpointContext):
    if not isinstance(context, ClassifierCheckpointContext):
        raise TypeError("A complete ClassifierCheckpointContext is mandatory")
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "phase4-classifier-checkpoint/1.0" or payload.get("parameter_count") != 99_973:
        raise ValueError("Classifier checkpoint identity mismatch")
    if payload.get("scientific_config_sha256") != ClassifierScientificConfig().sha256:
        raise ValueError("Classifier scientific configuration hash mismatch")
    intrinsic = {"dtype": "float64", "device": "cpu", "class_count": 5, "input_shape": (4, 30),
                 "rng_contract": RNG_CONTRACT}
    for key, value in intrinsic.items():
        if payload.get(key) != value: raise ValueError(f"Classifier checkpoint {key} mismatch")
    if payload.get("software", {}).get("torch") != context.torch_version or context.torch_version != str(torch.__version__):
        raise ValueError("Classifier checkpoint PyTorch reference version mismatch")
    contextual = {"normalizer_sha256": context.normalizer_sha256, "physics_identity": context.physics_identity,
                  "training_population_identity": context.training_population_identity,
                  "validation_population_identity": context.validation_population_identity,
                  "rng_contract": context.rng_contract}
    for key, value in contextual.items():
        if payload.get(key) != value: raise ValueError(f"Classifier checkpoint {key} mismatch")
    state = payload.get("model_state") if payload.get("artifact_type") == "best_validation" else payload.get("training_state", {}).get("model_state")
    if not isinstance(state, dict) or sum(value.numel() for value in state.values()) != 99_973:
        raise ValueError("Classifier checkpoint model state mismatch")
    return payload


def restore_resumable_checkpoint(path, trainer, context: ClassifierCheckpointContext):
    payload = load_classifier_checkpoint(path, context)
    if payload.get("artifact_type") != "resumable" or "training_state" not in payload:
        raise ValueError("Checkpoint is not resumable")
    trainer.load_state_dict(payload["training_state"]); return payload
