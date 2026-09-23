from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
import torch
from inverse_em.config.surrogate import MODEL_SEEDS,SurrogateScientificConfig
from inverse_em.provenance.canonical import file_sha256


REQUIRED=frozenset({"schema_version","artifact_type","field","representation","architecture","activation","dtype","device","seed","epoch","update_count","best_validation_mse","parameter_count","training_config_sha256","dataset_identity","split_identity","physics_identity","repository_revision","software","model_state","timestamp"})


def _validate(payload,expected=None):
    missing=REQUIRED-payload.keys()
    if missing:raise ValueError(f"Missing checkpoint fields: {sorted(missing)}")
    frozen=SurrogateScientificConfig()
    if payload["field"] not in frozen.fields or payload["seed"] not in MODEL_SEEDS or payload["representation"]!=frozen.representation or payload["architecture"]!=frozen.architecture or payload["activation"]!=frozen.activation or payload["dtype"]!=frozen.dtype or payload["device"]!=frozen.device or payload["parameter_count"]!=50306 or payload["training_config_sha256"]!=frozen.training.sha256 or payload["dataset_identity"]!="historical-surrogate-pcg64-20260915" or payload["split_identity"]!="sha256-sort-20260916" or payload["physics_identity"]!={"revision":frozen.physics_revision,"source_sha256":frozen.physics_source_sha256}:raise ValueError("Checkpoint scientific identity mismatch")
    if sum(v.numel() for v in payload["model_state"].values())!=50306:raise ValueError("Checkpoint state parameter count mismatch")
    for key,value in (expected or {}).items():
        if payload.get(key)!=value:raise ValueError(f"Checkpoint {key} mismatch")


def save_checkpoint(path,model,metadata,*,optimizer=None,scheduler=None,early_stopping=None,history=None,loader_generator=None):
    p=Path(path)
    if p.exists():raise FileExistsError(p)
    payload={**metadata,"software":{k:str(v) for k,v in metadata["software"].items()},"schema_version":"phase3-surrogate-checkpoint/1.0","model_state":copy_state(model.state_dict()),"timestamp":datetime.now(timezone.utc).isoformat()}
    if payload.get("artifact_type")=="resumable":
        if any(x is None for x in (optimizer,scheduler,early_stopping,history,loader_generator)):raise ValueError("Incomplete resumable checkpoint state")
        payload.update(optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict(),early_stopping_state={"best":early_stopping.best,"bad_epochs":early_stopping.bad_epochs,"patience":early_stopping.patience},history=history,loader_generator_state=loader_generator.get_state(),torch_rng_state=torch.get_rng_state())
    _validate(payload);p.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,p);return file_sha256(p)


def copy_state(state):return {k:v.detach().cpu().clone() for k,v in state.items()}


def load_checkpoint(path,model,*,expected=None):
    payload=torch.load(Path(path),map_location="cpu",weights_only=True);_validate(payload,expected);model.load_state_dict(payload["model_state"],strict=True);return payload


def load_resumable_checkpoint(path,model,optimizer,scheduler,early_stopping,loader_generator,*,expected=None):
    payload=load_checkpoint(path,model,expected=expected)
    if payload["artifact_type"]!="resumable":raise ValueError("Checkpoint is not resumable")
    required={"optimizer_state","scheduler_state","early_stopping_state","history","loader_generator_state","torch_rng_state"}
    if required-payload.keys():raise ValueError("Incomplete resumable checkpoint")
    optimizer.load_state_dict(payload["optimizer_state"]);scheduler.load_state_dict(payload["scheduler_state"])
    state=payload["early_stopping_state"];early_stopping.best=state["best"];early_stopping.bad_epochs=state["bad_epochs"]
    if early_stopping.patience!=state["patience"]:raise ValueError("Early-stopping patience mismatch")
    loader_generator.set_state(payload["loader_generator_state"]);torch.set_rng_state(payload["torch_rng_state"])
    return payload
