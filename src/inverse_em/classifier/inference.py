from __future__ import annotations

import torch

from .checkpoint import load_classifier_checkpoint


def load_best_for_inference(path, model, context):
    payload = load_classifier_checkpoint(path, context)
    if payload.get("artifact_type") != "best_validation":
        raise ValueError("Inference requires a BEST_VALIDATION classifier checkpoint")
    model.load_state_dict(payload["model_state"], strict=True); model.eval(); return payload


def predict_logits(model, observations):
    model.eval()
    with torch.no_grad(): return model(observations)


def predict_source_counts(model, observations):
    return torch.argmax(predict_logits(model, observations), dim=1) + 1
