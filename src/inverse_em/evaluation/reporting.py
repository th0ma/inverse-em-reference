"""Offline metrics only: no observations, checkpoint loaders or inference."""
from dataclasses import asdict
import numpy as np
from inverse_em.evaluation.contracts import PredictionBundle


def bundle_metrics(bundle: PredictionBundle):
    import torch
    from inverse_em.classifier.metrics import classification_metrics
    from inverse_em.localization.decoding import ScientificPrediction
    from inverse_em.localization.losses import CanonicalTargets
    from inverse_em.localization.metrics import canonical_metrics, canonical_errors, summarize
    from inverse_em.localization.matching import diagnostic_assignment
    if type(bundle) is not PredictionBundle:
        raise TypeError("Validated prediction bundle required")
    a = {k: torch.from_numpy(v.copy()) for k, v in bundle.arrays}
    if bundle.bindings.task == "classifier":
        return {"sample_count": len(bundle.ids), **asdict(classification_metrics(a["logits"], a["truth"] - 1))}
    prediction = ScientificPrediction(a["rho"], a["cos_like"], a["sin_like"], a["phi"])
    targets = CanonicalTargets(a["target_rho"], a["target_phi"])
    result = canonical_metrics(prediction, targets)
    result["d_max"] = summarize(canonical_errors(prediction, targets)["cartesian"].max(dim=1).values)
    if bundle.active_count > 1:
        assignment = diagnostic_assignment(prediction, targets)
        from inverse_em.localization.metrics import cartesian_coordinates
        error = torch.linalg.vector_norm(cartesian_coordinates(prediction.rho, prediction.phi) - assignment.matched_target_xy, dim=-1)
        result["diagnostic_matching"] = {"cartesian": summarize(error),
            "assignment_change_count": int(assignment.assignment_changed.sum()),
            "assignment_change_fraction": float(assignment.assignment_changed.to(torch.float64).mean())}
    return result


def _aggregate(values):
    first = values[0]
    if isinstance(first, dict):
        if any(set(v) != set(first) for v in values):
            raise ValueError("Metric schemas differ")
        return {k: _aggregate([v[k] for v in values]) for k in first}
    if isinstance(first, str):
        if any(v != first for v in values):
            raise ValueError("Metric conventions differ")
        return first
    data = np.asarray(values, dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite metrics")
    return {"mean": np.mean(data, axis=0).tolist(), "sample_sd": np.std(data, axis=0, ddof=1).tolist()}


def aggregate_realizations(records):
    """Records are verified per-bundle records, never pooled errors."""
    if len(records) != 10 or any(type(r["realization"]) is not int for r in records) or {r["realization"] for r in records} != set(range(10)):
        raise ValueError("Exactly ten distinct complete realizations required")
    from inverse_em.config.evaluation import EvaluationConfig
    if len({r["snr"] for r in records}) != 1 or any(type(r["snr"]) is not int or r["snr"] not in EvaluationConfig().snrs or r["complete"] is not True for r in records):
        raise ValueError("Incomplete or mixed-SNR records")
    for field in ("bindings", "seed"):
        if len({r[field] for r in records}) != 1:
            raise ValueError("Mixed robustness identities")
    ordered = sorted(records, key=lambda r: r["realization"])
    return {"aggregation": "per_realization_then_mean_sample_sd", "ddof": 1,
            "realizations": ordered, "summary": _aggregate([r["metrics"] for r in ordered])}
