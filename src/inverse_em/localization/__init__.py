"""Bounded common localization infrastructure; no task training or protected evaluation."""

from .decoding import RawLocalizerOutput, ScientificPrediction, decode_angle, decode_radius, decode_scientific
from .field_consistency import FIELD_CHANNELS, FieldConsistency, compose_fc_objective
from .losses import CanonicalTargets, LocalizationComponents, localization_components, reduce_outer_s3, reduce_s1, reduce_s2_canonical
from .masking import active_slot_mask, scientific_view
from .matching import DiagnosticAssignment, diagnostic_assignment
from .metrics import canonical_errors, canonical_metrics, summarize, wrapped_angular_error
from .model import CircularLocalizer, make_localizer, parameter_count

__all__ = [
    "CanonicalTargets", "CircularLocalizer", "DiagnosticAssignment", "FIELD_CHANNELS", "FieldConsistency",
    "LocalizationComponents", "RawLocalizerOutput", "ScientificPrediction", "active_slot_mask", "canonical_errors",
    "canonical_metrics", "compose_fc_objective", "decode_angle", "decode_radius", "decode_scientific",
    "diagnostic_assignment", "localization_components", "make_localizer", "parameter_count", "reduce_outer_s3",
    "reduce_s1", "reduce_s2_canonical", "scientific_view", "summarize", "wrapped_angular_error",
]
