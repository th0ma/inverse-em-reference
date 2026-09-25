"""Descriptive decisions are evidence-bound, not production execution authority."""
from dataclasses import dataclass
import json

from .contracts import Context, Domain, Identity, Outcome, Use, checked, identity, require


@dataclass(frozen=True)
class Decision:
    artifact: Identity
    context_sha256: str
    use: Use
    outcome: Outcome
    evidence: tuple
    reason: str


def classify(context, artifact, *, evidence=(), established=False, neutral_migration=False,
             normalizer_unrecoverable=False, trained_state_unrecoverable=False,
             contract_violation=False, approved_recovery_exhausted=False, continuation_only_missing=False):
    """Synthetic decision-rule exercise; cannot issue an accepted state.

    Facts are supplied by an external review, not inferred from file absence/metrics.
    Only integration.validate can establish direct compatibility.
    """
    require(type(context) is Context, "Context required")
    context.__post_init__()
    require(type(artifact) is Identity, "Artifact identity required")
    artifact.__post_init__()
    require(artifact.domain in (Domain.SOURCE, Domain.NATIVE), "Source/native artifact identity required")
    require(type(evidence) is tuple, "Immutable evidence required")
    for item in evidence:
        checked(item, Domain.NATIVE)
    flags = (established, neutral_migration, normalizer_unrecoverable, trained_state_unrecoverable,
             contract_violation, approved_recovery_exhausted, continuation_only_missing)
    require(all(type(v) is bool for v in flags), "Explicit evidence flags")
    if not established or not evidence or continuation_only_missing:
        outcome, reason = Outcome.INSUFFICIENT_EVIDENCE, "Missing evidence or continuation state is not a retraining finding"
    elif neutral_migration:
        require(not any((normalizer_unrecoverable, trained_state_unrecoverable, contract_violation)), "Conflicting migration evidence")
        outcome, reason = Outcome.CONTROLLED_MIGRATION_REQUIRED, "Neutral transformation requires exact destination validation"
    elif normalizer_unrecoverable:
        outcome, reason = Outcome.REGENERATION_REQUIRED, "Normalizer reconstruction unavailable; no fitting authorized"
    elif (trained_state_unrecoverable or contract_violation) and approved_recovery_exhausted:
        outcome, reason = Outcome.RETRAINING_REQUIRED, "Required trained state unavailable/incompatible and approved recovery exhausted"
    else:
        outcome, reason = Outcome.INSUFFICIENT_EVIDENCE, "Direct compatibility requires integration validation"
    return Decision(artifact, context.sha256, context.use, outcome, evidence, reason)


@dataclass(frozen=True)
class MigrationPlan:
    operation: str
    pairs: tuple
    source_semantics: tuple
    destination_semantics: tuple

    def __post_init__(self):
        require(self.operation in ("container_conversion", "metadata_key_rename", "state_key_remap"), "Unapproved scientific transformation")
        require(all(type(x) is tuple for x in (self.pairs, self.source_semantics, self.destination_semantics)), "Immutable mapping")
        require(all(type(p) is tuple and len(p) == 2 and all(type(v) is str and v for v in p)
                    for p in self.source_semantics + self.destination_semantics), "Immutable semantic bindings")
        require(all(type(p) is tuple and len(p) == 2 and all(type(v) is str and v for v in p) for p in self.pairs), "Mapping pairs")
        src, dst = tuple(p[0] for p in self.pairs), tuple(p[1] for p in self.pairs)
        require(len(set(src)) == len(src) and len(set(dst)) == len(dst), "Mapping collision")
        require(src == tuple(k for k, _ in self.source_semantics) and dst == tuple(k for k, _ in self.destination_semantics), "Mapping omissions/order")
        require(tuple(v for _, v in self.source_semantics) == tuple(v for _, v in self.destination_semantics), "Changed semantic association")

    @property
    def identity(self):
        self.__post_init__()
        return identity(Domain.NATIVE, "descriptive-migration-plan/1", self)


def validate_migration(plan, before, after):
    """Validate a manually transformed tiny fixture; never execute a migration."""
    require(type(plan) is MigrationPlan, "Typed migration plan required")
    plan.__post_init__()
    require(type(before) is dict and type(after) is dict, "Synthetic mappings only")
    require(set(before) == {a for a, _ in plan.pairs} and set(after) == {b for _, b in plan.pairs}, "Mapping omissions")
    # Compare representations as well as values (e.g. preserve signed zero).
    require(all(json.dumps(before[a], sort_keys=True, allow_nan=False) ==
                json.dumps(after[b], sort_keys=True, allow_nan=False) for a, b in plan.pairs), "Changed values")
    return plan.identity
