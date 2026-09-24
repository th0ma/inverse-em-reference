"""Fixture authorization and independent traversal accounting (not credentials)."""
from dataclasses import dataclass, asdict
from inverse_em.evaluation.contracts import Bindings, label, digest
from inverse_em.provenance.canonical import canonical_sha256
from inverse_em.scientific.state import ScientificState, validate_transition


@dataclass(frozen=True)
class FixtureAuthorization:
    identity: str
    bindings: Bindings
    before: ScientificState
    after: ScientificState
    predecessor: str
    issuer: str

    def __post_init__(self):
        label(self.identity)
        label(self.issuer)
        digest(self.predecessor)
        if type(self.bindings) is not Bindings:
            raise TypeError("Fixture bindings required")
        validate_transition(self.before, self.after)
        if self.after not in (ScientificState.SEALED_CLEAN, ScientificState.ROBUSTNESS):
            raise PermissionError("Only simulated evaluation transitions")

    @property
    def sha256(self):
        return canonical_sha256(self)


@dataclass
class Accounting:
    attempts_started: int = 1
    forward_calls: int = 0
    rows_processed: int = 0
    inference_complete_traversals: int = 0
    verified_primary_bundles: int = 0
    officially_committed_evaluations: int = 0
    exact_execution_count_known: bool = True

    def snapshot(self):
        return asdict(self)
