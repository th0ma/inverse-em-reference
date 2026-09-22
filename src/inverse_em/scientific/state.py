from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from inverse_em.errors import StateTransitionError


class ScientificState(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    TRAINING = "TRAINING"
    FROZEN = "FROZEN"
    SEALED_CLEAN = "SEALED_CLEAN"
    ROBUSTNESS = "ROBUSTNESS"
    CLOSED = "CLOSED"


_SUCCESSOR = {
    ScientificState.DEVELOPMENT: ScientificState.TRAINING,
    ScientificState.TRAINING: ScientificState.FROZEN,
    ScientificState.FROZEN: ScientificState.SEALED_CLEAN,
    ScientificState.SEALED_CLEAN: ScientificState.ROBUSTNESS,
    ScientificState.ROBUSTNESS: ScientificState.CLOSED,
}


def validate_transition(before: ScientificState, after: ScientificState) -> None:
    expected = _SUCCESSOR.get(before)
    if expected is None:
        raise StateTransitionError(f"{before.value} is terminal")
    if after is not expected:
        raise StateTransitionError(f"Illegal transition {before.value} -> {after.value}; expected {expected.value}")


@dataclass(frozen=True)
class StateTransition:
    study_id: str
    before: ScientificState
    after: ScientificState

    def __post_init__(self) -> None:
        if not self.study_id.strip():
            raise StateTransitionError("study_id must be non-empty")
        validate_transition(self.before, self.after)

