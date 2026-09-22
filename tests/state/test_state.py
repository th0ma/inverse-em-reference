import pytest

from inverse_em.errors import StateTransitionError
from inverse_em.scientific.state import ScientificState, validate_transition


LEGAL = [
    ("DEVELOPMENT", "TRAINING"), ("TRAINING", "FROZEN"),
    ("FROZEN", "SEALED_CLEAN"), ("SEALED_CLEAN", "ROBUSTNESS"),
    ("ROBUSTNESS", "CLOSED"),
]


@pytest.mark.parametrize("before,after", LEGAL)
def test_legal_transitions(before, after):
    validate_transition(ScientificState(before), ScientificState(after))


@pytest.mark.parametrize("before,after", [("DEVELOPMENT", "FROZEN"), ("FROZEN", "ROBUSTNESS"), ("TRAINING", "DEVELOPMENT")])
def test_illegal_transitions(before, after):
    with pytest.raises(StateTransitionError):
        validate_transition(ScientificState(before), ScientificState(after))


def test_closed_is_terminal():
    with pytest.raises(StateTransitionError, match="terminal"):
        validate_transition(ScientificState.CLOSED, ScientificState.DEVELOPMENT)


@pytest.mark.parametrize("before", list(ScientificState))
@pytest.mark.parametrize("after", list(ScientificState))
def test_all_36_state_pairs(before, after):
    legal = (before.value, after.value) in LEGAL
    if legal:
        validate_transition(before, after)
    else:
        with pytest.raises(StateTransitionError):
            validate_transition(before, after)


@pytest.mark.parametrize("before,after", [(a, b) for a in ScientificState for b in ScientificState])
def test_manifest_enforces_all_state_pairs(valid_manifest_mapping, before, after):
    from inverse_em.provenance.manifest import RunManifest
    value = dict(valid_manifest_mapping)
    value["state_before"], value["state_after"] = before.value, after.value
    if (before.value, after.value) in LEGAL:
        RunManifest.from_mapping(value)
    else:
        with pytest.raises(StateTransitionError):
            RunManifest.from_mapping(value)
