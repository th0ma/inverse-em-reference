from .roles import ScientificRole
from .state import ScientificState, StateTransition, validate_transition

__all__ = ["ScientificRole", "ScientificState", "StateTransition", "validate_transition", "AuthorizationReceipt"]


def __getattr__(name: str):
    if name == "AuthorizationReceipt":
        from .authorization import AuthorizationReceipt
        return AuthorizationReceipt
    raise AttributeError(name)
