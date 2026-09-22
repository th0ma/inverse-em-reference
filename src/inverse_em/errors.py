class InverseEMError(Exception):
    """Base error for explicit reference-repository failures."""


class SchemaValidationError(InverseEMError, ValueError):
    """A typed schema or canonical value is malformed."""


class StateTransitionError(InverseEMError, ValueError):
    """A requested scientific lifecycle transition is illegal."""


class PhaseNotImplementedError(InverseEMError, RuntimeError):
    """A command belongs to a later, unauthorized phase."""

