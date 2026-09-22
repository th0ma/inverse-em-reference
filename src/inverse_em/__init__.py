"""Phase-0 infrastructure for the inverse-EM reference repository.

Importing this package performs no RNG initialization and no filesystem writes.
"""

from ._version import __version__

__all__ = ["__version__"]
