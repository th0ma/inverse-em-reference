"""Immutable upstream PhysicsTM snapshot and its distribution identity."""

UPSTREAM_REPOSITORY = "https://github.com/th0ma/inverse-source-em"
UPSTREAM_COMMIT = "a079c3899bd33f636ab9c8b1571d8e919c839335"
UPSTREAM_SOURCE_PATH = "src/inverse_source_em/physics/physics_tm.py"
UPSTREAM_SOURCE_SHA256 = "b0d8200ef30d05720f0b697822dc0e037935243beea9084edf410c9725c32005"

from .physics_tm import PhysicsTM

__all__ = ["PhysicsTM", "UPSTREAM_REPOSITORY", "UPSTREAM_COMMIT", "UPSTREAM_SOURCE_PATH", "UPSTREAM_SOURCE_SHA256"]
