from __future__ import annotations

from dataclasses import dataclass

from inverse_em.provenance.canonical import canonical_sha256


@dataclass(frozen=True)
class LocalizationInfrastructureReceipt:
    configuration_sha256: str
    architecture_identity: str
    electric_surrogate_sha256: str
    magnetic_surrogate_sha256: str
    torch_version: str
    dtype: str = "float64"
    device: str = "cpu"
    correspondence: str = "fixed_slot"
    matching_role: str = "diagnostic_only"

    @property
    def sha256(self) -> str:
        return canonical_sha256(self)
