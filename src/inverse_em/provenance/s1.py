"""Descriptive S1 evidence, never authorization or checkpoint migration."""
from dataclasses import dataclass
from types import MappingProxyType
from inverse_em.provenance.canonical import canonical_sha256

HISTORICAL_IDENTITIES = MappingProxyType({
    "train_npz": "b04b5e80e83d338ee42b165e58244c90eb233a8e2cb4ce9ead96a2aed9029b54",
    "validation_npz": "3c0439bc9559b776db5ea61bfe55bbe0d3d5158734a454e3332eb5209f10371c",
    "normalizer_file": "c4868c3896cb4b9921dd5e69fb350471f86b2a756975567c2a9cb8aa5572d45b",
    "initialization": "55b6b9d0a405471d3b99425fc285b8adbfea64a6cbbef1e5a0eba7180fa0815a",
    "checkpoint_file": "d5a961fb83811cd09fa3b9b9993f1adbd10266ea10f83365993172ea7578f42d",
    "config_file": "4e8362fe24380a85964873bbcdbc452c920f0047a34e7f68610699dd797335a1",
    "protocol_manifest": "32019f69f075c9ee53a849fc913ed5062f2cbec356c6c782ea897d94cedc667e",
    "revision": "32693f0ba6b5417cbe7e4f815c8b0cd58a0969fd",
})
HISTORICAL_VALIDATION = MappingProxyType({
    "epoch": 200, "updates": 109400, "cartesian_mean": 0.0001860996458251261,
    "cartesian_median": 0.0001682870583430841, "cartesian_rmse": 0.00022168686968818813,
    "cartesian_p95": 0.0003725950979105944, "cartesian_p99": 0.000521099717607437,
    "cartesian_max": 0.0030421552102153, "radial_rmse": 0.00017324376089065025,
    "angular_rmse_radians": 0.00036055005839547716, "angular_rmse_degrees": 0.020657996649256216,
})


@dataclass(frozen=True)
class S1Receipt:
    configuration_sha256: str
    population_identities: tuple
    normalizer_identity: str
    phase5_identity: str
    surrogate_identities: tuple
    rng_convention: str
    updates: int
    best_epoch: int | None
    best_update: int | None
    terminal_epoch: int
    runtime_versions: tuple
    scope: str = "bounded_mechanism_fixture_only"

    @property
    def sha256(self):
        return canonical_sha256(self)
