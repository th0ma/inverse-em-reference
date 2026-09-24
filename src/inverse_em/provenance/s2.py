"""Descriptive historical evidence; never execution authority."""
from dataclasses import dataclass
from types import MappingProxyType
from inverse_em.provenance.canonical import canonical_sha256

HISTORICAL_IDENTITIES = MappingProxyType({
    "train_npz": "52e137e6cb703ab114d52ff0bc15fba95b79222fe4f9b6e60397e67cc5ad23c0",
    "validation_npz": "fea8c08c7748346207aac11f16d707265d459b20681e5f3aed8c92554fcfbe52",
    "normalizer_content": "c3af1b30d368376408289b62732821396040bd47e8b9ce623547f5063cdf9f5c",
    "normalizer_file": "61c8cfb63c406ca8fe54713778fb71e89ce2beefb7cc29a30af88af715fdd3fe",
    "config_file": "8fb988956f75ff885be030ecb55c95c8cc02e39c4b30ed56956e39bbc1bd798f",
    "best_checkpoint": "3a76356101f34a815e1b4c19ad30a1b0b5e2892e8fab9b293e9e27c6d4244584",
    "terminal_checkpoint": "ea01b89e5ed1338163ab83961aa755b9fd104ab1f6a1971af23afb53b362c68c",
    "history": "1d4a12e11b09a8444494117a21a53f2a232407e767a247d816f18e528610b842",
    "results": "e17668548145cd08abe06fe415584bf36225fb8e322d6e2521d0d313022a72d3",
    "initialization": None, "protocol_manifest": None,
})
HISTORICAL_BEST = MappingProxyType({
    "epoch": 199, "updates": 108853, "rmse": 0.0022449191022022525,
    "mean": 0.0018307097348069967,
})
HISTORICAL_TERMINAL = MappingProxyType({
    "epoch": 392, "updates": 214424, "rmse": 0.0028001476403387822,
    "mean": 0.002371487613891901,
})


@dataclass(frozen=True)
class S2Receipt:
    bindings: dict
    updates: int
    best_epoch: int | None
    best_update: int | None
    terminal_epoch: int
    termination_reason: str | None
    scope: str = "bounded_mechanism_fixture_only"

    @property
    def sha256(self):
        return canonical_sha256(self)
