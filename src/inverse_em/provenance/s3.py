"""Immutable descriptive S3 evidence; no credentials or scientific execution."""
from dataclasses import dataclass
from types import MappingProxyType
import math
from inverse_em.provenance.canonical import canonical_sha256

HISTORICAL_IDENTITIES = MappingProxyType({
    "config": "ad1c13f223785671a678bd71d0731f3cfc57ab4d953d5519cd51621f454833d3",
    "implementation": "6d27e756f12a4728a137ef6662a311cdf2ad72a852fe0fc8d53a18ac3c4c9385",
    "runner": "c8428e5289d8ff8193e4acaff13b1308748d1f0ac8d4ca3e7a56fddb29fa95fc",
    "history": "81aa549b1c1655aacd4c8ccc74b27347fc462235c2869b663038a6ac2b2cba3c",
    "best": "eb10f279a39882542086b6d9dc98d16bde6dfcd1e615494a89945a755e43eae7",
    "terminal": "ea78c0418f991914046d023a771e3ef73a6a603e652d820f9449c7f88b80fc58",
    "normalizer_file": "40681b189999e8851bc32bf63d79a34fc49dd402a89760406083aa106c532d49",
    "normalizer_content": "bbc77adb3b0f97e06a03d857158af563f88538848e06cbeeefaca03abef5b5ef",
    "initialization": None,
})


@dataclass(frozen=True)
class S3Receipt:
    bindings_json: str
    global_epoch: int
    global_updates: int
    best_identity: tuple
    boundary: str
    scope: str = "bounded_fixture_only_no_production_authority"

    @property
    def sha256(self):
        return canonical_sha256(self)


def replay_history(rows):
    """Pure scalar evidence replay, with no numerical/model imports or construction."""
    if not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= 400:
        raise ValueError("Replay requires 1..400 scalar rows")
    best, events = math.inf, []
    for i, row in enumerate(rows, 1):
        stage, local = (i-1)//50+1, (i-1)%50+1
        if (row["global_epoch"], row["stage"], row["stage_epoch"], row["updates"]) != (i, stage, local, i*547):
            raise ValueError("Historical counters inconsistent")
        score = row["validation_rmse"]
        if not math.isfinite(score):
            raise ValueError("Nonfinite historical score")
        improved = score < best
        if improved:
            best = score
            events.append((i, stage, local, i*547, score))
        if row["strict_best"] != improved or row["best_rmse"] != best:
            raise ValueError("Historical global BEST inconsistent")
        lr = 1e-6 + (.0005-1e-6)*(1+math.cos(math.pi*(local-1)/200))/2
        if not math.isclose(row["lr"], lr, rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError("Historical stage-local LR inconsistent")
    return {"events": tuple(events), "best": events[-1], "terminal":
            (len(rows), rows[-1]["updates"], rows[-1]["validation_rmse"])}
