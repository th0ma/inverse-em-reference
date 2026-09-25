import copy
import json

import pytest
from phase11_accounting import ROOT


def rows(phase):
    values = [json.loads(line) for line in (ROOT / f"tests/phase{phase}/history_scalars.jsonl").read_text().splitlines()]
    if phase == 7:
        return [{"epoch": e, "updates": u, "validation": {"canonical_cartesian_rmse": score},
                 "best": best, "best_epoch": be, "es_reference": ref,
                 "epochs_without_material_improvement": stale}
                for e, u, score, best, be, ref, stale in values]
    return values


def test_s2_scalar_evidence_not_training():
    from inverse_em.training.s2 import replay_history
    from inverse_em.provenance.s2 import HISTORICAL_BEST, HISTORICAL_TERMINAL
    data = rows(7)
    replay = replay_history(data)
    assert replay["rows"] == replay["first_stop"] == HISTORICAL_TERMINAL["epoch"] == 392
    assert replay["best_epoch"] == HISTORICAL_BEST["epoch"] == 199
    assert replay["best_update"] == 108853 and replay["terminal_updates"] == 214424


def test_s3_scalar_evidence_not_training():
    from inverse_em.provenance.s3 import replay_history
    replay = replay_history(rows(8))
    assert len(replay["events"]) == 78
    assert replay["best"] == (394, 8, 44, 215518, 0.01900846562333382)
    assert replay["terminal"] == (400, 218800, 0.020064019380502007)


@pytest.mark.parametrize("phase", [7, 8])
@pytest.mark.parametrize("change", ["updates", "score", "best"])
def test_historical_replay_detects_detached_weakening(phase, change):
    from inverse_em.training.s2 import replay_history as s2
    from inverse_em.provenance.s3 import replay_history as s3
    data = copy.deepcopy(rows(phase))
    if change == "updates":
        data[1]["updates"] += 1
    elif phase == 7:
        if change == "score":
            data[0]["validation"]["canonical_cartesian_rmse"] = float("nan")
        else:
            data[-1]["best_epoch"] += 1
    else:
        data[0]["validation_rmse" if change == "score" else "best_rmse"] = float("nan")
    with pytest.raises((ValueError, FloatingPointError)):
        (s2 if phase == 7 else s3)(data)


def test_best_and_material_are_not_interchangeable():
    from inverse_em.training.s2 import BestState, MaterialState
    best, material = BestState(), MaterialState()
    for epoch, score in enumerate((.1, .1 - 5e-6), 1):
        assert best.consider(score, epoch, epoch)
        material.observe(score)
    assert best.epoch == 2 and material.stale == 1
    assert not best.consider(best.metric, 3, 3)


def test_scalar_continuation_does_not_claim_optimizer_resume():
    from inverse_em.training.s2 import BestState, MaterialState
    scores = [.2, .1, .1, .1001]
    def step(best, material, start, values):
        for epoch, score in enumerate(values, start):
            best.consider(score, epoch, epoch)
            material.observe(score)
    a, b = BestState(), MaterialState()
    step(a, b, 1, scores)
    c, d = BestState(), MaterialState()
    step(c, d, 1, scores[:2])
    c, d = copy.deepcopy(c), copy.deepcopy(d)
    step(c, d, 3, scores[2:])
    assert vars(a) == vars(c) and vars(b) == vars(d)
