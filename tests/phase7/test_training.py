import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from inverse_em.errors import SchemaValidationError
from inverse_em.training import s2
from inverse_em.provenance.s2 import HISTORICAL_IDENTITIES, HISTORICAL_BEST, HISTORICAL_TERMINAL


def same(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a) == len(b)
        for x,y in zip(a,b):
            same(x,y)
    else:
        assert a == b


@pytest.mark.parametrize("direction,reset", [(-np.inf,True),(None,False),(np.inf,False)])
def test_exact_subtraction_boundary(direction,reset):
    reference=.002253249128912805
    boundary=reference-1e-5
    score=boundary if direction is None else np.nextafter(boundary,direction)
    state=s2.MaterialState(reference,12)
    state.observe(float(score))
    assert state.stale == (0 if reset else 13)
    assert state.es_reference == (score if reset else reference)


def test_best_material_divergence_tie_nonfinite():
    best=s2.BestState()
    assert best.metric == float("inf") and best.epoch == best.update == 0
    material=s2.MaterialState()
    for epoch,score in enumerate((.1,.1-5e-6),1):
        assert best.consider(score,epoch,epoch)
        material.observe(score)
    assert best.epoch == 2 and material.stale == 1 and material.es_reference == .1
    assert not best.consider(best.metric,3,3)
    for value in (float("nan"),float("inf")):
        with pytest.raises(FloatingPointError):
            best.consider(value,3,3)
        with pytest.raises(FloatingPointError):
            material.observe(value)


def test_exact_patience():
    state=s2.MaterialState()
    state.observe(1.)
    assert state.stale == 0
    for _ in range(199):
        state.observe(1.)
        assert not state.should_stop
    state.observe(1.)
    assert state.stale == 200 and state.should_stop


def history_rows():
    rows=[]
    for line in Path(__file__).with_name("history_scalars.jsonl").read_text().splitlines():
        e,u,s,b,be,ref,stale=json.loads(line)
        rows.append(dict(epoch=e,updates=u,validation={"canonical_cartesian_rmse":s},
                         best=b,best_epoch=be,es_reference=ref,epochs_without_material_improvement=stale))
    return rows


def test_historical_scalar_replay():
    rows=history_rows()
    assert s2.replay_history(rows) == dict(rows=392,best_epoch=199,best_update=108853,
        last_material_reset=192,first_stop=392,stale=200,terminal_updates=214424)
    rows[198]["best_epoch"]=198
    with pytest.raises(SchemaValidationError):
        s2.replay_history(rows)
    with pytest.raises(PermissionError):
        s2.replay_history([{}]*601)


def test_optimizer_ownership_and_optional_flag_equivalence(trainer_factory):
    t=trainer_factory()
    group=t.optimizer.param_groups[0]
    assert len(t.optimizer.param_groups) == 1 and type(t.optimizer) is torch.optim.Adam
    assert (group["lr"],group["betas"],group["eps"],group["weight_decay"]) == (.0005,(.9,.999),1e-8,1e-5)
    assert sum(p.numel() for p in group["params"]) == 399437
    assert set(map(id,group["params"])) == set(map(id,t.model.parameters())) | {id(t.fc.log_variances)}
    a=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float64))
    b=torch.nn.Parameter(a.detach().clone())
    historical=torch.optim.Adam([a],lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
    reference=torch.optim.Adam([b],lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=1e-5,
        amsgrad=False,maximize=False,foreach=False,fused=False,capturable=False,differentiable=False)
    for _ in range(2):
        a.grad=torch.tensor([.2,.3],dtype=torch.float64)
        b.grad=a.grad.clone()
        historical.step()
        reference.step()
    assert torch.equal(a,b)


def test_scheduler_lr_fixtures_without_optimization(trainer_factory):
    t=trainer_factory()
    assert (t.scheduler.T_0,t.scheduler.T_mult,t.scheduler.eta_min) == (200,2,1e-6)
    expected={1:.0005,199:1.1231131887499658e-06,200:1.0307801958256833e-06,
              201:.0005,392:.00026812143298982614}
    for epoch in range(1,393):
        if epoch in expected:
            assert t.optimizer.param_groups[0]["lr"] == expected[epoch]
        t.scheduler.step(epoch)
    assert t.updates == 0


def test_event_order_and_snapshot_boundaries(trainer_factory,monkeypatch):
    t=trainer_factory()
    events=[]
    val=s2.validate_clean
    observe=t.material.observe
    consider=t.best.consider
    step=t.scheduler.step
    monkeypatch.setattr(s2,"validate_clean",lambda *a:(events.append("validation"),val(*a))[1])
    monkeypatch.setattr(t.best,"consider",lambda *a:(events.append("best"),consider(*a))[1])
    monkeypatch.setattr(t.material,"observe",lambda *a:(events.append("material"),observe(*a))[1])
    def schedule(epoch):
        assert len(t.history) == epoch
        events.extend(["history","scheduler"])
        return step(epoch)
    monkeypatch.setattr(t.scheduler,"step",schedule)
    # Instance spies are not scientific state fields; remove them before capture.
    original_capture=t._capture
    def capture(history):
        result=original_capture(history)
        result["best"].pop("consider",None)
        result["material"].pop("observe",None)
        return result
    monkeypatch.setattr(t,"_capture",capture)
    t.run(1,on_best=lambda state:events.append("best_copy"))
    assert events == ["validation","best","material","best_copy","history","scheduler"]
    assert t.best_checkpoint["scheduler"]["last_epoch"] == 0
    assert t.best_checkpoint["boundary"] == "VALIDATION_AND_ACCOUNTING_COMPLETE_PRE_SCHEDULER"
    assert t.state_dict()["scheduler"]["last_epoch"] == 1
    assert t.terminal_checkpoint["scheduler"]["last_epoch"] == 1
    with pytest.raises(SchemaValidationError):
        trainer_factory().load_state_dict(t.best_checkpoint)


def test_continuation_exact_and_terminal_nonalias(trainer_factory,monkeypatch):
    # Synthetic scores exercise selection only; ordinary tiny forwards/backwards
    # still execute. A test-only short patience probes loop exit within four
    # updates; the separate 392-row replay tests the frozen patience of 200.
    monkeypatch.setattr(s2.MaterialState,"should_stop",property(lambda self:self.stale>=2))
    scores=[.1,.1-5e-6,.11]
    def install_scores():
        sequence=iter(scores)
        monkeypatch.setattr(s2,"validate_clean",lambda *args:{"cartesian":{"rmse":next(sequence)}})
    full=trainer_factory()
    install_scores()
    full.run(3)
    split=trainer_factory()
    install_scores()
    split.run(1)
    saved=split.state_dict()
    resumed=trainer_factory()
    before=torch.get_rng_state().clone()
    resumed.load_state_dict(saved)
    assert torch.equal(before,torch.get_rng_state())
    resumed.run(2)
    for key in ("model","kendall","optimizer","scheduler","history","best","material",
                "best_checkpoint","termination_reason"):
        same(full.state_dict()[key],resumed.state_dict()[key])
    assert full.termination_reason == "early_stopping"
    assert full.best.epoch == 2 and full.material.stale == 2
    frozen=copy.deepcopy(full.best_checkpoint["model"])
    with torch.no_grad():
        next(full.model.parameters()).add_(1.)
    same(frozen,full.best_checkpoint["model"])
    assert full.terminal_checkpoint["completed_epoch"] == 3
    assert full.best_checkpoint["completed_epoch"] == 2
    with pytest.raises(PermissionError):
        full.run(1)


@pytest.mark.parametrize("corruption", ["epoch","bindings","material","role","history","best"])
def test_restore_rejects_before_model_mutation(trainer_factory,corruption):
    t=trainer_factory()
    t.run(1)
    state=t.state_dict()
    if corruption == "epoch": state["completed_epoch"]=600
    elif corruption == "bindings": state["bindings"]["configuration"]="wrong"
    elif corruption == "material": state["material"]["stale"]=199
    elif corruption == "role": state["checkpoint_role"]="TERMINAL"
    elif corruption == "history": state["history"][0]["stale"]=5
    else: state["best_checkpoint"]=None
    fresh=trainer_factory()
    before=copy.deepcopy(fresh.model.state_dict())
    with pytest.raises(SchemaValidationError):
        fresh.load_state_dict(state)
    same(before,fresh.model.state_dict())


def test_cumulative_bound_and_receipt(trainer_factory):
    t=trainer_factory()
    t.run(2)
    fresh=trainer_factory()
    fresh.load_state_dict(t.state_dict())
    fresh.run(2)
    assert fresh.completed_epoch == fresh.updates == 4
    with pytest.raises(PermissionError):
        fresh.run(1)
    assert len(fresh.receipt().sha256) == 64
    assert HISTORICAL_IDENTITIES["initialization"] is None
    assert HISTORICAL_IDENTITIES["protocol_manifest"] is None
    assert HISTORICAL_BEST["epoch"] == 199 and HISTORICAL_TERMINAL["epoch"] == 392
