import copy
import numpy as np
import pytest
import torch
from inverse_em.training import s1
from inverse_em.localization import parameter_count


def equal_tree(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal_tree(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal_tree(x,y)
    else: assert a==b


def test_optimizer_ownership_and_defaults(trainer_factory):
    t=trainer_factory(); opt=t.optimizer
    assert type(opt) is torch.optim.Adam and len(opt.param_groups)==1
    group=opt.param_groups[0]
    assert {id(p) for p in group['params']}=={id(p) for p in t.model.parameters()}|{id(t.fc.log_variances)}
    assert sum(p.numel() for p in group['params'])==399437
    assert all(id(p) not in {id(q) for q in group['params']} for f in (t.fc.electric,t.fc.magnetic) for p in f.parameters())
    assert (group['lr'],group['betas'],group['eps'],group['weight_decay'])==(.0005,(.9,.999),1e-8,1e-5)
    assert torch.equal(t.fc.log_variances,torch.zeros(4,dtype=torch.float64))


@pytest.mark.parametrize('epoch,current,next_lr',[(1,.0005,.0004999692198041743),
    (100,.0002544189756692993,.0002505),(199,1.1231131887499658e-6,1.0307801958256833e-6),
    (200,1.0307801958256833e-6,.0005)])
def test_scheduler_reference_and_validation_order(trainer_factory,epoch,current,next_lr,monkeypatch):
    t=trainer_factory(); t.scheduler.step(epoch-1)
    assert t.optimizer.param_groups[0]['lr']==pytest.approx(current,rel=1e-12,abs=1e-14)
    events=[]; original=t.scheduler.step
    def step(value): events.append(('scheduler',value)); original(value)
    monkeypatch.setattr(t.scheduler,'step',step)
    def validate(): events.append(('validation',epoch)); return {'cartesian':{'rmse':1.}}
    monkeypatch.setattr(t.optimizer,'step',lambda: pytest.fail('No optimization in epoch-end seam'))
    s1.validate_then_schedule(t.scheduler,epoch,validate)
    assert events==[('validation',epoch),('scheduler',epoch)]
    assert t.optimizer.param_groups[0]['lr']==pytest.approx(next_lr,rel=1e-12,abs=1e-14)
    if epoch==200: assert t.scheduler.T_cur==0 and t.scheduler.T_i==400


def test_best_strict_tie_and_finite():
    best=s1.BestState()
    assert best.consider(1.,1,1)
    assert not best.consider(1.,2,2) and not best.consider(2.,3,3)
    assert best.epoch==1
    assert best.consider(np.nextafter(1.,0.),4,4) and best.epoch==4
    with pytest.raises(FloatingPointError): best.consider(float('nan'),5,5)


def test_fixed_epochs_order_and_best_terminal_distinction(trainer_factory,monkeypatch):
    t=trainer_factory(); events=[]; count=0
    original=t.optimizer.step
    def step(*args,**kwargs): events.append('update'); return original(*args,**kwargs)
    monkeypatch.setattr(t.optimizer,'step',step)
    def validation(*args):
        nonlocal count
        count+=1; events.append('validation')
        return {'cartesian':{'rmse':1. if count<3 else 2.},'other_metric':-count}
    monkeypatch.setattr(s1,'validate_clean',validation)
    sched=t.scheduler.step
    def schedule(epoch): events.append('scheduler'); sched(epoch)
    monkeypatch.setattr(t.scheduler,'step',schedule)
    def saved(state):
        events.append('best'); assert len(t.history)==0
        assert state['scheduler']['last_epoch']==1
    history=t.run(3,on_best=saved)
    assert events==['update','validation','scheduler','best','update','validation','scheduler','update','validation','scheduler']
    assert len(history)==3 and t.updates==3 and t.completed_epoch==3
    assert t.best.epoch==1 and t.best_checkpoint['completed_epoch']==1
    assert t.terminal_checkpoint['completed_epoch']==3
    assert t.terminal_checkpoint['checkpoint_role']=='TERMINAL'
    assert t.terminal_checkpoint['best_checkpoint']['checkpoint_role']=='BEST'
    assert t.terminal_checkpoint['best_checkpoint']['completed_epoch']==1
    assert all(np.isfinite(r['total']) for r in history)
    assert t.receipt().scope=='bounded_mechanism_fixture_only' and len(t.receipt().sha256)==64


def test_epoch_boundary_continuation_exact(trainer_factory):
    uninterrupted=trainer_factory(); uninterrupted.run(2)
    partial=trainer_factory(); partial.run(1); state=partial.state_dict()
    resumed=trainer_factory(); caller=torch.get_rng_state().clone()
    resumed.load_state_dict(state); resumed.run(1)
    assert torch.equal(caller,torch.get_rng_state())
    for key in ('model','kendall','optimizer','scheduler','history','best','bindings','updates','next_epoch'):
        equal_tree(uninterrupted.state_dict()[key],resumed.state_dict()[key])
    assert state['boundary']==s1.BOUNDARY and state['history_position']==1
    assert {'populations','normalizer','phase5','deployment_surrogates','rng_convention','seed_roles','runtime_versions','order_identity','noise_identity'}<=state['bindings'].keys()
    assert 'torch_rng' in state and 'best_checkpoint' in state


def test_best_and_terminal_snapshots_resume_without_losing_selected_state(trainer_factory,monkeypatch):
    t=trainer_factory()
    monkeypatch.setattr(s1,'validate_clean',lambda *args:{'cartesian':{'rmse':1.}})
    t.run(2)
    from_terminal=trainer_factory(); from_terminal.load_state_dict(t.terminal_checkpoint)
    assert from_terminal.best.epoch==1 and from_terminal.best_checkpoint['completed_epoch']==1
    from_best=trainer_factory(); from_best.load_state_dict(t.best_checkpoint)
    assert from_best.best_checkpoint['completed_epoch']==1
    from_terminal.run(1)
    assert from_terminal.best_checkpoint['completed_epoch']==1 and from_terminal.completed_epoch==3


@pytest.mark.parametrize('corruption',['config','epoch','boundary'])
def test_continuation_mismatch_rejected_before_loading(trainer_factory,corruption):
    t=trainer_factory(); state=t.state_dict()
    if corruption=='config': state['bindings']['configuration']='wrong'
    elif corruption=='epoch': state['next_epoch']=2
    else: state['boundary']='IN_EPOCH'
    with pytest.raises(ValueError): t.load_state_dict(state)


def test_production_training_and_mid_epoch_state_rejected(trainer_factory):
    t=trainer_factory()
    for epochs in (200,5,0,-1):
        with pytest.raises(PermissionError): t.run(epochs)
    assert t.updates==0
    t.boundary='IN_EPOCH'
    with pytest.raises(RuntimeError): t.state_dict()
    with pytest.raises(RuntimeError): t.run(1)
