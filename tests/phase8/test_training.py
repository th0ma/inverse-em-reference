import copy
import pytest
import torch
from inverse_em.training.s3 import GlobalBest, optimizer_scheduler, BOUNDARY, BEST_BOUNDARY
from inverse_em.errors import SchemaValidationError


def assert_nested_equal(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_nested_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_nested_equal(x,y)
    else:
        assert a==b


def test_optimizer_ownership_and_lr_fixtures(trainer_factory):
    t=trainer_factory()
    assert len(t.optimizer.param_groups)==1
    params=t.optimizer.param_groups[0]['params']
    assert sum(p.numel() for p in params)==399437 and params[-1] is t.fc.log_variances
    assert t.optimizer.param_groups[0]['weight_decay']==1e-5
    for local,lr in [(1,.0005),(2,.0004999692198041743),(44,.0004452173866309133),(50,.0004296725112919157)]:
        opt,sched=optimizer_scheduler(t.model,t.fc,t.config)
        if local>1:sched.step(local-1)
        assert opt.param_groups[0]['lr']==pytest.approx(lr,rel=1e-12,abs=1e-14)


def test_stage_transition_persistence_and_reset(trainer_factory):
    t=trainer_factory();t.run(2)
    oldopt,oldsched=t.optimizer,t.scheduler
    weights=copy.deepcopy(t.model.state_dict());kendall=t.fc.log_variances.detach().clone()
    best=copy.deepcopy(t.best);history=copy.deepcopy(t.history)
    assert t.transition_pending and t.stage_epoch==2 and len(oldopt.state)>0
    t._advance_stage()
    assert t.stage==2 and t.stage_epoch==0 and not t.transition_pending
    assert t.optimizer is not oldopt and t.scheduler is not oldsched and not t.optimizer.state
    assert t.optimizer.param_groups[0]['lr']==.0005
    assert_nested_equal(t.model.state_dict(),weights)
    assert torch.equal(kendall,t.fc.log_variances) and t.best==best and t.history==history
    assert t.global_epoch==2 and t.updates==2


@pytest.mark.parametrize('split',[1,2,3])
def test_exact_epoch_and_stage_continuation(trainer_factory,split):
    a=trainer_factory();a.run(4)
    b=trainer_factory();b.run(split);state=b.state_dict()
    c=trainer_factory();c.load_state_dict(state);c.run(4-split)
    assert_nested_equal(a.state_dict(),c.state_dict())
    assert c.global_epoch==4 and c.updates==4 and c.stage==2 and c.stage_epoch==2
    with pytest.raises(PermissionError):c.run(1)


def test_global_best_tie_no_early_stopping_and_snapshot_order(trainer_factory,monkeypatch):
    from inverse_em.localization import s3
    t=trainer_factory();scores=iter([.2,.1,.1,.3]);events=[]
    monkeypatch.setattr(s3,'validate_clean',lambda *args:(next(scores),{'fixture':True}))
    def observe(snapshot):
        assert snapshot['boundary']==BEST_BOUNDARY and t.boundary==BEST_BOUNDARY
        assert t.scheduler.last_epoch==t.stage_epoch-1
        events.append(snapshot['metadata']['global_epoch'])
    t.run(4,on_best=observe)
    assert events==[1,2] and t.best.epoch==2 and t.best.stage==1
    assert t.global_epoch==4 and t.history[-1]['validation_rmse']==.3
    assert t.terminal_checkpoint['checkpoint_role']=='TERMINAL'
    assert t.best_checkpoint['metadata']['global_epoch']==2 and t.boundary==BOUNDARY
    assert not hasattr(t,'material') and not t.config.early_stopping


@pytest.mark.parametrize('mutation', ['role','boundary','bindings','overflow','history','pending'])
def test_invalid_resume_rejected_before_model_change(trainer_factory,mutation):
    t=trainer_factory();t.run(2);state=t.state_dict();other=trainer_factory()
    before=copy.deepcopy(other.model.state_dict())
    if mutation=='role':state['checkpoint_role']='BEST'
    if mutation=='boundary':state['boundary']=BEST_BOUNDARY
    if mutation=='bindings':state['bindings']['config']='wrong'
    if mutation=='overflow':state['global_epoch']=5
    if mutation=='history':state['history'][0]['updates']=0
    if mutation=='pending':state['transition_pending']=False
    with pytest.raises((SchemaValidationError,PermissionError)):
        other.load_state_dict(state)
    assert_nested_equal(before,other.model.state_dict())


def test_limits_and_terminal_role(trainer_factory):
    t=trainer_factory()
    with pytest.raises(PermissionError):t.run(400)
    assert t.updates==0
    t.run(4)
    with pytest.raises(PermissionError):t.run(1)
    other=trainer_factory()
    with pytest.raises(SchemaValidationError):other.load_state_dict(t.terminal_checkpoint)
    assert len(t.receipt().sha256)==64
    with pytest.raises(Exception):t.receipt().global_epoch=0


def test_constructor_stage_and_population_limits(trainer_factory):
    from inverse_em.training.s3 import BoundedS3Trainer
    from dataclasses import replace
    t=trainer_factory()
    with pytest.raises(PermissionError):
        BoundedS3Trainer(t.model,t.fc,(*t.stages,t.stages[0]),t.validation,t.normalizer)
    d=t.stages[0]
    with pytest.raises(SchemaValidationError):
        replace(d,fields=d.fields*9)


def test_finite_fixture_state_and_receipt(trainer_factory):
    t=trainer_factory();t.run(4)
    assert all(torch.isfinite(v).all() for v in t.model.state_dict().values())
    assert torch.isfinite(t.fc.log_variances).all()
    r=t.receipt()
    assert r.global_epoch==4 and r.global_updates==4
    assert 'phase2' not in r.scope and r.boundary==BOUNDARY
    assert 'closed_phase2' in t.config.normalizer


def test_global_best_cross_stage_strict():
    b=GlobalBest()
    assert b.observe(.1,1,1,1,1)
    assert not b.observe(.1,3,2,1,3)
    assert b.observe(.09,4,2,2,4)
    assert b.stage==2 and b.epoch==4


@pytest.mark.parametrize('epochs', [0, 1, 2, 3, 4])
def test_valid_optimizer_scheduler_continuation(trainer_factory, epochs):
    source = trainer_factory()
    if epochs:
        source.run(epochs)
    state = source.state_dict()
    destination = trainer_factory()
    destination.load_state_dict(state)
    assert_nested_equal(destination.state_dict(), state)
    assert destination.scheduler.optimizer is destination.optimizer
    assert destination.optimizer.param_groups[0]['params'][-1] is destination.fc.log_variances
    if epochs:
        assert destination.optimizer.param_groups[0]['lr'] != destination.config.lr


@pytest.mark.parametrize('kind', ['optimizer', 'scheduler', 'moments'])
def test_incompatible_optimization_state_rejected_atomically(trainer_factory, kind):
    source = trainer_factory(); source.run(2)
    destination = trainer_factory(); destination.run(1)
    valid = source.state_dict()
    before = destination.state_dict()
    terminal = copy.deepcopy(destination.terminal_checkpoint)
    optimizer, scheduler = destination.optimizer, destination.scheduler
    modes = [m.training for m in destination.model.modules()]
    gradients = [None if p.grad is None else p.grad.clone()
                 for p in [*destination.model.parameters(), destination.fc.log_variances]]
    rng = torch.get_rng_state().clone()
    if kind == 'optimizer':
        changes = [('lr', .123), ('weight_decay', .5), ('betas', (.8, .99)),
            ('eps', 1e-5), ('initial_lr', .01), ('amsgrad', True),
            ('maximize', True), ('foreach', True), ('capturable', True),
            ('differentiable', True), ('fused', True), ('decoupled_weight_decay', True),
            ('params', list(reversed(valid['optimizer']['param_groups'][0]['params']))),
            ('missing_option', None), ('extra_option', True), ('consistent_wrong_lr', .123)]
    elif kind == 'scheduler':
        changes = [('T_0', 100), ('T_mult', 3), ('eta_min', 1e-4),
            ('T_i', 100), ('T_cur', 1), ('last_epoch', 1),
            ('base_lrs', [.01]), ('_last_lr', [.123]), ('_step_count', 9),
            ('_is_initial', True), ('_get_lr_called_within_step', True),
            ('missing_option', None), ('extra_option', True)]
    else:
        changes = [(key, None) for key in ('missing_state', 'step', 'nan', 'negative', 'shape', 'dtype', 'extra')]
    for key, value in changes:
        state = copy.deepcopy(valid)
        if kind == 'optimizer':
            group = state['optimizer']['param_groups'][0]
            if key == 'missing_option': del group['eps']
            elif key == 'consistent_wrong_lr':
                group['lr'] = value
                state['scheduler']['_last_lr'] = [value]
            else: group[key] = value
        elif kind == 'scheduler':
            if key == 'missing_option': del state['scheduler']['T_0']
            else: state['scheduler'][key] = value
        else:
            moment = state['optimizer']['state'][0]
            if key == 'missing_state': del state['optimizer']['state'][0]
            elif key == 'step': moment['step'] = torch.tensor(1.)
            elif key == 'nan': moment['exp_avg'].fill_(float('nan'))
            elif key == 'negative': moment['exp_avg_sq'].fill_(-1.)
            elif key == 'shape': moment['exp_avg'] = torch.zeros(1, dtype=torch.float64)
            elif key == 'dtype': moment['exp_avg'] = moment['exp_avg'].float()
            elif key == 'extra': moment['unexpected'] = 1
        with pytest.raises(SchemaValidationError):
            destination.load_state_dict(state)
        assert_nested_equal(destination.state_dict(), before)
        assert_nested_equal(destination.terminal_checkpoint, terminal)
        assert destination.optimizer is optimizer and destination.scheduler is scheduler
        assert modes == [m.training for m in destination.model.modules()]
        for p, gradient in zip([*destination.model.parameters(), destination.fc.log_variances], gradients):
            if gradient is None: assert p.grad is None
            else: assert torch.equal(p.grad, gradient)
        assert torch.equal(rng, torch.get_rng_state())


def test_original_epoch_one_optimizer_probe_rejected(trainer_factory):
    source = trainer_factory(); source.run(1)
    state = source.state_dict()
    state['optimizer']['param_groups'][0]['lr'] = .123
    state['optimizer']['param_groups'][0]['weight_decay'] = .5
    destination = trainer_factory()
    before = destination.state_dict()
    with pytest.raises(SchemaValidationError): destination.load_state_dict(state)
    assert_nested_equal(destination.state_dict(), before)
