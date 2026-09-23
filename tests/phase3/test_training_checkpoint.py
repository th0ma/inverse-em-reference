import math
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset
from inverse_em.config.surrogate import MODEL_SEEDS,SurrogateScientificConfig,SurrogateTrainingConfig
from inverse_em.surrogates.checkpoint import load_checkpoint,load_resumable_checkpoint,save_checkpoint
from inverse_em.surrogates.data import LazyPointwiseFieldDataset,generate_historical_sources,historical_split
from inverse_em.surrogates.inference import FrozenSurrogate
from inverse_em.surrogates.model import make_initialized_model
from inverse_em.surrogates.training import BestState,EarlyStopping,component_mse,make_loader,make_optimizer,make_scheduler,train_surrogate,validate_component_mse

def fixture(n=7,partition="TRAINING"):
    sources=generate_historical_sources();split=historical_split(sources.source_id);group=split.training if partition=="TRAINING" else split.validation;ids=np.asarray(group[:n],dtype="U12");lookup={value:i for i,value in enumerate(sources.source_id.tolist())};indices=np.asarray([lookup[x] for x in ids]);rho=sources.rho[indices];phi=sources.phi[indices];target=np.zeros((n,72,2),np.float64);target[...,0]=rho[:,None];target[...,1]=phi[:,None]
    return LazyPointwiseFieldDataset(ids,rho,phi,sources.theta,target,partition=partition,split=split,sources=sources)
def api(seed=MODEL_SEEDS[0]):model,_=make_initialized_model(seed);return FrozenSurrogate(model)

def test_loss_optimizer_and_scheduler_exact_configuration():
    model=api();cfg=SurrogateTrainingConfig();pred=torch.tensor([[1.,2.]],dtype=torch.float64);target=torch.tensor([[0.,0.]],dtype=torch.float64);assert component_mse(pred,target).item()==2.5
    opt=make_optimizer(model,cfg);group=opt.param_groups[0];assert (group["lr"],group["betas"],group["eps"],group["weight_decay"])==(1e-3,(.9,.999),1e-8,0.)
    sch=make_scheduler(opt,cfg);state=sch.state_dict();assert (sch.mode,sch.factor,sch.patience,sch.threshold,sch.threshold_mode,sch.cooldown,sch.min_lrs,sch.eps)==("min",.5,5,1e-4,"rel",0,[0.],1e-8);sch.step(1.);assert state.keys()==sch.state_dict().keys()

def test_strict_best_and_independent_early_stop():
    model=api();best=BestState();assert best.consider(1.,1,2,model) and not best.consider(1.,2,4,model) and best.epoch==1
    stop=EarlyStopping(2);assert not stop.update(1.) and not stop.update(1.) and stop.update(2.) and best.value==stop.best==1.

def test_loader_order_determinism_incomplete_batch_and_update_arithmetic():
    data=fixture(8);a,_=make_loader(data,MODEL_SEEDS[0],training=True);b,_=make_loader(data,MODEL_SEEDS[0],training=True);first_a=next(iter(a));first_b=next(iter(b))
    assert all(torch.equal(x,y) for x,y in zip(first_a,first_b)) and len(a)==math.ceil(len(data)/512)==2 and math.ceil((7000*72)/512)*200==197000

def test_validation_is_record_weighted():
    class Constant(torch.nn.Module):
        def forward(self,r,p,t):return torch.zeros((len(r),2),dtype=torch.float64)
    data=fixture(8,partition="VALIDATION");data.targets[:7]=1.;data.targets[7:]=3.;loader,_=make_loader(data,1,training=False)
    assert validate_component_mse(Constant(),loader)==pytest.approx(float(np.mean(data.targets**2)))

def test_deterministic_bounded_training_and_nonfinite_rejection():
    def run():
        model=api();train,_=make_loader(fixture(),MODEL_SEEDS[0],training=True);val,_=make_loader(fixture(),MODEL_SEEDS[0],training=False);result=train_surrogate(model,train,val,maximum_epochs=2);return model,result
    a,ra=run();b,rb=run();assert ra["history"]==rb["history"] and ra["updates"]==rb["updates"]==2 and all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),b.state_dict().values()))
    class Bad(Dataset):
        def __len__(self):return 1
        def __getitem__(self,i):return *(torch.tensor(.2,dtype=torch.float64) for _ in range(3)),torch.tensor([float("nan"),0.],dtype=torch.float64)
    train,_=make_loader(Bad(),MODEL_SEEDS[0],training=True);val,_=make_loader(fixture(),MODEL_SEEDS[0],training=False)
    with pytest.raises(FloatingPointError):train_surrogate(api(),train,val,maximum_epochs=1)

def metadata(artifact="inference"):
    c=SurrogateScientificConfig();return {"artifact_type":artifact,"field":"E_z","representation":"relative_angle_B","architecture":(3,128,128,128,128,2),"activation":"ReLU","dtype":"float64","device":"cpu","seed":MODEL_SEEDS[0],"epoch":2,"update_count":2,"best_validation_mse":.1,"parameter_count":50306,"training_config_sha256":c.training.sha256,"dataset_identity":"historical-surrogate-pcg64-20260915","split_identity":"sha256-sort-20260916","physics_identity":{"revision":c.physics_revision,"source_sha256":c.physics_source_sha256},"repository_revision":"fixture","software":{"torch":torch.__version__}}

def test_checkpoint_exact_reload_and_identity_validation(tmp_path):
    source=api();point=(torch.tensor(.3,dtype=torch.float64),torch.tensor(.4,dtype=torch.float64),torch.tensor(.5,dtype=torch.float64));before=source(*point).detach().clone();path=tmp_path/"model.pt";digest=save_checkpoint(path,source,metadata());target=api(MODEL_SEEDS[1]);payload=load_checkpoint(path,target,expected={"field":"E_z","representation":"relative_angle_B"})
    assert len(digest)==64 and torch.equal(before,target(*point)) and payload["best_validation_mse"]==.1
    with pytest.raises(ValueError):load_checkpoint(path,api(),expected={"field":"H_phi"})

def test_resumable_checkpoint_restores_training_state(tmp_path):
    model=api();cfg=SurrogateTrainingConfig();optimizer=make_optimizer(model,cfg);scheduler=make_scheduler(optimizer,cfg);stop=EarlyStopping();loader,generator=make_loader(fixture(),MODEL_SEEDS[0],training=True);next(iter(loader));path=tmp_path/"resume.pt"
    save_checkpoint(path,model,metadata("resumable"),optimizer=optimizer,scheduler=scheduler,early_stopping=stop,history=[{"epoch":1}],loader_generator=generator)
    restored=api(MODEL_SEEDS[1]);new_optimizer=make_optimizer(restored,cfg);new_scheduler=make_scheduler(new_optimizer,cfg);new_stop=EarlyStopping();new_generator=torch.Generator()
    payload=load_resumable_checkpoint(path,restored,new_optimizer,new_scheduler,new_stop,new_generator,expected={"field":"E_z"})
    assert payload["history"]==[{"epoch":1}] and torch.equal(generator.get_state(),new_generator.get_state()) and new_stop.best==stop.best
