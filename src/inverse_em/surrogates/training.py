from __future__ import annotations
from dataclasses import dataclass
import copy,math
import torch
from torch.utils.data import DataLoader
from inverse_em.config.surrogate import SurrogateTrainingConfig


def require_torch_2_8():
    if torch.__version__.split("+")[0]!="2.8.0": raise RuntimeError(f"Phase-3 scientific training requires PyTorch 2.8.0, found {torch.__version__}")
def component_mse(prediction,target):return torch.mean((prediction-target)**2)
def make_loader(dataset,seed:int,*,training:bool,config:SurrogateTrainingConfig=SurrogateTrainingConfig()):
    generator=torch.Generator(device="cpu");generator.manual_seed(seed)
    return DataLoader(dataset,batch_size=config.batch_size,shuffle=training,generator=generator,num_workers=config.num_workers,drop_last=config.drop_last),generator
def make_optimizer(model,config:SurrogateTrainingConfig=SurrogateTrainingConfig()):
    return torch.optim.Adam(model.parameters(),lr=config.learning_rate,betas=config.betas,eps=config.epsilon,weight_decay=config.weight_decay)
def make_scheduler(optimizer,config:SurrogateTrainingConfig=SurrogateTrainingConfig()):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode=config.scheduler_mode,factor=config.scheduler_factor,patience=config.scheduler_patience,threshold=config.scheduler_threshold,threshold_mode=config.scheduler_threshold_mode,cooldown=config.scheduler_cooldown,min_lr=config.scheduler_min_lr,eps=config.scheduler_epsilon)


@dataclass
class BestState:
    value:float=math.inf;epoch:int=0;updates:int=0;state_dict:dict|None=None
    def consider(self,value:float,epoch:int,updates:int,model)->bool:
        if value<self.value:
            self.value=float(value);self.epoch=epoch;self.updates=updates;self.state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};return True
        return False


@dataclass
class EarlyStopping:
    patience:int=20;best:float=math.inf;bad_epochs:int=0
    def update(self,value:float)->bool:
        if value<self.best:self.best=float(value);self.bad_epochs=0
        else:self.bad_epochs+=1
        return self.bad_epochs>=self.patience


def validate_component_mse(model,loader)->float:
    model.eval();total=0.;records=0
    with torch.no_grad():
        for rho,phi,theta,target in loader:
            loss=component_mse(model(rho,phi,theta),target)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite validation loss")
            total+=float(loss)*len(target);records+=len(target)
    if records==0:raise ValueError("Validation loader is empty")
    return total/records


def train_surrogate(model,train_loader,validation_loader,config:SurrogateTrainingConfig=SurrogateTrainingConfig(),*,maximum_epochs:int|None=None):
    require_torch_2_8();epochs=config.maximum_epochs if maximum_epochs is None else maximum_epochs
    if not 1<=epochs<=config.maximum_epochs:raise ValueError("Invalid bounded epoch count")
    optimizer=make_optimizer(model,config);scheduler=make_scheduler(optimizer,config);best=BestState();stop=EarlyStopping(config.early_stopping_patience);history=[];updates=0
    for epoch in range(1,epochs+1):
        model.train();total=0.;records=0
        for rho,phi,theta,target in train_loader:
            optimizer.zero_grad(set_to_none=True);loss=component_mse(model(rho,phi,theta),target)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError("Missing or nonfinite gradient")
            optimizer.step();updates+=1;total+=float(loss.detach())*len(target);records+=len(target)
        validation=validate_component_mse(model,validation_loader);scheduler.step(validation);improved=best.consider(validation,epoch,updates,model);should_stop=stop.update(validation)
        history.append({"epoch":epoch,"updates":updates,"training_mse":total/records,"validation_mse":validation,"learning_rate":optimizer.param_groups[0]["lr"],"best":improved})
        if should_stop:break
    return {"history":history,"epochs":len(history),"updates":updates,"best":best,"early_stopping":stop,"optimizer":optimizer,"scheduler":scheduler}
