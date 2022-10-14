import itertools

import torch
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler

from .adamp import AdamP
from .sgdp import SGDP
from .radam import RAdam
from utils import HParams
from functional import partition_params
from .lr_scheduler import EmptyScheduler, CosineAnnealingWarmupRestarts, ReduceLROnPlateau


class MergedModel:
    """Use to apply a single optimizer to multiple models.
    Input: List of torch.nn.Module
    Output: An object that has "parameters" and "named_parameters" method
    Usage:
    >> merged_model = MergedModel([model1, model2, ...])
    >> optim = torch.optim.Adam(merged_model.parameters(), lr=0.001)
    or
    >> optim = get_optimizer(merged_model, hp)"""
    def __init__(self, model_list):
        self.model_list = model_list
    
    def parameters(self):
        return itertools.chain.from_iterable([model.parameters() for model in self.model_list])
    
    def named_parameters(self):
        for model in self.model_list:
            for name, param in model.named_parameters():
                yield name, param


def get_optimizer(model: torch.nn.Module, hp: HParams) -> torch.optim.Optimizer:
    optimizer_name = hp.optimizer
    if optimizer_name == "AdamP": 
        optimizer = AdamP
    elif optimizer_name == "SGDP":
        optimizer = SGDP
    elif optimizer_name == "RAdam":
        optimizer = RAdam
    else:
        optimizer = getattr(torch.optim, optimizer_name)
    
    param_names_without_wd = getattr(hp, "params_without_weight_decay", None)
    if param_names_without_wd is not None:
        params_without_wd, params_with_wd = partition_params(model.named_parameters(), param_names_without_wd)
        params = [
            {"params": params_with_wd},
            {"params": params_without_wd, "weight_decay": 0.0}
        ]
    else:
        params = model.parameters()

    return optimizer(params, **hp.optimizer_kwargs)


def get_scheduler(optimizer: optim.Optimizer, hp: HParams) -> _LRScheduler:
    scheduler_name = hp.scheduler
    if scheduler_name == "EmptyScheduler":
        return EmptyScheduler()
    elif scheduler_name == "CosineAnnealingWarmupRestarts":
        return CosineAnnealingWarmupRestarts(optimizer, max_lr=hp.optimizer_kwargs.lr, **hp.scheduler_kwargs)
    elif scheduler_name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, **hp.scheduler_kwargs)
    else:
        return getattr(optim.lr_scheduler, scheduler_name)(optimizer, **hp.scheduler_kwargs)
