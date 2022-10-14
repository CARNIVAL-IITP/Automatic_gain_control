import os, math, sys
from array import array
from fcntl import ioctl
from termios import TIOCGWINSZ
from typing import Optional

import torch
from torch.cuda import amp
import torch.distributed as dist
from tqdm import tqdm
import torch.nn.functional as F

from models.modelwrapper import AudioModelWrapper
from functional import stft, spec_to_mel
from utils import plot_param_and_grad

from .models import DCTCRN
from .losses import si_snr, abs_mse_loss
import numpy as np
from pypesq import pesq


class ModelWrapper(AudioModelWrapper):
    def __init__(self, hps, train=False, rank=0, device='cpu'):
        self.h = hps.data

        super().__init__(hps, train, rank, device)
        
        if train:
            hp = hps.train
            self.lr_reduce = True if hp.scheduler == "ReduceLROnPlateau" else False
            if getattr(hp, "loss_aux", None) == "abs_mse":
                def loss(x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
                    return abs_mse_loss(x, y, hp.power)
                self.loss_aux = loss
                self.lambda_aux: float = hp.lambda_aux
            else:
                self.loss_aux = None
    
    def get_model(self, hps):
        return DCTCRN(**hps.model_kwargs)
    
    @torch.no_grad()
    def inference(self, wav):
        if wav.dim == 3:
            wav = wav.squeeze(1)
        wav_hat = self.model(wav)
        return wav_hat
    
    def train_epoch(self, dataloader):
        self.train()

        max_items = len(dataloader)
        padding = int(math.log10(max_items)) + 1
        
        summary = {"scalars": {}, "hists": {}}
        loss_total = 0.0
        for idx, batch in enumerate(dataloader, start=1):
            self.optim.zero_grad(set_to_none=True)
            clean = batch["clean"].cuda(self.rank, non_blocking=True)
            noisy = batch["noisy"].cuda(self.rank, non_blocking=True)
            
            with amp.autocast(enabled=self.fp16):
                wav_hat = self.model(noisy)
                loss = F.mse_loss(wav_hat, clean)
                loss_total += loss.item()
                '''
                if self.loss_aux is not None:
                    spec_near = self._module.dct(near)
                    loss_aux = self.loss_aux(spec_hat, spec_near)
                    loss_aux_total += loss_aux.item()
                    loss += loss_aux * self.lambda_aux
                '''
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim)
            if idx == len(dataloader) and self.plot_param_and_grad:
                plot_param_and_grad(summary["hists"], self.model)
            self.clip_grad(self.model.parameters())
            self.scaler.step(self.optim)
            self.scaler.update()
            if self.rank == 0:
                print(
                    f"\rEpoch {self.epoch} - Train {idx:{padding}d}/{max_items} ({idx/max_items*100:>4.1f}%)"
                    f"    loss: {loss_total / idx:6.4f}"
                    f"    scale {self.scaler.get_scale():.4f}",
                    sep=' ', end='', flush=True
                )
        if self.rank == 0:
            cols = array('h', ioctl(sys.stdout.fileno(), TIOCGWINSZ, '\0' * 8))[1]
            print("\r" + " " * cols, flush=True, end="")
        if not self.lr_reduce:
            self.scheduler.step()
        self.optim.zero_grad(set_to_none=True)

        summary["scalars"] = {
            "loss": loss_total / idx,
        }
        if self.loss_aux is not None:
            summary["scalars"]["loss/aux"] = loss_aux_total / idx
        return summary

    @torch.no_grad()
    def valid_epoch(self, dataloader):
        self.eval()
        loss_total = 0.0
        n_items = 0
        for batch in tqdm(dataloader, desc="Valid", disable=(self.rank!=0), leave=False, dynamic_ncols=True):
            clean = batch["clean"].cuda(self.rank, non_blocking=True)
            noisy = batch["noisy"].cuda(self.rank, non_blocking=True)
            batch_size = clean.size(0)
            n_items += batch_size

            with amp.autocast(enabled=self.fp16):
                wav_hat = self.model(noisy)
                loss = F.mse_loss(wav_hat, clean)
                loss_total += loss.detach() * batch_size
                if self.loss_aux is not None:
                    spec_near = self._module.dct(near)
                    loss_aux = self.loss_aux(spec_hat, spec_near)
                    loss_aux_total += loss_aux.detach() * batch_size
        dist.reduce(loss_total, dst=0, op=dist.ReduceOp.SUM)
        n_items *= dist.get_world_size()
        loss = loss_total.item() / n_items
        summary_scalars = {
            "loss": loss,
        }
        if self.loss_aux is not None:
            dist.reduce(loss_aux_total, dst=0, op=dist.ReduceOp.SUM)
            loss_aux_total = loss_aux_total.item() / n_items
            summary_scalars["loss/aux"] = loss_aux_total
            loss += loss_aux_total * self.lambda_aux

        if self.lr_reduce:
            self.scheduler.step(loss)

        return {"scalars": summary_scalars}
    
    @torch.no_grad()
    def infer_epoch(self, dataloader):
        return
        

    def load(self, epoch: Optional[int] = None, path: Optional[str] = None):
        checkpoint = self.get_checkpoint(epoch, path)
        if checkpoint is None:
            return

        self._module.load_state_dict(checkpoint['model'])
        self.epoch = checkpoint['epoch']

        if self.train_mode:
            self.optim.load_state_dict(checkpoint['optim'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.scaler.load_state_dict(checkpoint['scaler'])
    
    def save(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(self.base_dir, f"{self.epoch:0>5d}.pth")
        wrapper_dict = {
            "model": self._module.state_dict(),
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch
        }

        torch.save(wrapper_dict, path)
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
    
    def remove_weight_reparameterizations(self):
        return
