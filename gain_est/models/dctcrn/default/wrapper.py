import os, math, sys
from array import array
from fcntl import ioctl
from termios import TIOCGWINSZ
from typing import Optional

import torch
from torch.cuda import amp
import torch.distributed as dist
from tqdm import tqdm

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
        wav_hat, _ = self.model(wav)
        return wav_hat
    
    def train_epoch(self, dataloader):
        self.train()

        max_items = len(dataloader)
        padding = int(math.log10(max_items)) + 1
        
        summary = {"scalars": {}, "hists": {}}
        loss_sisnr_total, loss_aux_total = 0.0, 0.0
        for idx, batch in enumerate(dataloader, start=1):
            self.optim.zero_grad(set_to_none=True)
            clean = batch["clean"].cuda(self.rank, non_blocking=True)
            noisy = batch["noisy"].cuda(self.rank, non_blocking=True)
            
            with amp.autocast(enabled=self.fp16):
                wav_hat, spec_hat = self.model(far, mix)
                loss = si_snr(wav_hat, near)
                loss_sisnr_total += loss.item()
                if self.loss_aux is not None:
                    spec_near = self._module.dct(near)
                    loss_aux = self.loss_aux(spec_hat, spec_near)
                    loss_aux_total += loss_aux.item()
                    loss += loss_aux * self.lambda_aux

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
                    f"    si_snr: {loss_sisnr_total / idx:6.4f}"
                    f"    aux: {loss_aux_total / idx:6.4f}"
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
            "loss/si_snr": loss_sisnr_total / idx,
        }
        if self.loss_aux is not None:
            summary["scalars"]["loss/aux"] = loss_aux_total / idx
        return summary

    @torch.no_grad()
    def valid_epoch(self, dataloader):
        self.eval()
        loss_sisnr_total, loss_aux_total = 0.0, 0.0
        n_items = 0
        for batch in tqdm(dataloader, desc="Valid", disable=(self.rank!=0), leave=False, dynamic_ncols=True):
            near = batch["near"].cuda(self.rank, non_blocking=True)
            far = batch["far"].cuda(self.rank, non_blocking=True)
            mix = batch["mix"].cuda(self.rank, non_blocking=True)
            batch_size = near.size(0)
            n_items += batch_size

            with amp.autocast(enabled=self.fp16):
                wav_hat, spec_hat = self.model(far, mix)
                loss = si_snr(wav_hat, near)
                loss_sisnr_total += loss.detach() * batch_size
                if self.loss_aux is not None:
                    spec_near = self._module.dct(near)
                    loss_aux = self.loss_aux(spec_hat, spec_near)
                    loss_aux_total += loss_aux.detach() * batch_size
        dist.reduce(loss_sisnr_total, dst=0, op=dist.ReduceOp.SUM)
        n_items *= dist.get_world_size()
        loss = loss_sisnr_total.item() / n_items
        summary_scalars = {
            "loss/si_snr": loss,
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
        self.eval()
        # assume mel shape: [1, n_mel, t_mel] (batch size = 1)
        summary = {"audios": {}, "specs": {}, "scalars": {}}
        pesq_mean = 0
        total_cnt = 0
        for idx, batch in enumerate(dataloader, start=1):
            near = batch["near"].cuda(self.rank, non_blocking=True)
            far = batch["far"].cuda(self.rank, non_blocking=True)
            mix = batch["mix"].cuda(self.rank, non_blocking=True)
            b = near.size(0)
            wav_len = near.size(-1)
            hop_size = 256
            discard_len = wav_len - wav_len // hop_size * hop_size
            if discard_len > 0:
                near = near[..., :-discard_len]
                far = far[..., :-discard_len]
                mix = mix[..., :-discard_len]
            
            if self.epoch == 1:
                spec_near = stft(near, 1024, 256, 1024)
                spec_far = stft(far, 1024, 256, 1024)
                spec_mix = stft(mix, 1024, 256, 1024)
                mel_near = spec_to_mel(spec_near, 1024, 80, self.h.sampling_rate)
                mel_far = spec_to_mel(spec_far, 1024, 80, self.h.sampling_rate)
                mel_mix = spec_to_mel(spec_mix, 1024, 80, self.h.sampling_rate)
                
                summary["audios"][f"near/wav_{idx}"] = batch["near"].squeeze().cpu().numpy()
                summary["specs"][f"near/mel_{idx}"] = mel_near.squeeze().cpu().numpy()
                summary["specs"][f"near/spec_{idx}"] = spec_near.clamp_min(1e-5).log().squeeze().cpu().numpy()
                summary["audios"][f"far/wav_{idx}"] = batch["far"].squeeze().numpy()
                summary["specs"][f"far/mel_{idx}"] = mel_far.squeeze().cpu().numpy()
                summary["specs"][f"far/spec_{idx}"] = spec_far.clamp_min(1e-5).log().squeeze().cpu().numpy()
                summary["audios"][f"mix/wav_{idx}"] = batch["mix"].squeeze().numpy()
                summary["specs"][f"mix/mel_{idx}"] = mel_mix.squeeze().cpu().numpy()
                summary["specs"][f"mix/spec_{idx}"] = spec_mix.clamp_min(1e-5).log().squeeze().cpu().numpy()
            
            with torch.no_grad():
                wav_hat, _ = self.model(far, mix)

                # SI-SNR produces scale-invariant output, which is usually too small.
                # Therefore, we increase the power of the output
                near_power = near.square().sum()
                wav_hat_power = wav_hat.square().sum()
                wav_hat *= math.sqrt(near_power / wav_hat_power)
            
                spec_hat = stft(wav_hat, 1024, 256, 1024)
                mel_hat = spec_to_mel(spec_hat, 1024, 80, self.h.sampling_rate)

            pesq_mean += pesq(near.squeeze().cpu().numpy(), wav_hat.squeeze().cpu().numpy(), 16000)
            #pesq_est, pesq_mix = cal_pesq_diff(near.squeeze().cpu().numpy(), wav_hat.squeeze().cpu().numpy(), mix.squeeze().cpu().numpy())

            #pesq_est, pesq_mix = cal_pesq_diff(near_ref, near_est, mix)
            #total_pesq_est += pesq_est
            #total_pesq_mix += pesq_mix
            #total_pesq_diff += (pesq_est - pesq_mix)
            print(f"\r{total_cnt} / 500  - {pesq_mean / (total_cnt + 1)}", end=" ", flush=True)
            # Compute SI-SNRi

            total_cnt += 1
            if total_cnt == 2:
                summary["audios"][f"gen/wav_{idx}"] = wav_hat.squeeze().cpu().numpy()
                summary["specs"][f"gen/mel_{idx}"] = mel_hat.squeeze().cpu().numpy()
                summary["specs"][f"gen/spec_{idx}"] = spec_hat.clamp_min(1e-5).log().squeeze().cpu().numpy()


        print("\nAverage PESQ(Enhanced) : {0:.2f}".format(pesq_mean / (total_cnt)))
        #print("Average PESQ(Noisy) : {0:.2f}".format(total_pesq_mix / (total_cnt)))
        #print("Average PESQ improvement: {0:.2f}".format(total_pesq_diff / (total_cnt)))
        summary['scalars']['pesq_diff'] = (pesq_mean / total_cnt)

        return summary

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
