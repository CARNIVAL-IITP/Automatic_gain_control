import math
import torch
from torch.nn import functional as F


def product(s1, s2):
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm 


def si_snr(s1, s2, eps=1e-7):
    s1_s2_norm = product(s1, s2)
    s2_s2_norm = product(s2, s2)
    s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = product(s_target, s_target)
    noise_norm = product(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return -torch.mean(snr)


@torch.jit.script
def abs_mse_loss(s1: torch.Tensor, s2: torch.Tensor, power: float = 1.0, eps: float = 1e-7) -> torch.Tensor:
    s1 = s1.abs()
    s2 = s2.abs()
    s1 = torch.where(s1.abs() >= eps, s1, eps - s1.detach() + s1)
    s2 = torch.where(s2.abs() >= eps, s2, eps - s2.detach() + s2)
    s1 = s1.pow(power)
    s2 = s2.pow(power)
    return F.mse_loss(s1, s2)
