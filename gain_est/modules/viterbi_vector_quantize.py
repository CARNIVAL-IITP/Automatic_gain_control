import random
from typing import Optional, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange, repeat
import numpy as np
from torch.cuda.amp import autocast


def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))

def laplace_smoothing(x, n_categories, eps = 1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def kmeans(samples, num_clusters, num_iters = 10, use_cosine_sim = False):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            distance = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') \
                    - rearrange(means, 'c d -> () c d')
            distance = -(diffs ** 2).sum(dim = -1)

        buckets = distance.max(dim = -1).indices
        bins = torch.bincount(buckets, minlength = num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype = dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        dim: int,
        codebook_size: int,
        kmeans_init: bool = False,
        kmeans_iters: int = 20,
        decay: float = 0.8,
        eps: float = 1e-5,
        ema_num_threshold: float = 0.0,
        ema_num_initial: float = 1.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.decay = decay
        init_fn = torch.randn if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.ema_num_threshold = ema_num_threshold
        self.ema_num_initial = ema_num_initial

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('embed', embed)
        self.register_buffer('ema_embed', embed.clone() * ema_num_initial)
        self.register_buffer('ema_num', torch.ones(codebook_size) * ema_num_initial)

        self.distributed = dist.is_initialized() and (dist.get_world_size() > 1)
        
    def init_embed_(self, data):
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        if self.distributed:
            dist.broadcast(embed, 0)
        self.embed.data.copy_(embed)
        self.ema_embed.data.copy_(embed * self.ema_num_initial)
        self.ema_num.data.fill_(self.ema_num_initial)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, samples, mask) -> int:
        idx = torch.nonzero(mask).squeeze(1)
        new_embed = sample_vectors(samples, idx.size(0)).detach().float()
        if self.distributed:
            dist.broadcast(new_embed, 0)
        self.embed.data[idx, :] = new_embed

        self.ema_embed.data[idx, :] = new_embed * self.ema_num_initial
        self.ema_num[idx].fill_(self.ema_num_initial)
        return idx.size(0)

    def expire_codes_(self, batch_samples):
        if self.ema_num_threshold == 0.0:
            return 0

        expired_codes = self.ema_num < self.ema_num_threshold
        if not torch.any(expired_codes):
            return 0
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        return self.replace(batch_samples, mask = expired_codes)

    @autocast(enabled=False)
    def forward(self, x: torch.Tensor, argmin: torch.Tensor, last: bool):
        # x: [N, M, Channel] where M=1 if self.layer_idx==0 else M=codebook_size
        N, M, C, cs = x.size(0), x.size(1), x.size(2), self.codebook_size
        embed = self.embed.t()      # [Channel, codebook_size]
        flatten = x.reshape(N * M, C)  # [N, M, Channel] -> [N x M, Channel]
        if not self.initted:
            self.init_embed_(flatten)

        if self.layer_idx == 0:
            argmin_curr = None
            residual = x - self.embed.unsqueeze(0)  # [N, 1, C] - [1, cs, C] = [N, cs, C]
        else:
            distance = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ embed
                + embed.pow(2).sum(0, keepdim=True)
            ).view(N, M, cs)        # [N, M, codebook_size]
            distance_min, argmin_prev = distance.min(dim = 1)       # [N, cs]
            argmin[self.layer_idx - 1, :, :] = argmin_prev
            if last:
                argmin_curr = distance_min.argmin(dim=1, keepdim=True)      # [N, 1]
                argmin_prev = torch.gather(input=argmin_prev, dim=1, index=argmin_curr) # [N, 1]
                argmin_prev = argmin_prev.unsqueeze(2).expand(-1, -1, C)    # [N, 1, C]
                residual_prev = torch.gather(input=x, dim=1, index=argmin_prev)         # [N, 1, C]
                quantize = F.embedding(argmin_curr, self.embed)             # [N, 1, C]
                residual = residual_prev - quantize                         # [N, 1, C]
            else:
                argmin_curr = None
                argmin_prev = argmin_prev.unsqueeze(2).expand(-1, -1, C)        # [N, cs, C]
                residual_prev = torch.gather(input=x, dim=1, index=argmin_prev) # [N, cs, C]
                residual = residual_prev - self.embed.unsqueeze(0)      # [N, cs, C]
        #print(f"forward {self.layer_idx}: {residual.squeeze()}")
        return residual, argmin_curr

    @autocast(enabled=False)
    def update(self, residual: torch.Tensor, argmin: torch.Tensor, argmin_curr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # residual: [N, 1, codebook_size] / argmin: [Layer-2, N, codebook_size] / argmin_curr: [N, 1]
        quantize = F.embedding(argmin_curr, self.embed)    # [N, 1, Channel]
        residual_prev = residual + quantize        # [N, 1, Channel]
        #print(f"backward {self.layer_idx}: {residual_prev.squeeze()}")
        if self.layer_idx > 0:       
            argmin_prev = torch.gather(input=argmin[self.layer_idx-1], dim=1, index=argmin_curr)    # [N, 1]
        else:
            argmin_prev = None
        embed_onehot = F.one_hot(argmin_curr.squeeze(1), self.codebook_size).float()   # [N, cs]
        x = residual_prev.squeeze(1)    # [N, Channel]
        if self.distributed:
            # Concatenate multiple tensors before all_reduce for speedup
            ema_num_numel, ema_embed_numel = self.ema_num.numel(), self.ema_embed.numel()
            bucket = torch.empty(ema_num_numel + ema_embed_numel, dtype=torch.float32)
            bucket[:ema_num_numel] = embed_onehot.sum(dim=0)
            torch.matmul(x.t(), embed_onehot, out=bucket[ema_num_numel:])
            dist.all_reduce(bucket)
            ema_num_new = bucket[:ema_num_numel]
            ema_embed_new = bucket[ema_num_numel:]
        else:
            ema_num_new = embed_onehot.sum(dim=0)   # [cs]
            ema_embed_new = x.t() @ embed_onehot    # [Channel, cs]
        ema_inplace(self.ema_num, ema_num_new, self.decay)
        ema_inplace(self.ema_embed, ema_embed_new.t(), self.decay)

        # Make sure that ema_num > eps (Not needed if use random reinitialization)
        if self.ema_num_threshold > 0.0:
            ema_num = self.ema_num
        else:
            ema_num = laplace_smoothing(self.ema_num, self.codebook_size, self.eps) * self.ema_num.sum()

        embed_normalized = self.ema_embed / ema_num.unsqueeze(1)
        self.embed.data.copy_(embed_normalized)
        num_replace = self.expire_codes_(x)

        return residual_prev, argmin_prev, num_replace


class ViterbiVQ(nn.Module):
    def __init__(
        self,
        num_quantizers: int,
        dropout: bool = False,
        dropout_index: Optional[List[int]] = None,
        commitment: float = 1.,
        use_shape_gain: bool = False,
        channel_last: bool = False,
        **kwargs
    ):
        super().__init__()
        self.commitment = commitment
        self.use_shape_gain = use_shape_gain
        self.channel_last = channel_last
        self.use_shape_gain = use_shape_gain
        self.codebook_size = kwargs["codebook_size"]

        if use_shape_gain:
            raise NotImplementedError("ViterbiVQ with shape-gain codebook is not implemented.")
        else:
            codebook_class = EuclideanCodebook

        assert num_quantizers >= 2, "num_quantizers must be >= 2"
        self.layers = nn.ModuleList([codebook_class(layer_idx = i, **kwargs) for i in range(num_quantizers)])
        self.dropout = dropout
        self.dropout_index = dropout_index      # index of VQ layers to choose when dropout==True

    def forward(self, x, n=None, calculate_loss=True):
        x_original = x

        x = x.detach()
        if not self.channel_last:
            x = x.transpose(1, 2)          # [batch, channel, time] -> [batch, time, channel]
        batch, time = x.size(0), x.size(1)
        x = x.reshape(batch * time, 1, -1)    # [batch, time, channel] -> [batch x time, 1, channel]
        argmin = torch.empty((len(self.layers)-1, x.size(0) * x.size(1), self.codebook_size), dtype=torch.long, device=x.device)
        
        k = 2 if self.use_shape_gain else 1
        num_replaces = np.zeros(k * len(self.layers), dtype=np.int64)

        if n is not None:
            assert 1 <= n <= len(self.layers), f"'n' must be in range of 1 <= n <= {len(self.layers)}"
            high = n
        elif self.training and self.dropout:
            high = random.sample(self.dropout_index, 1)[0]
        else:
            high = len(self.layers)
        
        residual = x.float()
        for idx, layer in enumerate(self.layers[:high]):
            last_layer = (idx == high - 1)
            residual, argmin_curr = layer(residual, argmin, last=last_layer)
        
        residual_out = residual.view(batch, time, -1)   # [batch x time, channel] -> [batch, time, channel]
        if not self.channel_last:
            residual_out = residual_out.transpose(1, 2)     # [batch, time, channel] -> [batch, channel, time]
        
        quantize = x_original - residual_out
        loss = F.mse_loss(quantize, x_original) * self.commitment if calculate_loss else None
        
        if self.training:
            for idx in range(high - 1, -1, -1):
                residual, argmin_curr, num_replace = self.layers[idx].update(residual, argmin, argmin_curr)
                num_replaces[k * idx : k * (idx + 1)] = num_replace
            assert torch.all((residual - x).abs() < 1e-5), "forward & backward in viterbi VQ unmatched!"
        
        return quantize, num_replaces, loss


class VQ(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._codebook = EuclideanCodebook(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        return self._codebook(*args, **kwargs)


class ViterbiVQLegacy(nn.Module):
    def __init__(
        self,
        num_quantizers: int,
        dropout: bool = False,
        dropout_index: Optional[List[int]] = None,
        commitment: float = 1.,
        use_shape_gain: bool = False,
        channel_last: bool = False,
        **kwargs
    ):
        super().__init__()
        self.commitment = commitment
        self.use_shape_gain = use_shape_gain
        self.channel_last = channel_last
        self.use_shape_gain = use_shape_gain
        self.codebook_size = kwargs["codebook_size"]

        assert num_quantizers >= 2, "num_quantizers must be >= 2"
        self.layers = nn.ModuleList([VQ(layer_idx = i, **kwargs) for i in range(num_quantizers)])
        self.dropout = dropout
        self.dropout_index = dropout_index      # index of VQ layers to choose when dropout==True

    def forward(self, x, n=None, calculate_loss=True):
        x_original = x

        x = x.detach()
        if not self.channel_last:
            x = x.transpose(1, 2)          # [batch, channel, time] -> [batch, time, channel]
        batch, time = x.size(0), x.size(1)
        x = x.reshape(batch * time, 1, -1)    # [batch, time, channel] -> [batch x time, 1, channel]
        argmin = torch.empty((len(self.layers)-1, x.size(0) * x.size(1), self.codebook_size), dtype=torch.long, device=x.device)
        
        k = 2 if self.use_shape_gain else 1
        num_replaces = np.zeros(k * len(self.layers), dtype=np.int64)

        if n is not None:
            assert 1 <= n <= len(self.layers), f"'n' must be in range of 1 <= n <= {len(self.layers)}"
            high = n
        elif self.training and self.dropout:
            high = random.sample(self.dropout_index, 1)[0]
        else:
            high = len(self.layers)
        
        residual = x.float()
        for idx, layer in enumerate(self.layers[:high]):
            last_layer = (idx == high - 1)
            residual, argmin_curr = layer(residual, argmin, last=last_layer)
        
        residual_out = residual.view(batch, time, -1)   # [batch x time, channel] -> [batch, time, channel]
        if not self.channel_last:
            residual_out = residual_out.transpose(1, 2)     # [batch, time, channel] -> [batch, channel, time]
        
        quantize = x_original - residual_out
        loss = F.mse_loss(quantize, x_original) * self.commitment if calculate_loss else None
        
        if self.training:
            for idx in range(high - 1, -1, -1):
                residual, argmin_curr, num_replace = self.layers[idx].update(residual, argmin, argmin_curr)
                num_replaces[k * idx : k * (idx + 1)] = num_replace
            assert torch.all((residual - x).abs() < 1e-5), "forward & backward in viterbi VQ unmatched!"
        
        return quantize, num_replaces, loss


if __name__=="__main__":
    B, T, D = 1, 4, 1
    codebook_size = 2
    num_quantizers = 3
    vq = ViterbiVQ(num_quantizers, dropout=False, dim=D, codebook_size=codebook_size, kmeans_init=False)
    for idx, layer in enumerate(vq.layers):
        if idx == 0:
            layer.embed.data = torch.tensor([[0.0], [3.0]])
        elif idx == 1:
            layer.embed.data = torch.tensor([[0.0], [1.9]])
        elif idx == 2:
            layer.embed.data = torch.tensor([[0.0], [1.0]])
    x = torch.arange(B*T*D).view(B, D, T).contiguous()
    print(x)
    vq(x)