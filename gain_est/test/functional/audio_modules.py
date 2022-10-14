import warnings
from typing import Optional, Dict
import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn


class STDCT(torch.jit.ScriptModule):
    '''Short-Time Discrete Cosine Transform II
    forward(x, inverse=False):
        x: [B, 1, hop_size*T] or [B, hop_size*T]
        output: [B, N, T+1] (center = True)
        output: [B, N, T]   (center = False)
    forward(x, inverse=True):
        x: [B, N, T+1] (center = True)
        x: [B, N, T]   (center = False)
        output: [B, 1, hop_size*T]'''

    __constants__ = ["N", "hop_size", "padding"]

    def __init__(self, N: int, hop_size: int, win_size: Optional[int] = None,
                 win_type: Optional[str] = "hann", center: bool = False,
                 window: Optional[Tensor] = None, device=None, dtype=None):
        super().__init__()
        self.N = N
        self.hop_size = hop_size
        self.padding = N // 2 if center else (N - hop_size) // 2

        factory_kwargs = {'device': device, 'dtype': dtype}

        if win_size is None:
            win_size = N
        
        if window is not None:
            win_size = window.size(-1)
            if win_size < N:
                padding = N - win_size
                window = F.pad(window, (padding//2, padding - padding//2))
        elif win_type is None:
            window = torch.ones(N, dtype=torch.float32, device=device)
        else:
            window: Tensor = getattr(torch, f"{win_type}_window")(win_size, device=device)
            if win_size < N:
                padding = N - win_size
                window = F.pad(window, (padding//2, padding - padding//2))
        assert N >= win_size, f"N({N}) must be bigger than win_size({win_size})"
        n = torch.arange(N, dtype=torch.float32, device=device).view(1, 1, N)
        k = n.view(N, 1, 1)
        _filter = torch.cos(math.pi/N*k*(n+0.5)) * math.sqrt(2/N)
        _filter[0, 0, :] *= 2
        dct_filter = (_filter * window.view(1, 1, N)).to(**factory_kwargs)
        window_square = window.square().view(1, -1, 1).to(**factory_kwargs)
        self.register_buffer('filter', dct_filter)
        self.register_buffer('window_square', window_square)
        self.filter: Tensor
        self.window_square: Tensor
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 1, hop_size*T] or [B, hop_size*T]
        # output: [B, N, T+1] (center = True)
        # output: [B, N, T]   (center = False)
        if x.dim() == 2:
            x = x.unsqueeze(1)
    
        return F.conv1d(x, self.filter, bias=None, stride=self.hop_size,
                        padding=self.padding)
    
    @torch.jit.export
    def inverse(self, spec: Tensor) -> Tensor:
        # x: [B, N, T+1] (center = True)
        # x: [B, N, T]   (center = False)
        # output: [B, 1, hop_size*T]
        wav =  F.conv_transpose1d(spec, self.filter, bias=None,
                                  stride=self.hop_size, padding=self.padding)
        B, T = spec.size(0), spec.size(-1)
        window_square = self.window_square.expand(B, -1, T)
        window_square_inverse = F.fold(
            window_square,
            output_size = (1, self.hop_size*T + (self.N-self.hop_size) - 2*self.padding),
            kernel_size = (1, self.N),
            stride = (1, self.hop_size),
            padding = (0, self.padding)
        ).squeeze(2)

        # NOLA(Nonzero Overlap-add) constraint
        assert torch.all(torch.ne(window_square_inverse, 0.0))
        return wav / window_square_inverse


class MDCT(torch.jit.ScriptModule):
    '''Modified Discrete Cosine Transform
    forward(x, inverse=False):
        y: [B, 1, N * T] -> pad N to left & right each.
        output: [B, N, T + 1]
    forward(x, inverse=True):
        y: [B, N, T + 1]
        output: [B, 1, N * T]'''

    __constants__ = ["N", "filter", "normalize"]

    def __init__(self, N: int, normalize: bool = True, device=None, dtype=None):
        super().__init__()
        self.N = N
        self.normalize = normalize

        k = torch.arange(N, dtype=torch.float32, device=device).view(N, 1, 1)
        n = torch.arange(2*N, dtype=torch.float32, device=device).view(1, 1, 2*N)
        mdct_filter = torch.cos(math.pi/N*(n+0.5+N/2)*(k+0.5))
        if normalize:
            mdct_filter /= math.sqrt(N)
        mdct_filter = mdct_filter.to(device=device, dtype=dtype)
        self.register_buffer("filter", mdct_filter)
        self.filter: Tensor

    def forward(self, x: Tensor) -> Tensor:
        return F.conv1d(x, self.filter, bias=None, stride=self.N, padding=self.N)
    
    @torch.jit.export
    def inverse(self, x: Tensor) -> Tensor:
        if self.normalize:
            mdct_filter = self.filter
        else:
            mdct_filter = self.filter / self.N
        return F.conv_transpose1d(x, mdct_filter, bias=None, stride=self.N, padding=self.N)


class STFT(nn.Module):
    ''' y shape: [batch_size, wav_len] or [batch_size, 1, wav_len]
    output shape: [batch_size, wav_len//hop_size, 2] (center == False, magnitude=False)
    output shape: [batch_size, wav_len//hop_size+1, 2] (center == True, magnitude=False)
    '''

    __constants__ = ["n_fft", "hop_size", "normalize"]
    __annotations__ = {'window': Optional[torch.Tensor]}

    def __init__(self, n_fft: int, hop_size: int, win_size: Optional[int], center: bool = False,
                 win_type: Optional[str] = "hann", window: Optional[Tensor] = None,
                 device=None, dtype=None):
        raise NotImplementedError("Currently STFT Module is not implemented. Use functional.stft instead.")
        if win_size is None:
            win_size = n_fft
        
        if window is not None:
            win_size = window.size(-1)
        elif win_type is None:
            self.register_buffer()
        else:
            window: Tensor = getattr(torch, f"{win_type}_window")(win_size, device=device)
        assert n_fft >= win_size, f"n_fft({n_fft}) must be bigger than win_size({win_size})"

