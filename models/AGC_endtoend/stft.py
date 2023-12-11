import warnings
from typing import Optional, Dict
import math
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.utils.data
hann_window: Dict[str, Tensor]  = {}

def stft(y: Tensor, n_fft: int, hop_size: int, win_size: int, center: bool = False, magnitude: bool = True,
         check_value: bool = False) -> Tensor:
    ''' y shape: [batch_size, wav_len] or [batch_size, 1, wav_len]
    output shape: [batch_size, wav_len//hop_size, 2] (center == False, magnitude=False)
    output shape: [batch_size, wav_len//hop_size+1, 2] (center == True, magnitude=False)
    '''
    if y.dim() == 3:  # [B, 1, T] -> [B, T]
        y = y.squeeze(1)
    
    if check_value:
        if torch.min(y) < -1.:
            print('min value is ', torch.min(y))
        if torch.max(y) > 1.:
            print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    if not center:
        y = F.pad(y.unsqueeze(0), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
        y = y.squeeze(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size,
            window=hann_window[wnsize_dtype_device], center=center, pad_mode='constant',
            normalized=False, onesided=True, return_complex=False)
    
    if magnitude:
        spec = torch.linalg.norm(spec, dim=-1)

    return spec