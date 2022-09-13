import warnings
from typing import Optional, Dict
import math
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

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


def istft(spec: Tensor, n_fft: int, hop_size: int, win_size: int, center: bool = False) -> Tensor:
    ''' if center == True @ stft:
        spec shape: [batch_size, n_fft//2+1, wav_len//hop_size + 1, 2]
        output shape: [batch_size, wav_len]
        -> input[:, :] ~= output[:, :]
    else:
        spec shape: [batch_size, n_fft//2+1, wav_len//hop_size, 2]
        output shape: [batch_size, wav_len - hop_size]
        -> input[:, hop_size//2:-hop_size//2] ~= output[:, :]
    '''
    if not center:
        raise NotImplementedError("center=False is not implemented. Please use center=True to both stft & istft")
    
    global hann_window
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=spec.dtype, device=spec.device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wav = torch.istft(spec, n_fft, hop_length=hop_size, win_length=win_size, center=center, normalized=False,
            window=hann_window[wnsize_dtype_device], onesided=True, return_complex=False)

    return wav

def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.
    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where
        output_size = (frames - 1) * frame_step + frame_length
    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length
    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)#.cuda()

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long()#.cuda()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result