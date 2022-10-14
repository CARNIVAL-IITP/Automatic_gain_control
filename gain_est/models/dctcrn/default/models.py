import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from functional import mdct, imdct, stdct, istdct
from .layers import CausalConv2d, CausalConvTranspose2d, LayerNorm2d, ChannelFreqNorm, BatchRenorm2d

class PMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu = nn.PReLU()
    
    def forward(self, x: Tensor, noisy: Tensor) -> Tensor:
        x = self.prelu(x)
        x = torch.where(
            torch.le(x.abs(), 1.0),
            x,
            x.detach().sign() - x.detach() + x
        )
        return x * noisy


class SMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: Tensor, noisy: Tensor) -> Tensor:
        return self.sigmoid(x) * noisy


class TMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()
    
    def forward(self, x: Tensor, noisy: Tensor) -> Tensor:
        return self.tanh(x) * noisy


class NoMask(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor, noisy: Tensor) -> Tensor:
        return x

class DCTCRN(nn.Module):
    def __init__(
        self, 
        rnn_layers=2,
        rnn_channels=256,
        N=512,
        hop_size=128,
        win_size=512,
        win_type="hann",
        kernel_size=(5, 2),
        channels=[8, 16, 32, 64, 128, 128, 256],
        masking_mode="T",
        norm="batchnorm",
        use_mdct=False,
    ):
        super(DCTCRN, self).__init__()

        self.win_size = win_size
        self.hop_size = hop_size
        self.N = N
        
        if masking_mode == "P":     # PReLU + clipping
            self.mask = PMask()
        elif masking_mode == "S":   # Sigmoid
            self.mask = SMask()
        elif masking_mode == "T":   # Tanh
            self.mask = TMask()
        elif masking_mode == "N":   # No mask
            self.mask = NoMask()
        else:
            raise NotImplementedError(f"Invalid masking_mode {masking_mode}")

        channels = [2] + channels

        if norm.lower() == "batchnorm":
            norm_layer = lambda idx : nn.BatchNorm2d(channels[idx])
        elif norm.lower() == "batchrenorm":
            norm_layer = lambda idx : BatchRenorm2d(channels[idx])
        elif norm.lower() == "layernorm":
            norm_layer = lambda idx : LayerNorm2d(channels[idx])
        elif norm.lower() == "channelfreqnorm":
            norm_layer = lambda idx : ChannelFreqNorm(channels[idx], N//2**idx)
        elif norm.lower() == "identity":
            norm_layer = nn.Identity
        else:
            raise NotImplementedError(f"model_kwargs.norm '{norm}' is not implemented.")
        
        if use_mdct:
            self.dct = lambda x: mdct(x, N, normalize=True)
            self.idct = lambda x: imdct(x, N, normalize=True)
        else:
            self.dct = lambda x: stdct(x, N, hop_size, win_size, center=True, win_type=win_type)
            self.idct = lambda x: istdct(x, N, hop_size, win_size, center=True, win_type=win_type)
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for idx in range(len(channels)-1):
            self.encoder.append(
                nn.Sequential(
                    CausalConv2d(
                        channels[idx],
                        channels[idx+1],
                        kernel_size=kernel_size,
                        stride=((kernel_size[0]-1)//2, 1),
                    ),
                    norm_layer(idx+1),
                    nn.PReLU()
                )
            )
        hidden_dim = self.N//(2**(len(channels)-1)) 
        self.enhance = nn.LSTM(
                input_size= hidden_dim*channels[-1],
                hidden_size=rnn_channels,
                num_layers=rnn_layers,
                dropout=0.0,
                bidirectional=False,
                batch_first=False
        )
        self.enhance.flatten_parameters()
        self.tranform = nn.Linear(rnn_channels, hidden_dim*channels[-1])
        #self.tranform = nn.Linear(hidden_dim*channels[-1], hidden_dim*channels[-1])

        for idx in range(len(channels)-1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    nn.Sequential(
                        CausalConvTranspose2d(
                            channels[idx]*2,
                            channels[idx-1],
                            kernel_size=(4, kernel_size[1]),
                            stride=(2, 1),
                        ),
                        norm_layer(idx-1),
                        nn.PReLU()
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        CausalConvTranspose2d(
                            channels[idx]*2,
                            1,
                            kernel_size=(4, kernel_size[1]),
                            stride=(2, 1),
                        ),
                    )
                )

    def forward(self, far, mix):
        # input: [B, Twav]
        # output wav: [B, Twav]
        # output spec: [B, N, Tspec] where Tspec * hop_size == Twav
        far_spec = self.dct(far).unsqueeze(1)       # [B, 1, N, Tspec]
        mix_spec = self.dct(mix).unsqueeze(1)       # [B, 1, N, Tspec]
        x = torch.cat([far_spec, mix_spec], dim=1)  # [B, 2, N, Tspec]
        #x = mix_spec
        encoder_out = []
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            encoder_out.append(x)
        
        batch_size, channels, dims, lengths = x.size()
        x = x.permute(3, 0, 1, 2)
        # to [L, B, C, D]
        x = torch.reshape(x, [lengths, batch_size, channels*dims])
        x, _ = self.enhance(x)
        x = self.tranform(x)
        x = torch.reshape(x, [lengths, batch_size, channels, dims])
       
        x = x.permute(1, 2, 3, 0)
        for idx in range(len(self.decoder)):
            x = torch.cat([x, encoder_out[-1 - idx]], 1)
            x = self.decoder[idx](x)
        x = self.mask(x, mix_spec).squeeze(1)
        wav = self.idct(x).squeeze(1)
        return wav, x

if __name__ == '__main__' :
    import numpy as np
    model = DCTCRN()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)