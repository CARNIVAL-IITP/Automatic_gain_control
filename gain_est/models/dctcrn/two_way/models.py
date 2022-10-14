import torch
import torch.nn as nn
from torch import Tensor

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
        rnn_channels=128,
        N=480,
        hop_size=100,
        win_size=480,
        win_type="hann",
        kernel_size=(5, 2),
        channels=[32,64,128,128,256,256],
        masking_mode="N",
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
        self.decoder_speech = nn.ModuleList()
        self.decoder_echo = nn.ModuleList()
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
        self.w1 = nn.Parameter(torch.randn(1)) # learnable parameter 실험
        #self.w2 = nn.Parameter(torch.randn(1))

        for idx in range(len(channels)-1, 0, -1):
            if idx != 1:
                self.decoder_speech.append(
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
                self.decoder_echo.append(
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
                self.decoder_speech.append(
                    nn.Sequential(
                        CausalConvTranspose2d(
                            channels[idx]*2,
                            1,
                            kernel_size=(4, kernel_size[1]),
                            stride=(2, 1),
                        ),
                    )
                )
                self.decoder_echo.append(
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
        x_speech = x.clone()
        x_echo = x.clone()
        for idx in range(len(self.decoder_speech)):
            x_speech = torch.cat([x_speech, encoder_out[-1 - idx]], 1)
            x_speech = self.decoder_speech[idx](x_speech)
            x_echo = torch.cat([x_echo, encoder_out[-1 - idx]], 1)
            x_echo = self.decoder_echo[idx](x_echo)
        
        mask_speech = self.mask(x_speech, mix_spec).squeeze(1)
        reg_speech = (mix_spec - x_echo).squeeze(1)
        res = 0.5 * reg_speech + 0.5 * mask_speech
        #res = self.w1 * reg_speech + (1 - self.w1) * mask_speech
        wav = self.idct(res).squeeze(1)
        #echo = self.idct(x_echo.squeeze(1)).squeeze(1)
        return wav, res
