import torch
import torch.nn as nn
from torch import Tensor
import einops
from functional import mdct, imdct, stdct, istdct
from .layers import CausalConv2d, CausalConvTranspose2d, LayerNorm2d, ChannelFreqNorm, BatchRenorm2d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

class ASA(nn.Module):
    def __init__(self, c=64, causal=True):
        super(ASA, self).__init__()
        self.d_c = c//8
        self.f_qkv = CausalConv2d(c, self.d_c * 3, kernel_size=(1,1), stride=(1,1))
        self.t_qk = CausalConv2d(c, self.d_c * 2, kernel_size=(1,1), stride=(1,1))
        self.proj = CausalConv2d(self.d_c, c, kernel_size=(1,1), stride=(1,1))
        self.causal = causal

    def forward(self, inp):
        """
        inp: B C F T
        """
        # f-attention
        f_qkv = self.f_qkv(inp)
        qf, kf, v = tuple(einops.rearrange(
            f_qkv, "b (c k) f t->k b c f t", k=3))
        f_score = torch.einsum("bcft,bcyt->btfy", qf, kf) / (self.d_c**0.5)
        f_score = f_score.softmax(dim=-1)
        print(f_score.shape)
        f_out = torch.einsum('btfy,bcyt->bcft', [f_score, v])
        # t-attention
        t_qk = self.t_qk(inp)
        qt, kt = tuple(einops.rearrange(t_qk, "b (c k) f t->k b c f t", k=2))
        t_score = torch.einsum('bcft,bcfy->bfty', [qt, kt]) / (self.d_c**0.5)
        print(t_score.shape)
        mask_value = max_neg_value(t_score)
        if self.causal:
            i, j = t_score.shape[-2:]
            mask = torch.ones(i, j, device=t_score.device).triu_(j - i + 1).bool()
            t_score.masked_fill_(mask, mask_value) 
        t_score = t_score.softmax(dim=-1)
        t_out = torch.einsum('bfty,bcfy->bcft', [t_score, f_out])
        out = self.proj(t_out)
        return out + inp


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
                    ASA(c=channels[idx+1]),
                    norm_layer(idx+1),
                    nn.PReLU()
                )
            )
        hidden_dim = self.N//(2**(len(channels)-1)) 
        '''
        self.enhance = nn.LSTM(
                input_size= hidden_dim*channels[-1],
                hidden_size=rnn_channels,
                num_layers=rnn_layers,
                dropout=0.0,
                bidirectional=False,
                batch_first=False
        )
        self.enhance.flatten_parameters()
        '''
        #self.tranform = nn.Linear(rnn_channels, hidden_dim*channels[-1])
        self.tranform = nn.Linear(hidden_dim*channels[-1], hidden_dim*channels[-1])
        
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
        print(x.shape)

        encoder_out = []
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            encoder_out.append(x)
        
        batch_size, channels, dims, lengths = x.size()
        x = x.permute(3, 0, 1, 2)
        # to [L, B, C, D]
        x = torch.reshape(x, [lengths, batch_size, channels*dims])
        #x, _ = self.enhance(x)
        x = self.tranform(x)
        x = torch.reshape(x, [lengths, batch_size, channels, dims])
       
        x = x.permute(1, 2, 3, 0)
        for idx in range(len(self.decoder)):
            x = torch.cat([x, encoder_out[-1 - idx]], 1)
            x = self.decoder[idx](x)
        x = self.mask(x, mix_spec).squeeze(1)
        wav = self.idct(x).squeeze(1)
        return wav, x
