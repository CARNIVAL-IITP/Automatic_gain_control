import math
import torch
from torch.nn import Parameter
from torch import Tensor
from torch import jit
import torch.nn as nn
import torch.nn.functional as F


class ChannelFreqNorm(jit.ScriptModule):
    __constants__ = ['channels', 'freq', 'eps']
    def __init__(self, channels, freq, zero_init=False, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.freq = freq
        self.eps = eps

        if zero_init:
            weight = torch.zeros(1, channels)
        else:
            weight = torch.ones(1, channels)
        self.weight = Parameter(weight)
        self.bias = Parameter(torch.zeros(1, channels))

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 3)   # [B, C, F, T] -> [B, T, F, C]
        x = F.layer_norm(x, (self.freq, self.channels), self.weight.expand(self.freq, -1), self.bias.expand(self.freq, -1), self.eps)
        x = x.transpose(1, 3)
        return x


class LayerNorm2d(jit.ScriptModule):
    __constants__ = ['channels', 'eps']
    def __init__(self, channels: int, zero_init: bool = False, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        if zero_init:
            weight = torch.zeros(channels)
        else:
            weight = torch.ones(channels)
        self.weight = Parameter(weight)
        self.bias = Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W]
        x = x.transpose(1, 3).contiguous()   # [B, W, H, C]
        x = F.layer_norm(x, (self.channels,), self.weight, self.bias, self.eps)
        return x.transpose(1, 3)


class BatchRenorm2d(jit.ScriptModule):
    """ Currently we didn't implemented r_max/d_max, because
    in our case, we use this module due to its ability of calculating
    identical output for training mode and evaluation mode, not
    because of small batch size. """
    __constants__ = ["momentum", "eps", "channels"]
    def __init__(self, channels: int, eps: float = 1e-5,
                 momentum: float = 0.1, zero_init: bool = False,):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.momentum = momentum

        if zero_init:
            weight = torch.zeros(channels)
        else:
            weight = torch.ones(1, channels, 1, 1)
        self.weight = Parameter(weight)
        self.bias = Parameter(torch.zeros(1, channels, 1, 1))
        self.register_buffer('running_mean', torch.zeros(1, channels, 1, 1))
        self.register_buffer('running_std', torch.ones(1, channels, 1, 1))
        self.register_buffer('num_batches_tracked', torch.zeros(1, dtype=torch.long))
        self.running_mean: Tensor
        self.running_std: Tensor
        self.num_batches_tracked: Tensor

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W]
        if self.training:
            var, mean = torch.var_mean(x, (0, 2, 3), unbiased=False, keepdim=True)
            std = torch.sqrt(var + self.eps)

            running_mean, running_std = self.running_mean.clone(), self.running_std.clone()
            r = std.detach() / running_std
            d = (mean.detach() - running_mean) / running_std
            #x = ((x - mean) / std * r + d) * self.weight + self.bias
            gain = self.weight * r / std
            bias = self.bias + d * self.weight - mean * gain
            x = torch.addcmul(bias, x, gain)

            with torch.no_grad():
                torch.lerp(self.running_mean, mean.float(), self.momentum, out=self.running_mean)
                torch.lerp(self.running_std, std.float(), self.momentum, out=self.running_std)
                self.num_batches_tracked.add_(1)
        else:
            #return (x - self.running_mean) / self.running_std * self.weight + self.bias
            gain = self.weight / self.running_std
            bias = self.bias - self.running_mean * gain
            x = torch.addcmul(bias, x, gain)
        return x


class CausalConv2d(nn.Conv2d):
    # input shape: [Batch, Channel, Freq, Time]
    # causal along the time dimension
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding_freq1 = self.dilation[0] * (self.kernel_size[0] - 1) // 2
        self.padding_freq2 = self.dilation[0] * (self.kernel_size[0] - 1) - self.padding_freq1
        self.padding_time = self.dilation[1] * (self.kernel_size[1] - 1)
        #torch.nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.weight.data.normal_(0, 1 / math.sqrt(fan_in))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        x = F.conv2d(
            F.pad(x, (self.padding_time, 0, self.padding_freq1, self.padding_freq2)),
            self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class CausalConvTranspose2d(nn.ConvTranspose2d):
    # input shape: [Batch, Channel, Freq, Time]
    # causal along the time dimension
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        kernel_size, stride, dilation = self.kernel_size, self.stride, self.dilation
        padding_freq = dilation[0] * (kernel_size[0] - 1) + 1 - stride[0]
        if padding_freq % 2 == 1:
            padding_freq += 1
            self.output_padding = (self.output_padding[0] + 1, self.output_padding[1])
        padding_freq = padding_freq // 2
        
        self.pre_padding_time = dilation[1] * (kernel_size[1] - 1) // stride[1]
        padding_time = self.pre_padding_time * stride[1]
        self.padding = (padding_freq, padding_time)
        
        output_padding_time = stride[1] - 1 + padding_time - dilation[1] * (kernel_size[1] - 1)
        self.output_padding = (self.output_padding[0], self.output_padding[1] + output_padding_time)

        #torch.nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.weight.data.normal_(0, 1 / math.sqrt(fan_in))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        x = F.conv_transpose2d(
            F.pad(x, (self.pre_padding_time, 0, 0, 0)),
            self.weight, self.bias, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation
        )
        return x
