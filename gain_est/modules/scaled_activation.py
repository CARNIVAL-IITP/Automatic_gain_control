import warnings

import torch
import torch.nn as nn
from torch import Tensor

# Some activations has keyword arguments which affects scale.
# For example, negative_slope of LeakyReLU
# Therefore, if a keyword argument exists, compare its value.
scale_dicts = {
    "GELU": {"scale": 1.7015043497085571},
    "GLU": {"scale": 1.8484294414520264},
    "LogSigmoid": {"scale": 1.9193484783172607},
    "LogSoftmax": {"scale": 1.0002083778381348},
    "ReLU": {"scale": 1.7139588594436646},
    "ReLU6": {"scale": 1.7131484746932983},
    "SELU": {"scale": 1.0008515119552612},
    "Sigmoid": {"scale": 4.803835391998291},
    "SiLU": {"scale": 1.7881293296813965},
    "Softsign": {"scale": 2.338853120803833},
    "Tanh": {"scale": 1.5939117670059204},

    "CELU": {
        "key": "alpha",
        "scale": {1.0: 1.270926833152771}
    },
    "ELU": {
        "key": "alpha",
        "scale": {1.0: 1.2716004848480225}
    },
    "LeakyReLU": {
        "key": "negative_slope",
        "scale": {0.01: 1.70590341091156}
    },
    'Softplus': {
        "key": "beta",
        "scale": {1.0: 1.9203323125839233}
    }
}


class ScaledActivation(nn.Module):
    def __init__(self, activation: str, *args, **kwargs):
        super().__init__()
        self.activation = getattr(nn, activation)(*args, **kwargs)
        self.inplace = getattr(self.activation, "inplace", False)

        global scale_dicts
        scale = None
        if activation in scale_dicts:
            scale_dict = scale_dicts[activation]
            if "key" not in scale_dict:
                scale = scale_dict["scale"]
            else:
                key = getattr(self.activation, scale_dict["key"])
                for k, v in scale_dict["scale"].items():
                    if k == key:
                        scale = v
                        break
                if scale is None:
                    scale = calculate_scale(self.activation)
                    scale_dict["scale"][key] = scale
                    print(f"Scale of {activation}({scale_dict['key']}={key}): {scale}")
        else:
            warnings.warn(
                f"ScaledActivation for {activation} not found! Scale will be calculated empirically. Another option is to manually add {activation} to the 'modules/scaled_activation.py'",
                RuntimeWarning
            )
            scale = calculate_scale(self.activation)
            scale_dicts[activation]["scale"] = scale
            print(f"Scale of {activation}: {scale}")
        self.register_buffer("scale", torch.ones(1) * scale)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(x)
        #return x.mul_(self.scale) if self.inplace else x.mul(self.scale)
        return x.mul(self.scale)


@torch.no_grad()
def calculate_scale(activation: torch.nn.Module) -> float:
    x = torch.randn(2048, 2048)
    x = (x - x.mean()) / x.var(unbiased=False)
    y = activation(x)
    scale = 1 * torch.rsqrt(y.var(dim=1, unbiased=False).mean())
    return scale.item()


if __name__=="__main__":
    sa1 = ScaledActivation("ELU", alpha=0.5, inplace=True)
    print(sa1.scale)
    sa2 = ScaledActivation("ELU", alpha=0.5, inplace=True)
    print(sa2.scale)