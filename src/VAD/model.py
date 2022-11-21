import torch
import torch.nn as nn
from .utils import stft

import torch.nn as nn
from torch.nn import GRU
import torch.nn.functional as F
from torch.autograd import Variable


class VAD(nn.Module):
    
    def __init__(self, n_fft, hidden_size, win_len, hop_len):
        super(VAD, self).__init__()
        self.n_fft = n_fft
        input_size = self.n_fft // 2 + 1
        self.hidden_size = hidden_size
        self.win_len = win_len
        self.hop_len = hop_len
        self.relu = nn.ReLU()
        self.rnn = GRU(input_size=input_size, hidden_size=input_size//2, num_layers=1, batch_first=True)
        self.lin = nn.Linear(input_size//2, 2)    
        self.softmax = nn.Softmax(dim=2)
    

    
    def forward(self, x):
        spec = stft(x, n_fft=self.n_fft, hop_size=self.hop_len, win_size = self.win_len, center=False)
        spec = spec.permute(0, 2, 1)
        spec, _ = self.rnn(spec)
        spec = self.lin(spec)

        spec = self.softmax(spec).squeeze(1)
        return spec

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc:storage)
        model = cls.load_model_from_package(package)

        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['n_fft'], package['hidden_size'], package['win_len'], package['hop_len'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, epoch, tr_loss = None, cv_loss = None):
        package = {'n_fft' : model.n_fft,
                'hidden_size' : model.hidden_size, 
                'win_len' : model.win_len, 
                'hop_len' : model.hop_len, 
                'state_dict' : model.state_dict(),
                'epoch' : epoch}
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss

        return package



if __name__=='__main__':
    a = torch.randn((1, 64000))
    vad = VAD(320, 40, 160, 80)
    stft_tag = vad(a)
    print(stft_tag.shape)
