import torch
import torch.nn as nn
import torch.nn.functional as F
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

class AGC_STFT_GRU(nn.Module):
    
    def __init__(self, n_fft, hidden_size, win_len, hop_len):
        super(AGC_STFT_GRU, self).__init__()
        self.n_fft = n_fft
        input_size = self.n_fft // 2 + 1
        self.hidden_size = hidden_size
        self.win_len = win_len
        self.hop_len = hop_len
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.lin = nn.Linear(hidden_size, 1)
        self.upsample = nn.Upsample(scale_factor=hop_len, mode='nearest')
        self.window = torch.hann_window(win_len, periodic=True)
    

    
    def forward(self, x, h_0 = None):
        
        if h_0 == None:
            distorted = x
        else:
            distorted = x[..., -self.hop_len:]
        if h_0 == None:
            x = F.pad(x, (self.win_len - 1 - ((x.shape[1] - 1) % self.hop_len) , 0, 0, 0))
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len, center=False, window=self.window.to(x.device))
        x_real, x_imag = x[:, :, :, 0], x[:, :, :, 1]
        x = torch.sqrt(torch.square(x_real) + torch.square(x_imag))
        x = x.permute(0, 2, 1)

        if h_0 == None:
            x, hidden = self.gru(x)
        else:
            x, hidden = self.gru(x, h_0)
        x = self.relu(x)
        x = self.lin(x)
        x = x[:, :, 0]
        x = x.unsqueeze(1)
        x = self.upsample(x)
        gain = x.squeeze(1)
        gain = gain[:, :distorted.shape[1]]
        gain = torch.exp2(gain)
        
        estimate = distorted * gain
        return estimate, hidden

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
    def serialize(model, optimizer, epoch, tr_loss = None, cv_loss = None):
        package = {'win_len' : model.win_len,
                   'hop_len' : model.hop_len,
                   'n_fft' : model.n_fft,
                'hidden_size' : model.hidden_size,
                'state_dict' : model.state_dict(),
                'optim_dict' : optimizer.state_dict(),
                'epoch' : epoch}
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss

        return package
    
class AGC_STFT_GRU_smooth(nn.Module):
    
    def __init__(self, n_fft, hidden_size, win_len, hop_len):
        super(AGC_STFT_GRU, self).__init__()
        self.n_fft = n_fft
        input_size = self.n_fft // 2 + 1
        self.hidden_size = hidden_size
        self.win_len = win_len
        self.hop_len = hop_len
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.lin = nn.Linear(hidden_size, 1)
        self.upsample = nn.Upsample(scale_factor=hop_len, mode='nearest')
        self.window = torch.hann_window(win_len, periodic=True)
    

    
    def forward(self, x, h_0 = None):
        
        if h_0 == None:
            distorted = x
        else:
            distorted = x[..., -self.hop_len:]
        if h_0 == None:
            x = F.pad(x, (self.win_len - 1 - ((x.shape[1] - 1) % self.hop_len) , 0, 0, 0))
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len, center=False, window=self.window.to(x.device))
        x_real, x_imag = x[:, :, :, 0], x[:, :, :, 1]
        x = torch.sqrt(torch.square(x_real) + torch.square(x_imag))
        x = x.permute(0, 2, 1)

        if h_0 == None:
            x, hidden = self.gru(x)
        else:
            x, hidden = self.gru(x, h_0)
        x = self.relu(x)
        x = self.lin(x)
        x = x[:, :, 0]
        x = x.unsqueeze(1)
        x = self.upsample(x)
        gain = x.squeeze(1)
        gain = gain[:, :distorted.shape[1]]
        gain = torch.exp2(gain)
        
        estimate = distorted * gain
        return estimate, hidden

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
    def serialize(model, optimizer, epoch, tr_loss = None, cv_loss = None):
        package = {'win_len' : model.win_len,
                   'hop_len' : model.hop_len,
                   'n_fft' : model.n_fft,
                'hidden_size' : model.hidden_size,
                'state_dict' : model.state_dict(),
                'optim_dict' : optimizer.state_dict(),
                'epoch' : epoch}
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss

        return package



if __name__=='__main__':
    a = torch.randn((1, 512))
    AGC = AGC_STFT_GRU(512, 40, 512, 128)
    estimate, _ = AGC(a)
    print(estimate.shape)
