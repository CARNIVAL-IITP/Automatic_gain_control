import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer
import time

class SelfAttentiveVAD(nn.Module):
    def __init__(self, win_len=480, hop_len=160, feature_size=32, d_model=32, num_layers=1, dropout = 0):
        super(SelfAttentiveVAD, self).__init__()
        
        d_ff = d_model * 4

        self.win_len = win_len
        self.hop_len = hop_len
        self.feature_size = feature_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.feature_layer = nn.Conv1d(1, feature_size, win_len, hop_len, bias=False)
        self.input_layer = nn.Sequential(
            nn.Linear(feature_size, d_model),
            transformer.SinusoidalPositionalEncoding(encoding_size=d_model, initial_length=10),
            nn.Dropout(dropout),
        )
        self.encoder = transformer.TransformerEncoder(num_layers, d_model, d_ff, n_heads=1, dropout=dropout)
        self.classifier = nn.Linear(d_model, 2)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()

    def forward(self, waveform):
        # input: [B, N]
        x = torch.unsqueeze(waveform, 1)
        x = self.relu(self.feature_layer(x))
        x = torch.transpose(x, 1, 2)
        x = self.input_layer(x) 
        x = self.encoder(x)
        x = self.classifier(x)
        x = self.softmax(x)
        x = torch.transpose(x, 0, 2)[0]
        x = torch.transpose(x, 0, 1)
        return x

    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc:storage)
        model = cls.load_model_from_package(package)

        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['win_len'], package['hop_len'], package['feature_size'], package['d_model'], package['num_layers'], package['dropout'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss = None, cv_loss = None):
        package = {'win_len' : model.win_len,
                   'hop_len' : model.hop_len,
                'feature_size' : model.feature_size,
                'd_model' : model.d_model,
                'num_layers' : model.num_layers,
                'dropout' : model.dropout, 
                'state_dict' : model.state_dict(),
                'optim_dict' : optimizer.state_dict(),
                'epoch' : epoch}
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss

        return package


class SelfAttentiveVAD_Normalize(nn.Module):
    def __init__(self, win_len=480, hop_len=160, feature_size=32, d_model=32, num_layers=1, dropout = 0):
        super(SelfAttentiveVAD_Normalize, self).__init__()
        
        d_ff = d_model * 4

        self.win_len = win_len
        self.hop_len = hop_len
        self.feature_size = feature_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.feature_layer = nn.Conv1d(1, feature_size, win_len, hop_len, bias=False)
        self.input_layer = nn.Sequential(
            nn.Linear(feature_size, d_model),
            transformer.SinusoidalPositionalEncoding(encoding_size=d_model, initial_length=10),
            nn.Dropout(dropout),
        )
        self.encoder = transformer.TransformerEncoder(num_layers, d_model, d_ff, n_heads=1, dropout=dropout)
        self.classifier = nn.Linear(d_model, 2)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()

    def forward(self, waveform):
        # input: [B, N]
        x = waveform
        x = x / torch.max(x, dim=1, keepdims=True)[0]
        x = torch.unsqueeze(x, 1)
        x = self.relu(self.feature_layer(x))
        x = torch.transpose(x, 1, 2)
        x = self.input_layer(x) 
        x = self.encoder(x)
        x = self.classifier(x)
        x = self.softmax(x)
        x = torch.transpose(x, 0, 2)[0]
        x = torch.transpose(x, 0, 1)
        return x

    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc:storage)
        model = cls.load_model_from_package(package)

        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['win_len'], package['hop_len'], package['feature_size'], package['d_model'], package['num_layers'], package['dropout'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss = None, cv_loss = None):
        package = {'win_len' : model.win_len,
                   'hop_len' : model.hop_len,
                'feature_size' : model.feature_size,
                'd_model' : model.d_model,
                'num_layers' : model.num_layers,
                'dropout' : model.dropout, 
                'state_dict' : model.state_dict(),
                'optim_dict' : optimizer.state_dict(),
                'epoch' : epoch}
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss

        return package


class SelfAttentiveVAD_Normalize_Convoutput(nn.Module):
    def __init__(self, win_len=480, hop_len=160, feature_size=16, d_model=32, num_layers=1, dropout = 0):
        super(SelfAttentiveVAD_Normalize_Convoutput, self).__init__()
        
        d_ff = d_model * 4

        self.win_len = win_len
        self.hop_len = hop_len
        self.feature_size = feature_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.feature_layer = nn.Conv1d(1, feature_size, win_len, hop_len, bias=False)
        self.input_layer = nn.Sequential(
            nn.Linear(feature_size, d_model),
            transformer.SinusoidalPositionalEncoding(encoding_size=d_model, initial_length=10),
            nn.Dropout(dropout),
        )
        self.encoder = transformer.TransformerEncoder(num_layers, d_model, d_ff, n_heads=1, dropout=dropout)
        self.classifier = nn.Linear(d_model, 2)
        self.softmax = nn.Softmax(dim=2)
        self.output_conv_layer = nn.Conv1d(1, 1, 11, 1, padding=5)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, waveform):
        # input: [B, N]
        start = time.time()
        x = waveform
        #x = x / torch.max(x, dim=1, keepdims=True)[0]
        x = torch.unsqueeze(x, 1)
        x = self.relu(self.feature_layer(x))
        x = torch.transpose(x, 1, 2)
        x = self.input_layer(x) 
        x = self.encoder(x)
        x = self.classifier(x)
        x = self.softmax(x)
        x = torch.transpose(x, 0, 2)[0]
        x = torch.transpose(x, 0, 1)
        x = torch.unsqueeze(x, 1)
        x = self.output_conv_layer(x - 0.5)
        x = x.squeeze(dim=1)
        x = self.sigmoid(x)

        return x

    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc:storage)
        model = cls.load_model_from_package(package)

        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['win_len'], package['hop_len'], package['feature_size'], package['d_model'], package['num_layers'], package['dropout'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss = None, cv_loss = None):
        package = {'win_len' : model.win_len,
                   'hop_len' : model.hop_len,
                'feature_size' : model.feature_size,
                'd_model' : model.d_model,
                'num_layers' : model.num_layers,
                'dropout' : model.dropout, 
                'state_dict' : model.state_dict(),
                'optim_dict' : optimizer.state_dict(),
                'epoch' : epoch}
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss

        return package

if __name__ == '__main__':
    model = SelfAttentiveVAD()
    features = torch.randn(size=[1, 48000])
    start = time.time()
    output = model(features)
    end = time.time()
    print(output.shape, end - start)