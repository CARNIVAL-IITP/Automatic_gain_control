import soundfile as sf
import numpy as np
import os
import json
import random
import torch
import matplotlib.pyplot as plt
import config

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

parser = config.parser
args= parser.parse_args("")

class AGCDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, sample_rate=16000, wav_len=48000, win_len=480, hop_len=160):
        # json_path: json file path
        # frame_size: ms
        super(AGCDataset, self).__init__()

        self.json_path = json_path
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        self.win_len = win_len
        self.frame_size = hop_len
        self.wav_len = wav_len
        self.distorted = []
        self.original = []


        self.prepare_files()

    def get_track_count(self):
        return len(self.data)

    def prepare_files(self, normalize=False):
        # Convert files to desired format and save raw content.
        for i, file in enumerate(self.data):

            wav_d, _ = sf.read(file['path'])
            wav_o, _ = sf.read(file['label'])

            self.distorted.append(wav_d)
            self.original.append(wav_o)

        
        print('\nDone!')

    
    def __len__(self):
        return len(self.distorted)
    
    def __getitem__(self, idx):
        return {'distorted' : self.distorted[idx].astype(np.float32), 'original' : self.original[idx].astype(np.float32)}
    



def main():
    dataset = AGCDataset('/home/yhjeon/AGC/data/agc_1/valid.json', args.sample_rate, args.wav_len, args.win_len, args.hop_len)
    print(dataset.__getitem__(0)['distorted'].shape, dataset.__getitem__(0)['original'].shape)



if __name__ == '__main__':
    main()