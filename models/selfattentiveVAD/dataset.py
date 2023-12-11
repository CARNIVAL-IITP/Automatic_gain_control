import soundfile as sf
import numpy as np
import os
import json
import random
import torch
import matplotlib.pyplot as plt
import config

parser = config.parser
args= parser.parse_args("")

class VADDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, sample_rate=16000, wav_len=48000, win_len=480, hop_len=160):
        # json_path: json file path
        # frame_size: ms
        super(VADDataset, self).__init__()

        assert win_len % hop_len == 0, 'win_len should be integer multiple of hop_len'

        self.json_path = json_path
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        self.win_len = win_len
        self.frame_size = hop_len
        self.wav_len = wav_len
        self.wavs = []
        self.temp_labels = []
        self.labels = []
        self.label_zero = 0
        self.label_one = 0

        # label_zero와 label_one의 비율에 따라 loss에 가중치를 부여할 것.

        self.prepare_files()
        self.frame_label()

    def get_track_count(self):
        return len(self.data)

    def prepare_files(self, normalize=False):
        # Convert files to desired format and save raw content.
        for i, file in enumerate(self.data):
            if i % 100 == 0:
                print('Processing {0} of {1}'.format(i + 1, self.get_track_count()), end='\r', flush=True)

            wav, sr = sf.read(file['path'])
            temp_label = np.zeros_like(wav)
            for item in file['label']:
                l, r = item[0], item[1]
                temp_label[l:r] = 1

            s = 0
            while s + self.wav_len < len(wav):
                self.wavs.append(wav[s:s+self.wav_len])
                self.temp_labels.append(temp_label[s:s+self.wav_len])
                s += self.wav_len
            
            self.wavs.append(np.concatenate([wav[s:], np.zeros(self.wav_len - ((len(wav) - 1) % self.wav_len + 1))]))
            self.temp_labels.append(np.concatenate([temp_label[s:], np.zeros(self.wav_len - ((len(temp_label) - 1) % self.wav_len + 1))]))

        
        print('\nDone!')
    
    def frame_label(self):

        for i, wav in enumerate(self.wavs):
            label = self.temp_labels[i]
            # Setup raw data with zero padding on the end to fit frame size.
            assert len(wav) == len(label), 'Invalid wav length(or label length)' 
            wav = np.concatenate((wav, np.zeros(self.frame_size - ((len(wav) - 1) % self.frame_size + 1))))
            label = np.concatenate((label, np.zeros(self.frame_size - ((len(wav) - 1) % self.frame_size + 1))))

            labels = (np.mean(np.split(label, len(label) / self.frame_size), axis=1))

            win_frames = self.win_len // self.frame_size
            total_labels = np.zeros(len(labels) - win_frames + 1)
            for j in range(win_frames):
                total_labels += labels[j : j + len(labels) - win_frames + 1]
            
            total_labels = np.round(total_labels / win_frames)

            self.label_one += sum(total_labels)
            self.label_zero += len(total_labels) - sum(total_labels)
            
            self.labels.append(total_labels)
    
    def __len__(self):
        return len(self.wavs)
    
    def __getitem__(self, idx):
        return {'wav' : self.wavs[idx].astype(np.float32), 'label' : self.labels[idx].astype(np.float32)}
    



def main():
    dataset = VADDataset('/home/yhjeon/AGC/src/vad/train.json', args.sample_rate, args.wav_len, args.win_len, args.hop_len)
    print(dataset.__getitem__(0)['wav'].shape, dataset.__getitem__(0)['label'].shape)



if __name__ == '__main__':
    main()