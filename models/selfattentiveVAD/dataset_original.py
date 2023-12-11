import soundfile as sf
import numpy as np
import os
import json
import random
import torch
import matplotlib.pyplot as plt

class VADDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, sample_rate=16000, frame_size=10):
        # json_path: json file path
        # frame_size: ms
        super(VADDataset, self).__init__()
        self.json_path = json_path
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        self.frame_size = int(0.001 * frame_size * sample_rate)
        self.wavs = []
        self.labels = []
        self.frames = []
        self.frames_label = []
        self.prepare_files()
        self.collect_frames()

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

            self.wavs.append(wav)
            self.labels.append(temp_label)
        
        print('\nDone!')
    
    def collect_frames(self):
        label_zero = 0
        label_one = 0

        for i, wav in enumerate(self.wavs):
            length = len(wav)
            label = self.labels[i]
            # Setup raw data with zero padding on the end to fit frame size.
            assert len(wav) == len(label), 'Invalid wav length(or label length)' 
            wav = np.concatenate((wav, np.zeros(self.frame_size - (len(wav) % self.frame_size))))
            label = np.concatenate((label, np.zeros(self.frame_size - (len(wav) % self.frame_size))))

            frames = np.split(wav, len(wav) / self.frame_size)
            labels = np.round(np.mean(np.split(label, len(label) / self.frame_size), axis=1)).tolist()

            label_one += sum(labels)
            label_zero += len(labels) - sum(labels)

            self.frames += frames
            self.frames_label += labels

        # augment zeros
        assert len(self.frames) == len(self.frames_label), 'frame processing error!'
        print('\nlabel_zero : {}, label_one : {}'.format(label_zero, label_one))
        print('\nAugment {} zeros to dataset...'.format(int(label_one - label_zero)))

        self.frames += self.noise_frames(int(label_one - label_zero))
        self.frames_label += [np.zeros(1) for i in range(int(label_one - label_zero))]

        assert len(self.frames) == len(self.frames_label), 'Augmentation error!'
        print('Shuffling...')
        self.shuffle()
        print('Final frames :  {}'.format(len(self.frames)), end='\r', flush=True)
        print('\nDone!')

    

    def noise_frames(self, f):
        noise_path = '/home/yhjeon/AGC/data/noise/noise_fileid_'
        noise_frames = []
        for i in range(f):
            if i % 1000 == 0:
                noise_file_num = np.random.randint(0, 60000)
                noise_file = noise_path + str(noise_file_num) + '.wav'
                wav, _ = sf.read(noise_file)
            
            start_time = np.random.randint(0, len(wav) - self.frame_size - 1)
            end_time = start_time + self.frame_size
            noise_frame = wav[start_time : end_time]
            noise_frames.append(noise_frame)

        return noise_frames


    def shuffle(self):
        temp = list(zip(self.frames, self.frames_label))
        random.shuffle(temp)
        self.frames, self.frames_label = zip(*temp)
        self.frames, self.frames_label = list(self.frames), list(self.frames_label)
    

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        temp = np.zeros(2)
        temp[int(self.frames_label[idx])] = 1
        return {'wav' : self.frames[idx].astype(np.float32), 'label' : temp}


if __name__ == '__main__':
    vad = VADDataset(json_path = '/home/yhjeon/AGC/src/vad/train.json', sample_rate = 16000, frame_size = 20)
    vad_loader = torch.utils.data.DataLoader(vad, batch_size=1, shuffle=False)
    for i, data in enumerate(vad_loader):
        if data['label'][0][0]:
            wav0 = data['wav']
        else:
            wav1 = data['wav']
        if 'wav0' in vars() and 'wav1' in vars():
            break
        
        
    plt.plot([i for i in range(len(wav0[0]))], wav0[0], 'k-')
    plt.show()
    plt.savefig('label0.png')

    plt.clf()
    plt.plot([i for i in range(len(wav1[0]))], wav1[0], 'k-')
    plt.show()
    plt.savefig('label1.png')