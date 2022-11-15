import soundfile as sf
import numpy as np
import os
import json
import random
import torch

class VADDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, mode='train', sample_rate = 16000, frame_size = 10):
        '''
        json_path : path to the json file containing meta data of the dataset
        frame_size : in milliseconds
        '''
        super(VADDataset, self).__init__()
        if mode == "train":
            self.json_path = os.path.join(json_path, 'train.json')
        else:
            self.json_path = os.path.join(json_path, 'test.json')        
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        self.frame_size = int(0.001 * frame_size * sample_rate)
        self.wavs = []
        self.labels = []
        self.noise = True
        self.prepare_files()
        self.collect_frames()
        
    def get_track_count(self):
        return len(self.data)
    

    def prepare_files(self, normalize=False):
        progress = 1
        # Convert files to desired format and save raw content.
        for i, file in enumerate(self.data):
            
            print('Processing {0} of {1}'.format(progress, self.get_track_count()), end='\r', flush=True)
            progress += 1


            # Convert file.             
            wav, sr = sf.read(file['path'])
            temp_label = np.zeros_like(wav)
            for item in file['label']:
                l, r = item
                temp_label[int(l):int(r)] = 1
            # Store data.
            self.wavs.append(wav)
            self.labels.append(temp_label)
            
        print('\nDone!')
        
    def collect_frames(self):
        frame_count = 0
        progress = 1
        label_zero = 0
        label_one = 0
        # Calculate number of frames needed.
        for wav in self.wavs:
            frame_count += int((len(wav) + (self.frame_size - (len(wav) % self.frame_size))) / self.frame_size)
            print('Counting frames ({} of {})'.format(progress, self.get_track_count()), end='\r', flush=True)
            progress += 1
        
        self.frames = []
        self.frames_label = []
        progress = 0

        buffer = np.array([])
        label_buffer = np.array([])
        buffer_limit = self.frame_size * 4096
        # Merge frames.
        for i, wav in enumerate(self.wavs):
            length = len(wav)
            label = self.labels[i]
            # Setup raw data with zero padding on the end to fit frame size.
            assert len(wav) == len(label), 'Invalid wav length(or label length)' 
            wav = np.concatenate((wav, np.zeros(self.frame_size - (length % self.frame_size))))
            label = np.concatenate((label, np.zeros(self.frame_size - (length % self.frame_size))))
            
            # Add to buffer.
            buffer = np.concatenate((buffer, wav))
            label_buffer = np.concatenate((label_buffer, label))
            
            # If buffer is not filled up and we are not done, keep filling the buffer up.
            if len(buffer) < buffer_limit and progress + (len(buffer) / self.frame_size) < frame_count:
                continue
            
            # Get frames.
        
            if self.noise == True:
                buffer = self.add_noise(buffer)
    
            
            frames = np.split(buffer, len(buffer) / self.frame_size)
            labels = np.round(np.mean(np.split(label_buffer, len(label_buffer) / self.frame_size), axis=1)).tolist()
            label_one += sum(labels)
            label_zero += len(labels) - sum(labels)
            # Add frames to list.
            self.frames += frames
            self.frames_label += labels
            
            progress += len(frames)
            buffer = np.array([])
            label_buffer = np.array([])
            print('Merging frames {} of {}'.format(progress, frame_count), end='\r', flush=True)

        assert len(self.frames) == len(self.frames_label), 'frame processing error!'
        
        print('\nlabel_zero : {}, label_one : {}'.format(label_zero, label_one))

        print('\nAugment {} zeros to dataset...'.format(int(label_one - label_zero)))
        self.frames += [np.zeros(self.frame_size) if not self.noise else np.random.normal(0, 0.01, self.frame_size) for i in range(int(label_one - label_zero))]
        self.frames_label += [np.zeros(1) for i in range(int(label_one - label_zero))]

        assert len(self.frames) == len(self.frames_label), 'Augmentation error!'
        print('Shuffling...')
        self.shuffle()
        print('Final frames :  {}'.format(len(self.frames)), end='\r', flush=True)
        print('\nDone!')
    
    def shuffle(self):
        temp = list(zip(self.frames, self.frames_label))
        random.shuffle(temp)
        self.frames, self.frames_label = zip(*temp)
        self.frames, self.frames_label = list(self.frames), list(self.frames_label)
    

    def add_noise(self, buffer):
        mu, sigma = 0, 0.00975
        noise = np.random.normal(mu, sigma, buffer.shape)
        buffer += noise
        
        return buffer 

            
    def add_silence(self):
        idx = 0
        total_silence_length = 0
        length = len(self.frames)
        slices = []

        while idx + self.slice_min < length:
            slice_idx = (idx, idx + np.random.randint(self.silce_min, self.slice_max + 1))
            slices.append(slice_idx)
            idx = slice_idx[1]
        
        slices[-1] = (slices[-1][0], length)

        while total_silence_length + self.slice_min < length:
            silence_len = np.random.randint(self.slice_min, self.slice_max + 1)
            slice_idx = (silence_len, silence_len)
            slices.append(slice_idx)
            total_silence_length += silence_len
        
        total = total_silence_length + length

        idx = 0

        for slice in slices:
            if slice[0] == slice[1]:
                frames = np.zeros((slice[0], self.frame_size))
                labels = np.zeros(slice[0])
            else:
                frames = self.frames[slice[0] : slice[1]]
                labels = self.labels[slice[0] : slice[1]]

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        temp = np.zeros(2)
        temp[int(self.frames_label[idx])] = 1
        
        return {'wav' : self.frames[idx].astype(np.float32), 'label' : temp}


if __name__=='__main__':
    vad = VADDataset(json_path = '/home/jhkim21/IITP/2022/AGC/AGC_IITP/src/VAD', mode='train', sample_rate=16000, frame_size=20)
