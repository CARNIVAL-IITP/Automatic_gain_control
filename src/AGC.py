import soundfile as sf
import numpy as np
from VAD.model import VAD
import torch
import time
import os
import matplotlib.pyplot as plt

FRAME_SIZE = 25
SAMPLE_RATE = 16000
SAMPLE_LENGTH = 10
FRAME_LENGTH = FRAME_SIZE * SAMPLE_RATE // 1000

class AGC():
    def __init__(self, smoothing = 0.3, neural_vad_model = None):
        self.decibels = [-0., -0.5, -1., -1.5, -2., -2.5, -3. , -3.5, -4., -4.5, -5., -5.5, -6., -6.5, -7., -7.5, -8.]
        self.gain =     [-5 , -4  , -4 , -3.5,  -3,    -2.5,  -2,   -1,  -0.5,   -0.5,  0,  0.5,  1,   2,  3,  4,   5]
        assert len(self.decibels) == len(self.gain), 'decibel - gain mapping is invalid'
        self.smoothing = smoothing
        self.vad_threshold = 1e-5 #temporary
        self.prev_gain = 0
        self.model = neural_vad_model
        self.vad = None
        self.peaks = []

    def cal_power(self):
        frame = np.trim_zeros(self.frame)
        frame[frame == 0] = 1e-7
        dbfs = np.log2((np.abs(frame)))
        return np.mean(dbfs)

    def cal_gain(self):
        db = self.cal_power()
        if not self.vad:
            return 0
        for idx, decibel in enumerate(self.decibels):
            if db > decibel:
                return self.gain[idx]
        return self.gain[-1]

    def process(self, frame):
        self.frame = frame
        #self.VAD()
        in_buffer = torch.Tensor(frame).unsqueeze(0)
        self.vad = np.argmax(self.model(in_buffer).detach().cpu().numpy().squeeze())
        gain = self.cal_gain()
        gain = self.smoothing * gain + (1-self.smoothing) * self.prev_gain

        self.prev_gain = gain

        return 2 ** gain

if __name__=='__main__':
    path = '/home/jhkim21/IITP/2022/AGC/AGC_IITP/sample/raw/female.wav'
    sample, sr = sf.read(path)
    '''
    plt.ylim(-1.0, 1.0)
    plt.plot(sample, color='C0')
    plt.rcParams['figure.figsize'] = (200, 10)
    plt.savefig(os.path.join('/home/jhkim21/IITP/2022/AGC/AGC_IITP/sample/enhanced/', path.split('/')[-1].split('.')[0] + '_raw.png'), dpi=200)
    '''
    vad = VAD.load_model('/home/jhkim21/IITP/2022/AGC/AGC_IITP/src/VAD/logs/ckpts/epoch20.pth.tar')
    vad.eval()
    agc = AGC(0.1,  neural_vad_model=vad)
    sample_size = len(sample) // FRAME_LENGTH * FRAME_LENGTH
    segment = sample_size // FRAME_LENGTH 
    sample = sample[:sample_size]
    res = np.zeros(sample_size)
    vad_plot = np.zeros(sample_size)
    avg_time = 0.0
    gain_list = np.zeros(sample_size)
    for i in range(segment):
        start = time.time()
        in_buffer = sample[i*FRAME_LENGTH : (i+1) * FRAME_LENGTH]
        gain = agc.process(in_buffer)
        vad_plot[i*FRAME_LENGTH : (i+1) * FRAME_LENGTH] = int(agc.vad)
        res[i*FRAME_LENGTH : (i+1) * FRAME_LENGTH] = in_buffer * gain
        gain_list[i*FRAME_LENGTH : (i+1) * FRAME_LENGTH] = gain
        avg_time += time.time() - start
    
    print('processing time : {}'.format(avg_time/segment))
    #plt.ylim(-1.0, 1.0)
    plt.plot(gain_list, color='C0')
    plt.rcParams['figure.figsize'] = (200, 10)
    #plt.plot(vad_plot)
    plt.savefig(os.path.join('/home/jhkim21/IITP/2022/AGC/AGC_IITP/sample/enhanced/', path.split('/')[-1].split('.')[0] + '_gain.png'), dpi=200)
    #sf.write(os.path.join('/home/jhkim21/IITP/2022/AGC/AGC_IITP/sample/enhanced/', path.split('/')[-1]), res, 16000)