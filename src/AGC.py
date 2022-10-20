import soundfile as sf
import numpy as np
from VAD.model import VAD
import torch
import time

FRAME_SIZE = 40
SAMPLE_RATE = 16000
SAMPLE_LENGTH = 10
FRAME_LENGTH = FRAME_SIZE * SAMPLE_RATE // 1000

class AGC():
    def __init__(self, smoothing = 0.3, neural_vad_model = None):
        self.decibels = [-0., -0.5, -1., -1.5, -2., -2.5, -3. , -3.5, -4., -4.5, -5., -5.5, -6., -6.5, -7., -7.5, -8.]
        self.gain =     [-5 , -4  , -4 , -3.5,  0 ,    0,  0,      0,  0,   0.5,  0.5,   1,  2 ,    3,  3 ,  3.5,   4]
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
        #gain = self.smoothing * gain + (1-self.smoothing) * self.prev_gain

        #self.prev_gain = gain

        return 2 ** gain

if __name__=='__main__':
    sample, sr = sf.read('/home/jhkim21/IITP/2022/AGC/AGC_IITP/sample/raw/nearend_speech_fileid_1.wav')
    res = np.zeros(len(sample))
    vad = VAD.load_model('/home/jhkim21/IITP/2022/AGC/AGC_IITP/src/VAD/logs/ckpts/epoch20.pth.tar')
    agc = AGC(0.1,  neural_vad_model=vad)
    segment = SAMPLE_LENGTH * 1000 // FRAME_SIZE

    start = time.time()
    for i in range(segment):
        in_buffer = sample[i*FRAME_LENGTH : (i+1) * FRAME_LENGTH]
        gain = agc.process(in_buffer)
        res[i*FRAME_LENGTH : (i+1) * FRAME_LENGTH] = in_buffer * gain
    
    print('processing time : {}'.format(time.time() - start))

    sf.write('/home/jhkim21/IITP/2022/AGC/AGC_IITP/sample/enhanced/nearend_speech_fileid_1_with_dnn_vad_wo_smoothing.wav', res, 16000)