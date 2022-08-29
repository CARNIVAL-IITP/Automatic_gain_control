import soundfile as sf
import numpy as np

class AGC():
    def __init__(self, smoothing = 0.3):
        self.decibels = [-0., -0.5, -1., -1.5, -2., -2.5, -3., -3.5, -4., -4.5, -5., -5.5, -6., -6.5, -7., -7.5, -8.]
        self.gain = [-3, -2, -2, -2, -1, -1, -1, -1, 0, 0, 0, 1, 2, 3.5, 4, 4.5, 5.5]
        assert len(self.decibels) == len(self.gain), 'decibel - gain mapping is invalid'
        self.smoothing = smoothing
        self.vad_threshold = 1e-5 #temporary
        self.prev_gain = 0
        self.vad = None
        self.peaks = []

    def cal_power(self):
        frame = np.trim_zeros(self.frame)
        frame[frame == 0] = 1e-7
        dbfs = np.log2((np.abs(frame)))
        return np.mean(dbfs)

    ######Temporary functions(will be deprecated after DNN VAD is implemented)#####
    def find_peak(self):
        self.peaks.append(np.max(np.power(self.frame, 2)))

    def VAD(self):
        self.vad = np.mean(self.peaks[-10:]) > self.vad_threshold
        
    ################################################################################

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
        self.peak()
        self.VAD()
        gain = self.cal_gain()
        gain = self.smoothing * gain + (1-self.smoothing) * self.prev_gain

        self.prev_gain = gain

        return 2 ** gain

if __name__=='__main__':
    res = np.zeros(160000)
    sample, sr = sf.read('/home/jhkim21/IITP/2022/AGC/AGC_IITP/sample/raw/nearend_speech_fileid_1.wav')
    agc = AGC(0.1)
    for i in range(1000):
        in_buffer = sample[i*160 : (i+1) * 160]
        gain = agc.process(in_buffer)
        res[i*160 : (i+1) * 160] = in_buffer * gain

    sf.write('/home/jhkim21/IITP/2022/AGC/AGC_IITP/sample/enhanced/nearend_speech_fileid_1.wav', res, 16000)