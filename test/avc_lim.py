from math import floor
import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt
import librosa

class AGC_module():
    def __init__(self):
        # envelope/floor/gain_smooth factor : 0 < x < 1
        self.frame_gain = 0
        self.curr_slow_env = 0
        self.prev_slow_env = 0
        self.curr_fast_env = 0
        self.prev_fast_env = 0

        self.curr_amp_level = 0
        self.prev_amp_level = 0
        self.vad = 0
        self.curr_floor = 0
        self.prev_floor = 0

        self.curr_gain = 1
        self.prev_gain = 1
        self.gmod=0


    def slow_envelope_tracker(self, frame):
        self.frame_gain = sum(map(lambda x:x*x, frame))
        self.curr_slow_env = self.prev_slow_env
        #r_s = 0.199 if self.frame_gain > self.prev_slow_env else 0.1997
        r_s = 0.999 if self.frame_gain > self.prev_slow_env else 0.995 # rising : 0.999 / falling : 0.9997
        self.curr_slow_env = (1 - r_s) *self.frame_gain + r_s * self.prev_slow_env
    
    def fast_envelope_tracker(self):
        self.curr_fast_env = self.prev_fast_env
        r_s = 0.992 if self.frame_gain > self.prev_fast_env else 0.990 # rising : 0.992 / falling : 0.9990
        self.curr_fast_env = (1 - r_s) * self.frame_gain + r_s * self.prev_fast_env

    def voiceactivitydetection(self):
        self.vad = 1 if self.curr_fast_env > max(self.curr_slow_env, 1e-8) else 0

    def track_amp(self):
        self.prev_amp_level = self.curr_amp_level
        r_p = 0.995 if self.curr_fast_env > self.prev_amp_level else 0.990
        self.curr_amp_level = (1 - r_p) * self.curr_fast_env + r_p * self.prev_amp_level if self.vad else self.prev_amp_level

    def gain_update(self, vol):
        self.prev_gain = self.curr_gain
        self.gmod = 1.005 if self.curr_amp_level * self.prev_gain < (400 / vol) else 0.995
        self.curr_gain = self.gmod * self.prev_gain if self.vad else self.prev_gain

    def process(self, frame, vol):
        self.slow_envelope_tracker(frame)
        self.fast_envelope_tracker()
        self.voiceactivitydetection()
        self.track_amp()
        self.gain_update(vol) 
        
        return self.curr_gain