import os
import re
import math
import random
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import librosa
from scipy.interpolate import interp1d
from tqdm import tqdm
try:
    import parselmouth  # used for pitch extractor
except ImportError:
    pass

from functional import stft, spec_to_mel, global_masking_threshold


class TensorDict(dict):
    '''To set pin_memory=True in DataLoader, need to use this class instead of dict.'''
    def pin_memory(self):
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                self[key] = value.pin_memory()


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, hp, keys, textprocessor=None, mode="train", batch_size=1, verbose=True):
        super().__init__()
        self.wav_dir = hp.wav_dir
        self.data_dir = hp.data_dir
        self.segment_size = None if mode == "infer" else getattr(hp, "segment_size", None)
        
        self.textprocessor = textprocessor
        self.keys = keys
        self.hp = hp

        filelist = hp.filelists[mode]

        self.wav_idx, self.text_idx = [], []
        with open(filelist, encoding="utf-8") as txt:
            wav_text_tag = [l.strip().split("|") for l in txt.readlines()]
        if mode=="infer":
            wav_text_tag = wav_text_tag[:hp.num_infer]
        for wtt in wav_text_tag:
            self.wav_idx.append(re.sub(f"\.{hp.extension}$", "", wtt[0]))
            self.text_idx.append(wtt[1:])
        
        filter = False if mode == "infer" else hp.filter[mode]

        if filter:
            self.batch_size = batch_size
            # 1. filter out very short or long utterances
            wav_idx_ = []
            text_idx_ = []
            wav_len = []
            min_length = getattr(hp, "min_length", 0)
            max_length = getattr(hp, "max_length", float("inf"))
            for i in tqdm(range(len(self.wav_idx)), desc=f"Filtering {mode} dataset", dynamic_ncols=True, leave=False, disable=(not verbose)):
                wav_length = self.get_wav_length(i)
                if min_length < wav_length < max_length:
                    wav_idx_.append(self.wav_idx[i])
                    text_idx_.append(self.text_idx[i])
                    wav_len.append(wav_length)
            if verbose:
                print(f'{mode} dataset filtered: {len(wav_idx_)}/{len(self.wav_idx)}')

            # 2. group wavs with similar lengths in a same batch
            idx_ascending = np.array(wav_len).argsort()
            self.wav_idx = np.array(wav_idx_)[idx_ascending]
            self.text_idx = np.array(text_idx_)[idx_ascending, :]
        else:
            self.batch_size = 1
            self.wav_idx = np.array(self.wav_idx)
            self.text_idx = np.array(self.text_idx)

    def get_wav_length(self, idx: int) -> float:
        raise NotImplementedError()

    def shuffle(self, seed: int):
        rng = np.random.default_rng(seed)   # deterministic random number generator
        bs = self.batch_size
        len_ = len(self.wav_idx) // bs
        idx_random = np.arange(len_)
        rng.shuffle(idx_random)
        self.wav_idx[:len_ * bs] = self.wav_idx[:len_ * bs].reshape((len_, bs))[idx_random, :].reshape(-1)
        self.text_idx[:len_ * bs, :] = self.text_idx[:len_ * bs, :].reshape((len_, bs, -1))[idx_random, :, :].reshape(len_ * bs, -1)
    
    def get_text(self, idx: int) -> torch.LongTensor:
        # text shape: [text_len]
        text = self.text_idx[idx]
        text = self.textprocessor(text)
        return torch.LongTensor(text)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.wav_idx)


class Dataset(_Dataset):
    def load_wav(self, idx: int, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        wav_dir = os.path.join(self.wav_dir, f"{self.wav_idx[idx]}.{self.hp.extension}")
        wav, sr = librosa.core.load(wav_dir, sr=sr)
        if self.hp.trim:
            wav, _ = librosa.effects.trim(wav, top_db=self.hp.trim_db, frame_length=800, hop_length=200)
        return wav, sr
    
    def get_wav_length(self, idx: int) -> float:
        wav, sr = self.load_wav(idx)
        return len(wav) / sr

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = {}
        if "filename" in self.keys:
            data["filename"] = self.wav_idx[idx]
        
        if "text" in self.keys:
            text = self.get_text(idx)
            data["text"] = text     # shape: [num_mels, mel_len]
        if "text_len" in self.keys:
            data["text_len"] = text.size(0)

        wav, _ = self.load_wav(idx, sr=self.hp.sampling_rate)
        wav = 0.99*wav / np.abs(wav).max()
        wav = torch.from_numpy(wav).to(torch.float32)

        # pad 0 & make wav_len = multiple of hop_size
        if self.segment_size is None:
            wav_len = wav.size(0)
            hop_size = getattr(self.hp, "hop_size", 1)
            discard_len = wav_len - wav_len // hop_size * hop_size
            #wav = F.pad(wav, (2400, 2400 - discard_len))
            if discard_len > 0:
                wav = wav[:-discard_len]
            assert wav.size(0) % hop_size == 0
        else:
            if wav.size(0) >= self.segment_size:
                #start_idx = torch.randint(0, wav.size(0) - self.segment_size + 1, (1,))
                start_idx = random.randint(0, wav.size(0) - self.segment_size)  # 0 <= start_idx <= wav.size(-1) - segment_size
                wav = wav[start_idx:start_idx+self.segment_size]
            else:
                wav = F.pad(wav, (0, self.segment_size - wav.size(0)), value=0)

        if "wav" in self.keys:
            data["wav"] = wav       # shape: [wav_len]
        if "wav_len" in self.keys:
            data["wav_len"] = wav.size(0)
        
        if "mel" in self.keys or "mel_loss" in self.keys or "mask" in self.keys:
            spec = stft(wav.unsqueeze(0), self.hp.n_fft, self.hp.hop_size, self.hp.win_size, magnitude=True)

        if "mel" in self.keys:
            mel = spec_to_mel(
                spec, self.hp.n_fft, self.hp.n_mel, self.hp.sampling_rate, self.hp.mel_fmin, self.hp.mel_fmax, self.hp.clip_val
            ).squeeze(0)
            assert mel.size(1) * self.hp.hop_size == wav.size(0)

            if self.hp.mel_normalize:
                mel = (mel - self.hp.mel_mean) / self.hp.mel_std
            data["mel"] = mel       # shape: [num_mels, mel_len]
        if "mel_loss" in self.keys:
            mel = spec_to_mel(
                spec, self.hp.n_fft, self.hp.n_mel, self.hp.sampling_rate, self.hp.mel_fmin, self.hp.mel_fmax_loss, self.hp.clip_val
            ).squeeze(0)
            data["mel_loss"] = mel  # shape: [num_mels, mel_len]
        if "mel_len" in self.keys:
            data["mel_len"] = mel.size(-1)
        
        if "pitch" in self.keys:
            fmin, fmax = 75, 600
            padding = math.floor(self.hp.sampling_rate / fmin * 3 / 2 - self.hp.hop_size / 2) + 1
            _wav = np.pad(wav.numpy(), (padding, padding))
            snd = parselmouth.Sound(_wav, self.hp.sampling_rate)
            spec_len = wav.size(0) // self.hp.hop_size

            pitch = snd.to_pitch(
                time_step=self.hp.hop_size/self.hp.sampling_rate,
                pitch_floor=fmin,
                pitch_ceiling=fmax
            ).selected_array['frequency']
            
            voiced = np.sign(pitch, dtype=np.float32)

            start_f0 = pitch[pitch != 0][0]
            end_f0 = pitch[pitch != 0][-1]
            start_idx = np.where(pitch == start_f0)[0][0]
            end_idx = np.where(pitch == end_f0)[0][-1]
            pitch[:start_idx] = start_f0
            pitch[end_idx:] = end_f0

            # get non-zero frame index
            nonzero_idxs = np.where(pitch != 0.)[0]

            # perform linear interpolation
            interp_fn = interp1d(nonzero_idxs, pitch[nonzero_idxs])
            pitch = interp_fn(np.arange(0, pitch.shape[0]))
            
            if self.hp.log_pitch:
                pitch = np.log(pitch)
            if self.hp.pitch_normalize:
                pitch = (pitch - self.hp.pitch_mean) / self.hp.pitch_std

            pitch = pitch.astype(np.float32)    # pitch > 0
            pitch = torch.from_numpy(pitch).unsqueeze(0)
            voiced = torch.from_numpy(voiced).unsqueeze(0)
            assert pitch.size(1) == spec_len, f"filename: {self.wav_idx[idx]}, padding: {padding}, pitch: {pitch.shape}, mel: {spec_len}, wav: {wav.shape}"
            
            data["pitch"] = pitch       # shape: [1, mel_len]
            data["voiced"] = voiced     # shape: [1, mel_len]
        
        if "mask" in self.keys:
            mask = global_masking_threshold(spec, self.hp.n_fft, self.hp.sampling_rate).squeeze(0)
            data["mask"] = torch.from_numpy(mask)
        
        return data


class DatasetPreprocessed(_Dataset):
    def __init__(self, hp, keys, textprocessor=None, mode="train", batch_size=1, verbose=True):
        super().__init__(hp, keys, textprocessor, mode, batch_size, verbose)
        if "mel" in self.keys:
            self.tail = getattr(hp, "tail", None)
            if self.tail is None:
                self.tail = "none" if hp.mel_fmax is None else f"{int(hp.mel_fmax/1000)}k"
        if "opus" in self.keys:
            self.opus_tails = hp.opus_tails
    
    def get_wav_length(self, idx: int) -> float:
        wav_dir = os.path.join(self.data_dir, f"{self.wav_idx[idx]}_wav.npy")
        return os.path.getsize(wav_dir) / (4 * self.hp.sampling_rate)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = {}
        mel_start_idx = None
        if self.segment_size is not None:
            wav_ = np.load(os.path.join(self.data_dir, f"{self.wav_idx[idx]}_wav.npy"))
            wav = torch.from_numpy(wav_)
            if wav.size(0) >= self.segment_size:
                if "mel" in self.keys or "mel_loss" in self.keys or "pitch" in self.keys or "mask" in self.keys:
                    mel_size = wav.size(0) // self.hp.hop_size
                    mel_seg_size = self.segment_size // self.hp.hop_size
                    mel_start_idx = random.randint(0, mel_size - mel_seg_size)
                    wav_start_idx = mel_start_idx * self.hp.hop_size
                else:
                    wav_start_idx = random.randint(0, wav.size(0) - self.segment_size)
                wav = wav[wav_start_idx:wav_start_idx+self.segment_size]
            else:
                wav_pad_len = self.segment_size - wav.size(0)
                wav = F.pad(wav, (0, wav_pad_len), value=0)
                if "mel" in self.keys or "mel_loss" in self.keys or "pitch" in self.keys or "mask" in self.keys:
                    mel_pad_len = wav_pad_len // self.hp.hop_size
                    mel_pad_value = math.log(self.hp.clip_val)
        if "filename" in self.keys:
            data["filename"] = self.wav_idx[idx]
        if "text" in self.keys:
            text = self.get_text(idx)
            data["text"] = text     # shape: [num_mels, mel_len]
        if "text_len" in self.keys:
            data["text_len"] = text.size(0)
        if "wav" in self.keys:
            if self.segment_size is None:
                wav = np.load(os.path.join(self.data_dir, f"{self.wav_idx[idx]}_wav.npy"))
                wav = torch.from_numpy(wav)
            data["wav"] = wav       # shape: [wav_len]
        if "wav_len" in self.keys:
            data["wav_len"] = wav.size(0)
        if "mel" in self.keys:
            mel = np.load(os.path.join(self.data_dir, f"{self.wav_idx[idx]}_mel_{self.tail}.npy"))
            mel = torch.from_numpy(mel)
            if self.segment_size is not None:
                if mel_start_idx is None:
                    mel = F.pad(mel, (0, mel_pad_len), value=mel_pad_value)
                else:
                    mel = mel[..., mel_start_idx:mel_start_idx+mel_seg_size]
            if self.hp.mel_normalize:
                mel = (mel - self.hp.mel_mean) / self.hp.mel_std
            data["mel"] = mel       # shape: [num_mels, mel_len]
        if "mel_loss" in self.keys:
            tail = "none" if self.hp.mel_fmax_loss is None else f"{int(self.hp.mel_fmax_loss/1000)}k"
            mel = np.load(os.path.join(self.data_dir, f"{self.wav_idx[idx]}_mel_{tail}.npy"))
            mel = torch.from_numpy(mel)
            if self.segment_size is not None:
                if mel_start_idx is None:
                    mel = F.pad(mel, (0, mel_pad_len), value=mel_pad_value)
                else:
                    mel = mel[..., mel_start_idx:mel_start_idx+mel_seg_size]
            data["mel_loss"] = mel  # shape: [num_mels, mel_len]
        if "mel_len" in self.keys:
            data["mel_len"] = mel.size(-1)
        if "pitch" in self.keys:
            pitch = np.load(os.path.join(self.data_dir, f"{self.wav_idx[idx]}_pitch.npy"))
            voiced = np.load(os.path.join(self.data_dir, f"{self.wav_idx[idx]}_voiced.npy"))
            pitch, voiced = torch.from_numpy(pitch), torch.from_numpy(voiced)
            if self.segment_size is not None:
                if mel_start_idx is None:
                    pitch = F.pad(pitch, (0, mel_pad_len), value=pitch[-1])
                    voiced = F.pad(voiced, (0, mel_pad_len), value=0)
                else:
                    pitch = pitch[..., mel_start_idx:mel_start_idx+mel_seg_size]
                    voiced = voiced[..., mel_start_idx:mel_start_idx+mel_seg_size]
            if self.hp.log_pitch:
                pitch = torch.log(pitch)
            if self.hp.pitch_normalize:
                pitch = (pitch - self.hp.pitch_mean) / self.hp.pitch_std
            
            data["pitch"] = pitch       # shape: [1, mel_len]
            data["voiced"] = voiced     # shape: [1, mel_len]
        if "dur" in self.keys:
            dur = np.load(os.path.join(self.data_dir, f"{self.wav_idx[idx]}_{self.hp.duration_tail}.npy"))
            dur = torch.from_numpy(dur)
            data["dur"] = dur
        if "mask" in self.keys:
            mask = np.load(os.path.join(self.data_dir, f"{self.wav_idx[idx]}_mask_{self.hp.n_fft}.npy"))
            mask = torch.from_numpy(mask)
            if self.segment_size is not None:
                if mel_start_idx is None:
                    mask = F.pad(mask, (0, mel_pad_len), value=0)
                else:
                    mask = mask[..., mel_start_idx:mel_start_idx+mel_seg_size]
            data["mask"] = mask
        if "opus" in self.keys:
            tail = random.sample(self.opus_tails, k=1)[0]
            try:
                opus = np.load(os.path.join(self.data_dir, f"{self.wav_idx[idx]}_opus_{tail}.npy"))
            except Exception as e:
                print("error at file", self.wav_idx[idx])
                print(e)
                exit()
            opus = torch.from_numpy(opus)
            if self.segment_size is not None:
                if opus.size(0) >= self.segment_size:
                    opus = opus[wav_start_idx:wav_start_idx+self.segment_size]
                    opus = opus.to(torch.float32) / 2**15
                else:
                    opus = opus.to(torch.float32) / 2**15
                    opus = F.pad(opus, (0, wav_pad_len), value=0)
            else:
                opus = opus.to(torch.float32) / 2**15
            data["opus"] = opus
        
        return data


def collate(list_of_dicts: Dict[str, Any]) -> TensorDict:
    data = TensorDict()
    batch_size = len(list_of_dicts)
    keys = list_of_dicts[0].keys()

    for key in keys:
        if key == "filename":
            data["filename"] = [x["filename"] for x in list_of_dicts]
            continue
        elif key.endswith("_len"):
            data[key] = torch.LongTensor([x[key] for x in list_of_dicts])
            continue
        max_len = max([x[key].size(-1) for x in list_of_dicts])
        tensor = torch.zeros(batch_size, *[x for x in list_of_dicts[0][key].shape[:-1]], max_len, dtype=list_of_dicts[0][key].dtype)
        for i in range(batch_size):
            value = list_of_dicts[i][key]
            tensor[i, ..., :value.size(-1)] = value
        data[key] = tensor
    return data


class DNSDataset(torch.utils.data.Dataset):
    def __init__(self, hp, keys=None, textprocessor=None, mode="train", batch_size=1, verbose=False):
        super().__init__()
        self.hp = hp
        self.clean_dir = hp.clean_dir
        self.noisy_dir = hp.noisy_dir
        self.segment_size = getattr(hp, "segment_size", None)
        self.sampling_rate = hp.sampling_rate
        self.length_warned = False

        if mode == "train":
            self.files = list(range(hp.train_idx.start, hp.train_idx.end + 1))
        elif mode == "valid":
            self.files = list(range(hp.valid_idx.start, hp.valid_idx.end + 1))
        elif mode == "infer":
            #self.files = list(hp.infer_idx)
            self.files = list(range(hp.infer_idx.start, hp.infer_idx.end + 1))
            self.segment_size = 160000
            self.length_warned = True
        self.files: List[int]

    def shuffle(self, seed: int):
        random.seed(seed)
        random.shuffle(self.files)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        _id = self.files[idx]
        clean, sr = librosa.core.load(
            os.path.join(self.clean_dir, f"original_signal_{_id}.wav"),
            sr=None
        )
        assert sr == self.sampling_rate

        noisy, sr = librosa.core.load(
            os.path.join(self.noisy_dir, f"gain_controlled_{_id}.wav"),
            sr=None
        )
        assert sr == self.sampling_rate


        if self.segment_size is not None:
            if clean.size >= self.segment_size:
                # 0 <= start_idx <= wav.size(-1) - segment_size
                start_idx = random.randint(0, clean.size - self.segment_size)
                clean = clean[start_idx:start_idx+self.segment_size]
                noisy = noisy[start_idx:start_idx+self.segment_size]
            else:
                if not self.length_warned:
                    print(f"Warning - segment_size {self.segment_size} is longer than the"
                          f" data size {clean.size}")
                    self.length_warned = True
                padding = self.segment_size - clean.size
                clean = np.pad(clean, (padding // 2, padding - padding // 2))
                noisy = np.pad(noisy, (padding // 2, padding - padding // 2))
               
        res = {"clean": clean, "noisy": noisy}
        
        return res



class AECDataset(torch.utils.data.Dataset):
    def __init__(self, hp, keys=None, textprocessor=None, mode="train", batch_size=1, verbose=False):
        super().__init__()
        self.hp = hp
        self.near_dir = hp.near_dir
        self.far_dir = hp.far_dir
        self.mix_dir = hp.mix_dir
        if hp.two_way:
            self.echo_dir = hp.echo_dir
        self.segment_size = getattr(hp, "segment_size", None)
        self.sampling_rate = hp.sampling_rate
        
        self.length_warned = False

        if mode == "train":
            self.files = list(range(hp.train_idx.start, hp.train_idx.end + 1))
        elif mode == "valid":
            self.files = list(range(hp.valid_idx.start, hp.valid_idx.end + 1))
        elif mode == "infer":
            #self.files = list(hp.infer_idx)
            self.files = list(range(hp.infer_idx.start, hp.infer_idx.end + 1))
            self.segment_size = 160000
            self.length_warned = True
        self.files: List[int]

    def shuffle(self, seed: int):
        random.seed(seed)
        random.shuffle(self.files)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        _id = self.files[idx]
        near, sr = librosa.core.load(
            os.path.join(self.near_dir, f"nearend_speech_fileid_{_id}.wav"),
            sr=None
        )
        assert sr == self.sampling_rate

        far, sr = librosa.core.load(
            os.path.join(self.far_dir, f"farend_speech_fileid_{_id}.wav"),
            sr=None
        )
        assert sr == self.sampling_rate

        mix, sr = librosa.core.load(
            os.path.join(self.mix_dir, f"nearend_mic_fileid_{_id}.wav"),
            sr=None
        )
        assert sr == self.sampling_rate

        if self.hp.two_way:
            echo, sr = librosa.core.load(
                os.path.join(self.echo_dir, f"echo_fileid_{_id}.wav"),
                sr=None
            )
        assert sr == self.sampling_rate

        if self.segment_size is not None:
            if near.size >= self.segment_size:
                # 0 <= start_idx <= wav.size(-1) - segment_size
                start_idx = random.randint(0, near.size - self.segment_size)
                near = near[start_idx:start_idx+self.segment_size]
                mix = mix[start_idx:start_idx+self.segment_size]
                far = far[start_idx:start_idx+self.segment_size]
                if self.hp.two_way:
                    echo = echo[start_idx:start_idx+self.segment_size]
            else:
                if not self.length_warned:
                    print(f"Warning - segment_size {self.segment_size} is longer than the"
                          f" data size {near.size}")
                    self.length_warned = True
                padding = self.segment_size - near.size
                near = np.pad(near, (padding // 2, padding - padding // 2))
                far = np.pad(far, (padding // 2, padding - padding // 2))
                mix = np.pad(mix, (padding // 2, padding - padding // 2))
                if self.hp.two_way:
                    echo = np.pad(echo, (padding // 2, padding - padding // 2))

        res = {"near": near, "far": far, "mix": mix}
        if self.hp.two_way:
            res['echo'] = echo
        
        return res
