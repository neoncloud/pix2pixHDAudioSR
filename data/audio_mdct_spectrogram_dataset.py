import os
from numpy import ceil
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as aF
import random
from data.base_dataset import BaseDataset

class AudioMDCTSpectrogramDataset(BaseDataset):
    def __init__(self, opt) -> None:
        BaseDataset.__init__(self)
        self.lr_sampling_rate = opt.lr_sampling_rate
        self.hr_sampling_rate = opt.hr_sampling_rate
        self.segment_length = opt.segment_length
        self.n_fft = opt.n_fft
        self.hop_length = opt.hop_length
        self.win_length = opt.win_length
        self.audio_file = self.get_files(opt.dataroot)
        self.center = opt.center

        random.seed(1234)

    def __len__(self):
        return len(self.audio_file)

    def name(self):
        return 'AudioMDCTSpectrogramDataset'

    def __getitem__(self, idx):
        file_path = self.audio_file[idx]
        try:
            waveform, orig_sample_rate = torchaudio.load(file_path)
        except:
            waveform = []
            print("load audio failed")

        hr_waveform = aF.resample(waveform=waveform, orig_freq=orig_sample_rate, new_freq=self.hr_sampling_rate)
        lr_waveform = aF.resample(waveform=waveform, orig_freq=orig_sample_rate, new_freq=self.lr_sampling_rate)
        lr_waveform = aF.resample(waveform=lr_waveform, orig_freq=self.lr_sampling_rate, new_freq=self.hr_sampling_rate)
        #lr_waveform = aF.lowpass_biquad(waveform, sample_rate=self.hr_sampling_rate, cutoff_freq = self.lr_sampling_rate//2) #Meet the Nyquest sampling theorem
        hr, lr = self.seg_pad_audio(lr_waveform, hr_waveform)
        return {'image': hr.squeeze(0), 'label': lr.squeeze(0), 'inst':0, 'feat':0, 'path': file_path}


    def get_files(self, file_path):
        file_list = []
        for root, dirs, files in os.walk(file_path, topdown=False):
            for name in files:
                if os.path.splitext(name)[1] == ".wav" or ".mp3":
                    file_list.append(os.path.join(root, name))

        print(len(file_list))
        return file_list

    def seg_pad_audio(self, lr, hr):
        length = min(hr.size(1),lr.size(1))
        if length >= self.segment_length:
            max_audio_start = length - self.segment_length
            start = random.randint(0, max_audio_start)
            lr_waveform = lr[0][start : start + self.segment_length]
            hr_waveform = hr[0][start : start + self.segment_length]
        else:
            lr_waveform = F.pad(
                lr, (0, self.segment_length - lr.size(1)), 'constant'
            ).data
            hr_waveform = F.pad(
                hr, (0, self.segment_length - hr.size(1)), 'constant'
            ).data

        return hr_waveform, lr_waveform
class AudioMDCTSpectrogramTestDataset(BaseDataset):
    def __init__(self, opt) -> None:
        BaseDataset.__init__(self)
        self.lr_sampling_rate = opt.lr_sampling_rate
        self.hr_sampling_rate = opt.hr_sampling_rate
        self.segment_length = opt.segment_length
        self.n_fft = opt.n_fft
        self.hop_length = opt.hop_length
        self.win_length = opt.win_length
        self.center = opt.center
        self.dataroot = opt.dataroot
        try:
            self.raw_audio, self.in_sampling_rate = torchaudio.load(self.dataroot)
            self.audio_len = len(self.raw_audio)
        except:
            self.raw_audio = []
            print("load audio failed")
            exit(0)
        if opt.is_lr_input == False:
            self.raw_audio = aF.resample(waveform=self.raw_audio, orig_freq=self.in_sampling_rate, new_freq=self.lr_sampling_rate)
        self.raw_audio = aF.resample(waveform=self.raw_audio, orig_freq=self.lr_sampling_rate, new_freq=self.hr_sampling_rate)
        self.seg_audio = self.seg_pad_audio(self.raw_audio)

    def __len__(self):
        return self.seg_audio.size(0)

    def name(self):
        return 'AudioMDCTSpectrogramTestDataset'

    def __getitem__(self, idx):
        return {'image': torch.empty(1), 'label': self.seg_audio[idx,:].squeeze(0), 'inst':torch.empty(1), 'feat':torch.empty(1), 'path': self.dataroot}

    def seg_pad_audio(self, audio):
        audio = audio.squeeze(0)
        length = len(audio)
        if length >= self.segment_length:
            num_segments = int(ceil(length/self.segment_length))
            audio = F.pad(audio, (0, self.segment_length*num_segments - length), "constant").data
            audio = audio.unfold(dimension=0,size=self.segment_length,step=self.segment_length)
        else:
            audio = F.pad(audio, (0, self.segment_length - length), 'constant').data
            audio = audio.unsqueeze(0)

        return audio