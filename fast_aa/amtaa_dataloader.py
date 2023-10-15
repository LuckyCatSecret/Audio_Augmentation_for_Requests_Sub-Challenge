# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader.py.py

# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch
import csv
import json
import timm

from sklearn.metrics import rand_score
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import math
import random
import torch.nn as nn

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from spafe.fbanks import gammatone_fbanks
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torchvision.transforms
from torchvision.transforms import transforms
from .augmentations import apply_wav_augment,apply_spec_augment,apply_cv_augment

class AugmentGammatoneSTFT(nn.Module):
    def __init__(self, n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024,
                 htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=1, fmax_aug_range=1000):
        torch.nn.Module.__init__(self)
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e
        # Similar config to the spectrograms used in AST: https://github.com/YuanGongND/ast

        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.htk = htk
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
            print(f"Warning: FMAX is None setting to {fmax} ")
        self.fmax = fmax
        self.norm = norm
        self.hopsize = hopsize
        self.register_buffer('window',
                             torch.hann_window(win_length, periodic=False),
                             persistent=False)
        assert fmin_aug_range >= 1, f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation"
        assert fmin_aug_range >= 1, f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)

    def forward(self, x):

        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        x = torch.stft(x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                       center=True, normalized=False, window=self.window, return_complex=False)
        x = (x ** 2).sum(dim=-1)  # power mag
        # fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        # fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()
        # don't augment eval data
        # if not self.training:
        fmin = self.fmin
        fmax = self.fmax


        gamma_filbanks = gammatone_fbanks.gammatone_filter_banks(nfilts=self.n_mels,
                                                     nfft=self.n_fft,
                                                     fs=self.sr,
                                                     low_freq=fmin,
                                                     high_freq=fmax,
                                                     scale="constant")
        gamma_filbanks = torch.from_numpy(gamma_filbanks).float().to(x.device)
        with torch.cuda.amp.autocast(enabled=False):
            gammalspec = torch.matmul(gamma_filbanks, x)

        gammalspec = (gammalspec + 0.00001).log()

        gammalspec = (gammalspec + 4.5) / 5.  # fast normalization

        return gammalspec

class AudioMaskGenerator:
    def __init__(self, input_size=(128,1024), mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_H = input_size[0]
        self.input_W = input_size[1]
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_H % self.mask_patch_size == 0
        assert self.input_W % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size_H = self.input_H // self.mask_patch_size
        self.rand_size_W = self.input_W // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size_H*self.rand_size_W
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size_H, self.rand_size_W))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return torch.from_numpy(mask).long()

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])


class AudioDataset_pretrian(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.save_fig = True
        self.data = data_json['data']
        self.audio_conf = audio_conf
        self.mode = self.audio_conf.get('mode')
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))

        self.GammaBank = AugmentGammatoneSTFT(n_mels=self.melbins)
        self.mask_generator = AudioMaskGenerator()

        self.view = self.audio_conf.get('view')

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
            #这里添加噪音            
            # waveform = awgn(waveform.numpy(),20)
            # waveform = torch.from_numpy(waveform).type(torch.float32)
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()
            #这里添加噪音            
            # waveform1 = awgn(waveform1.numpy(),20)
            # waveform1 = torch.from_numpy(waveform1).type(torch.float32)
            # waveform2 = awgn(waveform2.numpy(),20)
            # waveform2 = torch.from_numpy(waveform2).type(torch.float32)

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            #mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def _wav2fbank_CQT(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        #     waveform = awgn(waveform.numpy(),0)
            waveform = torch.from_numpy(waveform).type(torch.float32)
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            # waveform1 = awgn(waveform1.numpy(),0)
            waveform1 = torch.from_numpy(waveform1).type(torch.float32)
            # waveform2 = awgn(waveform2.numpy(),0)
            waveform2 = torch.from_numpy(waveform2).type(torch.float32)

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            #mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        # fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        #                                           window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        fbank = librosa.cqt(np.array(waveform), sr=sr, hop_length=1280, fmin=None ,n_bins=128, bins_per_octave=24, tuning=0.0, filter_scale=1,
                                                    norm=1, sparsity=0.01, window='hann', scale=True, pad_mode='reflect', res_type=None, dtype=None)
        # print("fbank.shape",fbank.shape)
        fbank = librosa.amplitude_to_db(np.abs(fbank), ref=np.max)

        fbank = torch.from_numpy(fbank).squeeze(0).transpose(0,1)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        # if self.save_fig:
        #     self.save_fig = False
        #     plt.figure()
        #     img = librosa.display.specshow(np.array(fbank.transpose(0,1)), y_axis='cqt_hz', x_axis='time',
        #                                 sr=sr)
        #     plt.title("CQT spectrogram")
        #     plt.colorbar(img, format="%+2.f dB")
        #     plt.savefig("./voice_display.png")
        #     print("save voice_display.png success!!!")
        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda


    def _wav2fbank_gamma(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            #mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        gammafbank = self.GammaBank(waveform)
        gammafbank = gammafbank.squeeze(0).transpose(0,1)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]
        n_frames1 = gammafbank.shape[0]

        p = target_length - n_frames
        p1 = target_length - n_frames1

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        if p1 > 0:
            m1 = torch.nn.ZeroPad2d((0, 0, 0, p1))
            gammafbank = m1(gammafbank)
        if p < 0:
            fbank = fbank[0:target_length, :]
        if p1 < 0:
            gammafbank = gammafbank[0:target_length, :]


        if filename2 == None:
            # global save_fig
            # if save_fig:
            #     save_fig = False
            #     fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
            #     img = librosa.display.specshow(np.array(fbank.transpose(0,1)), y_axis='log', x_axis='time',
            #                                 sr=sr, ax=ax[0])
            #     ax[0].set(title='log mel spectrogram')
            #     ax[0].label_outer()

            #     librosa.display.specshow(np.array((gammafbank*5-4.5).transpose(0,1)), y_axis='log', x_axis='time',
            #                                 sr=sr, ax=ax[1])
            #     ax[1].set(title='gamma filter spectrogram')
            #     ax[1].label_outer()

            #     fig.colorbar(img, ax=ax, format="%+2.f dB")
            #     fig.savefig("./voice_display.png")
            #     print("save voice_display.png success!!!")
            return fbank, gammafbank, 0
        else:
            return fbank, gammafbank, mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup:
            datum = self.data[index]
            # find another sample to mix, also do balance sampling
            # sample the other sample from the multinomial distribution, will make the performance worse
            # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
            # sample the other sample from the uniform distribution
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]
            # get the mixed fbank
            if self.view=="CQT":
                fbank, mix_lambda = self._wav2fbank_CQT(datum['wav'])
            else:
                fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'])
            # initialize the label
            label_indices = np.zeros(self.label_num)
            # add sample 1 labels
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += (1.0-mix_lambda)
            label_indices = torch.FloatTensor(label_indices)
        # if not do mixup
        else:
            datum = self.data[index]
            label_indices = np.zeros(self.label_num)
            if self.view == "MEL":
                if self.mode=="SimMIM":
                    fbank, gammafbank, mix_lambda = self._wav2fbank_gamma(datum['wav'])
                else:
                    fbank, mix_lambda = self._wav2fbank(datum['wav'])
            elif self.view=="CQT":
                fbank, mix_lambda = self._wav2fbank_CQT(datum['wav'])

            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0

            label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)        
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        rand_seed = random.uniform(0, 1)

        mask = self.mask_generator()

        if self.mode=="SimMIM" and self.view == "MEL":
            return fbank,gammafbank,mask,label_indices
        else:
            return fbank,mask,label_indices

    def __len__(self):
        return len(self.data)

class AudioDataset_finetue(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None,aug_wav=None,aug_spec=None,aug_cv=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        ###augmentations for finetune
        self.aug_wav=aug_wav
        self.aug_spec=aug_spec
        self.aug_cv=aug_cv
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.save_fig = True
        self.data = data_json['data']
        self.audio_conf = audio_conf
        self.mode = self.audio_conf.get('mode')
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')
        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        ### add label for search
        # self.labels = [int(self.index_dict[datam['labels']]) for datam in self.data]
        ###
        self.labels = []
        for datum in self.data:
            label_indices = np.zeros(self.label_num)
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0
            # label_indices = torch.FloatTensor(label_indices)
            self.labels.append(label_indices)
        # self.labels=torch.tensor(np.array(self.labels))
        self.labels = torch.FloatTensor(np.array(self.labels))
        print('number of classes is {:d}'.format(self.label_num))
        self.mask_generator = AudioMaskGenerator()
        self.view = self.audio_conf.get('view')

    def _wavef2fbank(self,waveform,sr):
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0,
                                                  frame_shift=10)
        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]
        return fbank

    def __getitem__(self,index):
        datum = self.data[index]
        label_indices = np.zeros(self.label_num)

        torchaudio.set_audio_backend("sox_io")
        waveform, sr = torchaudio.load(datum['wav'])
        waveform = waveform - waveform.mean()

        ###### stage1 augment for waveform 面向waveform的数据增强
        if self.aug_wav is not None:
            waveform = waveform.unsqueeze(0)
            waveform=self.aug_wav(waveform)
            waveform = waveform.squeeze(0)
        fbank=self._wavef2fbank(waveform,sr)

        # deal labels
        for label_str in datum['labels'].split(','):
            label_indices[int(self.index_dict[label_str])] = 1.0
        label_indices = torch.FloatTensor(label_indices)

        ###### stage2 augment for spectrogram 面向频谱图的数据增强
        # fbank = fbank.to(torch.device('cuda'))
        if self.aug_spec is not None:
            fbank=self.aug_spec(fbank)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
            ### [-1,1]->[0,1]
            fbank=(fbank+1.0)/2.0
        else:
            pass

        ###### stage3 augment for cv torchvision.transform (original fast-autoaug policy)
        if self.aug_cv is not None:
            #########
            # fbank = fbank.to(torch.device('cpu'))
            fbank=self.aug_cv(fbank)
        
        fbank = fbank * 2.0 - 1.0

        return fbank, label_indices

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


### pretrain loader
def build_loader_ast(dataset_json_file, audio_conf, label_csv=None):
    dataset = AudioDataset_pretrian(dataset_json_file, audio_conf, label_csv)
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    dataloader = DataLoader(dataset,audio_conf.get("batch_size") , sampler=sampler, num_workers=16, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    return dataloader

### finetune loader
"""
                                finetune dataloader
for finetuen_train_dataloader:
    dataset_train, data_loader_train = build_loader_ast_finetune(train_json_file, audio_conf, True,True,label_csv)
for finetune_val_dataloader:
    dataset_val, data_loader_val = build_loader_ast_finetune(eval_json_file, eval_audio_conf, False,False,label_csv)
"""
def build_loader_ast_finetune(dataset_json_file, audio_conf,drop_last = True,shuffle = True, label_csv=None):

    dataset = AudioDataset_finetue(dataset_json_file, audio_conf, label_csv)
    
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=shuffle)
    dataloader = DataLoader(dataset,audio_conf.get("batch_size") , sampler=sampler, num_workers=16, pin_memory=True, drop_last=drop_last, collate_fn=collate_fn)
    
    return dataset,dataloader

### my finetuen dataloader with aug
def build_loader_finetune(dataset_json_file,audio_conf,drop_last=True,shuffle=True,label_csv=None,aug_wav=None,aug_spec=None,aug_cv=None):
    dataset=AudioDataset_finetue(dataset_json_file,audio_conf,label_csv,aug_wav,aug_spec,aug_cv)
    sampler=DistributedSampler(dataset,num_replicas=dist.get_world_size(),rank=dist.get_rank(),shuffle=shuffle)
    dataloader=DataLoader(dataset,audio_conf.get("batch_size"),sampler=sampler,num_workers=16,pin_memory=True,drop_last=drop_last,collate_fn=collate_fn)
    return dataset,dataloader


def make_wav_aug(final_policy_set_wav,sr):
    aug_wav=transforms.Compose([WavAugmentation(final_policy_set_wav,sr)])
    # return transforms.Compose([WavAugmentation(final_policy_set_wav,sr)])
    return aug_wav

def make_spec_aug(final_policy_set_spec):
    aug_spec=transforms.Compose([SpecAugmentation(final_policy_set_spec)])
    return aug_spec

def make_cv_aug(final_policy_set_cv):
    aug_cv=transforms.Compose([CVAugmentation(final_policy_set_cv)])
    return aug_cv

class WavAugmentation(object):
    def __init__(self,policies,sr):
        self.policies=policies
        self.sr=sr
    def __call__(self,waveform):
        for _ in range(1):
            policy=random.choice(self.policies)
            for name,pr in policy:
                waveform=apply_wav_augment(waveform,name,pr,self.sr)
        return waveform
    # def __len__(self):
    #     return len(self.policies)

class SpecAugmentation(object):
    def __init__(self,policies):
        self.policies=policies
    def __call__(self,fbank):
        for _ in range(1):
            policy=random.choice(self.policies)
            for name,pr,level in policy:
                if random.random()>pr:
                    continue
                fbank=apply_spec_augment(fbank,name,level)
        return fbank

class CVAugmentation(object):
    def __init__(self,policies):
        self.policies=policies
    def __call__(self,img):
        for _ in range(1):
            policy=random.choice(self.policies)
            for name,pr,level in policy:
                if random.random()>pr:
                    continue
                img=apply_cv_augment(img,name,level)
        return img
