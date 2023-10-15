# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random
import numpy as np
import torch
# from torchvision.transforms.transforms import Compose
import torch_audiomentations
import torchvision.transforms.functional as TVFun
import torchvision.transforms.functional_tensor as FunTen
import torchaudio

random_mirror = False

#0 wav
def BandPassFilter(wave,p,sr=16000):
    transform=torch_audiomentations.BandPassFilter(p=p)
    return transform(wave,sample_rate=sr)
#1 wav
def BandStopFilter(wave,p,sr=16000):
    transform=torch_audiomentations.BandStopFilter(p=p)
    return transform(wave,sample_rate=sr)
#2 wav
def AddColoredNoise(wave,p,sr=16000):
    transform=torch_audiomentations.AddColoredNoise(p=p)
    return transform(wave, sample_rate=sr)
#3 wav
def Gain(wave,p,sr=16000):
    transform=torch_audiomentations.Gain(p=p)
    return transform(wave, sample_rate=sr)
#4 wav
def HighPassFilter(wave,p,sr=16000):
    transform=torch_audiomentations.HighPassFilter(p=p)
    return transform(wave, sample_rate=sr)
#5 wav
def Identity(wave,p,sr=16000):
    transform=torch_audiomentations.Identity(p=p)
    return transform(wave, sample_rate=sr)
#6 wav
def LowPassFilter(wave,p,sr=16000):
    transform=torch_audiomentations.LowPassFilter(p=p)
    return transform(wave, sample_rate=sr)
#7 wav
def PeakNormalization(wave,p,sr=16000):
    transform=torch_audiomentations.PeakNormalization(p=p)
    return transform(wave, sample_rate=sr)
#8 wav
def PolarityInversion(wave,p,sr=16000):
    transform=torch_audiomentations.PolarityInversion(p=p)
    return transform(wave, sample_rate=sr)
#9 wav
def Shift(wave,p,sr=16000):
    transform=torch_audiomentations.Shift(p=p)
    return transform(wave, sample_rate=sr)
#10 wav
def TimeInversion(wave,p,sr=16000):
    transform=torch_audiomentations.TimeInversion(p=p)
    return transform(wave, sample_rate=sr)

#0 spec
def FrequencyMasking(img,v): #mask ratio
    assert 0.1<=v<=0.7
    masksize=img.shape[1]*v

    img = torch.transpose(img, 0, 1)
    img = img.unsqueeze(0)

    # ### test for frequency need to device cuda
    # img = img.to(torch.device('cuda'))

    freqm=torchaudio.transforms.FrequencyMasking(int(masksize))
    img=freqm(img)

    img = img.squeeze(0)
    img = torch.transpose(img, 0, 1)
    return img
#1 spec
def TimeMasking(img,v): #mask ratio
    assert 0.1<=v<=0.7
    masksize=img.shape[0]*v

    img=torch.transpose(img,0,1)
    img=img.unsqueeze(0)

    timem=torchaudio.transforms.TimeMasking(int(masksize))
    img=timem(img)

    img=img.squeeze(0)
    img=torch.transpose(img,0,1)
    return img
#2 spec
def TimeFrequencyMasking(img,v): #total mask ratio
    assert 0.1<=v<=0.8
    freq_ms=img.shape[1]*v/2
    time_ms=img.shape[0]*v/2
    img = torch.transpose(img, 0, 1)
    img = img.unsqueeze(0)

    timem=torchaudio.transforms.TimeMasking(int(time_ms))
    img=timem(img)

    # ### test for frequency need to device cuda
    # img=img.to(torch.device('cuda'))

    freqm=torchaudio.transforms.FrequencyMasking(int(freq_ms))
    img=freqm(img)

    img = img.squeeze(0)
    img = torch.transpose(img, 0, 1)
    return img
#3 spec
def TimeStretch(img,v):
    assert 0.5<=v<=1.5
    n_freq=img.shape[1]

    img = torch.transpose(img, 0, 1)
    img = img.unsqueeze(0)

    strech=torchaudio.transforms.TimeStretch(n_freq=n_freq)

    ### convert to cpu
    # img = img.to(torch.device('cpu'))

    img=strech(img, v)

    ### convert return cuda
    # img = img.to(torch.device('cuda'))

    img = img.squeeze(0)
    img = torch.transpose(img, 0, 1)

    return img

#0 cv
def ShearX(img, v):
    assert -0.3 <= v <= 0.3
    img = img.unsqueeze(0)
    if random_mirror and random.random() > 0.5:
        v = -v
    return FunTen.affine(img, [1, v, 0, 0, 1, 0]).squeeze(0)

#1 cv
def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    img = img.unsqueeze(0)
    if random_mirror and random.random() > 0.5:
        v = -v
    return FunTen.affine(img, [1, 0, 0, v, 1, 0]).squeeze(0)

#2 cv
def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    img = img.unsqueeze(0)
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.shape[2]
    return FunTen.affine(img, [1, 0, v, 0, 1, 0]).squeeze(0)

#3 cv
def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    img = img.unsqueeze(0)
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.shape[1]
    return FunTen.affine(img, [1, 0, 0, 0, 1, v]).squeeze(0)

#4 cv
def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    img = img.unsqueeze(0)
    if random_mirror and random.random() > 0.5:
        v = -v
    return TVFun.rotate(img, v).squeeze(0)

#5 cv
def AutoContrast(img, _):
    img = img.unsqueeze(0)
    return TVFun.autocontrast(img).squeeze(0)

#6 cv
def Invert(img, _):
    img = img.unsqueeze(0)
    return TVFun.invert(img).squeeze(0)

#7 cv
def Solarize(img, v):
    assert 0 <= v <= 1
    # assert 0 <= v <= 256
    img = img.unsqueeze(0)
    return TVFun.solarize(img, v).squeeze(0)

#8 cv
def Contrast(img, v):
    assert 0.1 <= v <= 1.9
    img = img.unsqueeze(0)
    return TVFun.adjust_contrast(img, v).squeeze(0)

#9 cv
def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    img = img.unsqueeze(0)
    return TVFun.adjust_saturation(img, v).squeeze(0)

#10 cv
def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    img = img.unsqueeze(0)
    return TVFun.adjust_brightness(img, v).squeeze(0)

#11 cv
def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    img = img.unsqueeze(0)
    return TVFun.adjust_sharpness(img, v).squeeze(0)

#12 cv
def Add_noise(img,v):
    assert 0.0<=v<=0.9
    ###### to img cuda
    # img=img+torch.rand(img.shape[0],img.shape[1], device=torch.device('cuda'))*v/10
    img = img + torch.rand(img.shape[0], img.shape[1]) * v / 10
    return img


def augment_wav_list():  # 11 oeprations and their ranges
    l = [
        (BandPassFilter),  # 0
        (BandStopFilter),  # 1
        (AddColoredNoise),  # 2
        (Gain),  # 3
        (HighPassFilter),  # 4
        (Identity),  # 5
        (LowPassFilter),  # 6
        (PeakNormalization),  # 7
        (PolarityInversion),  # 8
        (Shift),  # 9
        (TimeInversion),  # 10
    ]
    return l

def augment_spec_list():
    l=[
        (FrequencyMasking, 0.1, 0.7),  # 0
        (TimeMasking, 0.1, 0.7),  # 1
        (TimeFrequencyMasking, 0.1, 0.8),  # 2
        # (TimeStretch, 0.5, 1.5),  # 3
    ]
    return l

def augment_cv_list():
    l=[
        (ShearX, -0.3, 0.3),  # 0
        (ShearY, -0.3, 0.3),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Solarize, 0, 1),  # 7
        (Contrast, 0.1, 1.9),  # 8
        # (Color, 0.1, 1.9),  # 9
        (Brightness, 0.1, 1.9),  # 10
        (Sharpness, 0.1, 1.9),  # 11
        (Add_noise,0.1,0.9), # 12
    ]
    return l

augment_wav_dict = {fn.__name__: fn for fn in augment_wav_list()}
augment_spec_dict={fn.__name__:(fn,v1,v2) for fn,v1,v2 in augment_spec_list()}
augment_cv_dict={fn.__name__:(fn,v1,v2) for fn,v1,v2 in augment_cv_list()}

def get_wav_augment(name):
    return augment_wav_dict[name]
def get_spec_augment(name):
    return augment_spec_dict[name]
def get_cv_augment(name):
    return augment_cv_dict[name]


def apply_wav_augment(wave,name,pr,sr):
    # augment_fn=augment_wav_dict[name]
    augment_fn=get_wav_augment(name)
    return augment_fn(wave=wave.clone(),p=pr,sr=sr)

def apply_spec_augment(fbank, name, level):
    # augment_fn, low, high = augment_spec_dict[name]
    augment_fn, low, high = get_spec_augment(name)
    return augment_fn(fbank.clone(), level * (high - low) + low)

def apply_cv_augment(img, name, level):
    # augment_fn, low, high = augment_cv_dict[name]
    augment_fn, low, high = get_cv_augment(name)
    return augment_fn(img.clone(), level * (high - low) + low)
