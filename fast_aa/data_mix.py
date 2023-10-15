import logging

import numpy as np
import os

import math
import random
import torch
import torchvision
from PIL import Image

from torch.utils.data import SubsetRandomSampler, Sampler, Subset, ConcatDataset
import torch.distributed as dist
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit,ShuffleSplit
from theconf import Config as C

# from archive import arsaug_policy, autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10, fa_reduced_svhn, fa_resnet50_rimagenet
from .archive import arsaug_policy, autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10, fa_reduced_svhn, fa_resnet50_rimagenet
# from augmentations import *
# from FastAutoAugment.augmentations import *
# from common import get_logger
from .common import get_logger
# from imagenet import ImageNet
# from .imagenet import ImageNet
# from networks.efficientnet_pytorch.model import EfficientNet
# from .networks.efficientnet_pytorch.model import EfficientNet

# from FastAutoAugment.dataloader import AudioDataset
# from .amtaa_dataloader import AudioDataset_finetue,make_wav_aug,make_spec_aug,make_cv_aug,WavAugmentation,SpecAugmentation,CVAugmentation
from .amtaa_dataloader_mix import AudioDataset_finetue,make_wav_aug,make_spec_aug,make_cv_aug,make_mix_aug,WavAugmentation,SpecAugmentation,CVAugmentation
# from FastAutoAugment.amtaa_dataloader import AudioDataset_finetue,make_wav_aug,make_spec_aug,make_cv_aug,WavAugmentation,SpecAugmentation,CVAugmentation
from dataset import AudiosetDataset

logger = get_logger('Fast AutoAugment')
logger.setLevel(logging.INFO)


def get_dataloaders(conf,dataset, batch, dataroot, split=0.15, split_idx=0, multinode=False, target_lb=-1):
    ## 首先分别对不同类型的数据集，进行图像大小、归一化上的预处理/transform；生成transform_train, transform_test
    if 'esc50' in dataset:
        audio_conf = {'num_mel_bins': conf['num_mel_bins'], 'target_length': conf['target_length'], 'freqm': conf['freqm'],
                      'timem': conf['timem'], 'mixup': conf['mixup'], 'dataset': conf['dataset'],
                      'mode': 'train', 'mean': conf['dataset_mean'], 'std': conf['dataset_std'], 'noise': conf['noise']}
        val_audio_conf = {'num_mel_bins': conf['num_mel_bins'], 'target_length': conf['target_length'], 'freqm': 0,
                          'timem': 0, 'mixup': 0, 'dataset': conf['dataset'],
                          'mode': 'evaluation', 'mean': conf['dataset_mean'], 'std': conf['dataset_std'], 'noise': False}
        transform_train=transforms.Compose([
            transforms.ToTensor(),
 #           transforms.Normalize(conf['dataset_mean'], conf['dataset_std'])
        ])
        transform_test=transforms.Compose([
            transforms.ToTensor(),
#            transforms.Normalize(conf['dataset_mean'], conf['dataset_std'])
        ])
    elif 'BalancedAudioSet' in dataset:
        audio_conf = {'num_mel_bins': conf['num_mel_bins'], 'target_length': conf['target_length'],
                      'freqm': conf['freqm'],
                      'timem': conf['timem'], 'mixup': conf['mixup'], 'dataset': conf['dataset'],
                      'mode': 'train', 'mean': conf['dataset_mean'], 'std': conf['dataset_std'], 'noise': conf['noise']}
        val_audio_conf = {'num_mel_bins': conf['num_mel_bins'], 'target_length': conf['target_length'], 'freqm': 0,
                          'timem': 0, 'mixup': 0, 'dataset': conf['dataset'],
                          'mode': 'evaluation', 'mean': conf['dataset_mean'], 'std': conf['dataset_std'],
                          'noise': False}
        wav_aug = C.get()['aug'][0]
        spec_aug = C.get()['aug'][1]
        cv_aug = C.get()['aug'][2]
        mix_aug=C.get()['aug'][3]

        aug_wav=make_wav_aug(wav_aug,conf['sample_rate'])
        aug_spec=make_spec_aug(spec_aug)
        aug_cv=make_cv_aug(cv_aug)
        aug_mix=make_mix_aug(mix_aug)
    elif 'audioset' in dataset:
        norm_stats = {'audioset':[-4.2677393, 4.5689974], 'k400':[-4.2677393, 4.5689974],
                  'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
        target_length = {'audioset':1024, 'k400':1024, 'esc50':512, 'speechcommands':128}
        multilabel_dataset = {'audioset': True, 'esc50': False, 'k400': False, 'speechcommands': True}
        audio_conf_train = {'num_mel_bins': 128,
                    'target_length': target_length[conf.dataset],
                    'freqm': 48,
                    'timem': 192,
                    'mixup': conf.mixup,
                    'dataset': conf.dataset,
                    'mode':'train',
                    'mean':norm_stats[conf.dataset][0],
                    'std':norm_stats[conf.dataset][1],
                    'noise':False,
                    'multilabel':multilabel_dataset[conf.dataset],
                    'sample_rate':32000,
                    'batch_size':conf.batch_size,
                    'view':'mel',
                    }
        audio_conf_val = {'num_mel_bins': 128,
                    'target_length': target_length[conf.dataset],
                    'freqm': 0,
                    'timem': 0,
                    'mixup': 0,
                    'dataset': conf.dataset,
                    'mode':'val',
                    'mean':norm_stats[conf.dataset][0],
                    'std':norm_stats[conf.dataset][1],
                    'noise':False,
                    'multilabel':multilabel_dataset[conf.dataset],
                    'sample_rate': 32000,
                    'batch_size': conf.batch_size,
                    'view': 'mel',
                    }
        wav_aug = C.get()['aug'][0]
        spec_aug = C.get()['aug'][1]
        cv_aug = C.get()['aug'][2]
        mix_aug=C.get()['aug'][3]

        aug_wav=make_wav_aug(wav_aug,audio_conf_train.get('sample_rate'))
        aug_spec=make_spec_aug(spec_aug)
        aug_cv=make_cv_aug(cv_aug)
        aug_mix=make_mix_aug(mix_aug)
    elif 'voc-c' in dataset:
        norm_stats = {
            'audioset':[-4.2677393, 4.5689974], 
            'k400':[-4.2677393, 4.5689974], 
            'esc50':[-6.6268077, 5.358466], 
            'speechcommands':[-6.845978, 5.5654526],
            'deepship':[-6.6268077,5.358466],
            'ksf-c':[-4.2677393,4.5689974],
            'voc-c':[-4.2677393,4.5689974]}
        target_length = {
            'audioset':1024,
            'k400':1024,
            'esc50':1024,#512
            'speechcommands':128,
            'deepship':1024, #512
            'ksf-c':1024,    #384
            'voc-c':1024}    #128
        multilabel_dataset = {
            'audioset': True, 
            'esc50': False, 
            'k400': False, 
            'speechcommands': True,
            'deepship':False,
            'ksf-c':False,
            'voc-c':False}
        aug_param={
            'esc50':[48,192,0.],
            'deepship':[48,192,0.],
            'ksf-c':[48,192,0.],#96
            'voc-c':[48,192,0.]#48
        }
        audio_conf_train = {'num_mel_bins': 128, 
                        'target_length': target_length[conf.dataset], 
                        'freqm': aug_param[conf.dataset][0],
                        'timem': aug_param[conf.dataset][1],
                        'mixup': aug_param[conf.dataset][2],
                        'dataset': conf.dataset,
                        'mode':'train',
                        'mean':norm_stats[conf.dataset][0],
                        'std':norm_stats[conf.dataset][1],
                        'noise':False,
                        'multilabel':multilabel_dataset[conf.dataset],
                        }
        audio_conf_val = {'num_mel_bins': 128, 
                        'target_length': target_length[conf.dataset], 
                        'freqm': 0,
                        'timem': 0,
                        'mixup': 0,
                        'dataset': conf.dataset,
                        'mode':'val',
                        'mean':norm_stats[conf.dataset][0],
                        'std':norm_stats[conf.dataset][1],
                        'noise':False,
                        'multilabel':multilabel_dataset[conf.dataset],
                        }  
        wav_aug = C.get()['aug'][0]
        spec_aug = C.get()['aug'][1]
        cv_aug = C.get()['aug'][2]
        mix_aug=C.get()['aug'][3]

        aug_wav=make_wav_aug(wav_aug,16000)
        aug_spec=make_spec_aug(spec_aug)
        aug_cv=make_cv_aug(cv_aug)
        aug_mix=make_mix_aug(mix_aug)
    elif 'ksf-c' in dataset:
        norm_stats = {
            'audioset':[-4.2677393, 4.5689974], 
            'k400':[-4.2677393, 4.5689974], 
            'esc50':[-6.6268077, 5.358466], 
            'speechcommands':[-6.845978, 5.5654526],
            'deepship':[-6.6268077,5.358466],
            'ksf-c':[-4.2677393,4.5689974],
            'voc-c':[-4.2677393,4.5689974]}
        target_length = {
            'audioset':1024,
            'k400':1024,
            'esc50':1024,#512
            'speechcommands':128,
            'deepship':1024, #512
            'ksf-c':1024,    #384
            'voc-c':1024}    #128
        multilabel_dataset = {
            'audioset': True, 
            'esc50': False, 
            'k400': False, 
            'speechcommands': True,
            'deepship':False,
            'ksf-c':False,
            'voc-c':False}
        aug_param={
            'esc50':[48,192,0.],
            'deepship':[48,192,0.],
            'ksf-c':[48,192,0.],#96
            'voc-c':[48,192,0.]#48
        }
        audio_conf_train = {'num_mel_bins': 128, 
                        'target_length': target_length[conf.dataset], 
                        'freqm': aug_param[conf.dataset][0],
                        'timem': aug_param[conf.dataset][1],
                        'mixup': aug_param[conf.dataset][2],
                        'dataset': conf.dataset,
                        'mode':'train',
                        'mean':norm_stats[conf.dataset][0],
                        'std':norm_stats[conf.dataset][1],
                        'noise':False,
                        'multilabel':multilabel_dataset[conf.dataset],
                        }
        audio_conf_val = {'num_mel_bins': 128, 
                        'target_length': target_length[conf.dataset], 
                        'freqm': 0,
                        'timem': 0,
                        'mixup': 0,
                        'dataset': conf.dataset,
                        'mode':'val',
                        'mean':norm_stats[conf.dataset][0],
                        'std':norm_stats[conf.dataset][1],
                        'noise':False,
                        'multilabel':multilabel_dataset[conf.dataset],
                        }  
        wav_aug = C.get()['aug'][0]
        spec_aug = C.get()['aug'][1]
        cv_aug = C.get()['aug'][2]
        mix_aug=C.get()['aug'][3]

        aug_wav=make_wav_aug(wav_aug,audio_conf_train.get('sample_rate'))
        aug_spec=make_spec_aug(spec_aug)
        aug_cv=make_cv_aug(cv_aug)
        aug_mix=make_mix_aug(mix_aug)
    elif 'esc50' in dataset:
        norm_stats = {
            'audioset':[-4.2677393, 4.5689974], 
            'k400':[-4.2677393, 4.5689974], 
            'esc50':[-6.6268077, 5.358466], 
            'speechcommands':[-6.845978, 5.5654526],
            'deepship':[-6.6268077,5.358466],
            'ksf-c':[-4.2677393,4.5689974],
            'voc-c':[-4.2677393,4.5689974]}
        target_length = {
            'audioset':1024,
            'k400':1024,
            'esc50':1024,#512
            'speechcommands':128,
            'deepship':1024, #512
            'ksf-c':1024,    #384
            'voc-c':1024}    #128
        multilabel_dataset = {
            'audioset': True, 
            'esc50': False, 
            'k400': False, 
            'speechcommands': True,
            'deepship':False,
            'ksf-c':False,
            'voc-c':False}
        aug_param={
            'esc50':[48,192,0.],
            'deepship':[48,192,0.],
            'ksf-c':[48,192,0.],#96
            'voc-c':[48,192,0.]#48
        }
        audio_conf_train = {'num_mel_bins': 128, 
                        'target_length': target_length[conf.dataset], 
                        'freqm': aug_param[conf.dataset][0],
                        'timem': aug_param[conf.dataset][1],
                        'mixup': aug_param[conf.dataset][2],
                        'dataset': conf.dataset,
                        'mode':'train',
                        'mean':norm_stats[conf.dataset][0],
                        'std':norm_stats[conf.dataset][1],
                        'noise':False,
                        'multilabel':multilabel_dataset[conf.dataset],
                        }
        audio_conf_val = {'num_mel_bins': 128, 
                        'target_length': target_length[conf.dataset], 
                        'freqm': 0,
                        'timem': 0,
                        'mixup': 0,
                        'dataset': conf.dataset,
                        'mode':'val',
                        'mean':norm_stats[conf.dataset][0],
                        'std':norm_stats[conf.dataset][1],
                        'noise':False,
                        'multilabel':multilabel_dataset[conf.dataset],
                        }  
        wav_aug = C.get()['aug'][0]
        spec_aug = C.get()['aug'][1]
        cv_aug = C.get()['aug'][2]
        mix_aug=C.get()['aug'][3]

        aug_wav=make_wav_aug(wav_aug,audio_conf_train.get('sample_rate'))
        aug_spec=make_spec_aug(spec_aug)
        aug_cv=make_cv_aug(cv_aug)
        aug_mix=make_mix_aug(mix_aug)
    elif 'deepship' in dataset:
        norm_stats = {
            'audioset':[-4.2677393, 4.5689974], 
            'k400':[-4.2677393, 4.5689974], 
            'esc50':[-6.6268077, 5.358466], 
            'speechcommands':[-6.845978, 5.5654526],
            'deepship':[-6.6268077,5.358466],
            'ksf-c':[-4.2677393,4.5689974],
            'voc-c':[-4.2677393,4.5689974]}
        target_length = {
            'audioset':1024,
            'k400':1024,
            'esc50':1024,#512
            'speechcommands':128,
            'deepship':1024, #512
            'ksf-c':1024,    #384
            'voc-c':1024}    #128
        multilabel_dataset = {
            'audioset': True, 
            'esc50': False, 
            'k400': False, 
            'speechcommands': True,
            'deepship':False,
            'ksf-c':False,
            'voc-c':False}
        aug_param={
            'esc50':[48,192,0.],
            'deepship':[48,192,0.],
            'ksf-c':[48,192,0.],#96
            'voc-c':[48,192,0.]#48
        }
        audio_conf_train = {'num_mel_bins': 128, 
                        'target_length': target_length[conf.dataset], 
                        'freqm': aug_param[conf.dataset][0],
                        'timem': aug_param[conf.dataset][1],
                        'mixup': aug_param[conf.dataset][2],
                        'dataset': conf.dataset,
                        'mode':'train',
                        'mean':norm_stats[conf.dataset][0],
                        'std':norm_stats[conf.dataset][1],
                        'noise':False,
                        'multilabel':multilabel_dataset[conf.dataset],
                        }
        audio_conf_val = {'num_mel_bins': 128, 
                        'target_length': target_length[conf.dataset], 
                        'freqm': 0,
                        'timem': 0,
                        'mixup': 0,
                        'dataset': conf.dataset,
                        'mode':'val',
                        'mean':norm_stats[conf.dataset][0],
                        'std':norm_stats[conf.dataset][1],
                        'noise':False,
                        'multilabel':multilabel_dataset[conf.dataset],
                        }  
        wav_aug = C.get()['aug'][0]
        spec_aug = C.get()['aug'][1]
        cv_aug = C.get()['aug'][2]
        mix_aug=C.get()['aug'][3]

        aug_wav=make_wav_aug(wav_aug,audio_conf_train.get('sample_rate'))
        aug_spec=make_spec_aug(spec_aug)
        aug_cv=make_cv_aug(cv_aug)
        aug_mix=make_mix_aug(mix_aug)
    elif 'hcc' in dataset:
        norm_stats = {
        'audioset':[-4.2677393, 4.5689974], 
        'k400':[-4.2677393, 4.5689974], 
        'esc50':[-6.6268077, 5.358466], 
        'speechcommands':[-6.845978, 5.5654526],
        'deepship':[-6.6268077,5.358466],
        'ksf-c':[-4.2677393,4.5689974],
        'voc-c':[-4.2677393,4.5689974],
        'hcc': [-7.5503645, 5.946564]
        }
        target_length = {
            'audioset':1024,
            'k400':1024,
            'esc50':1024,#512
            'speechcommands':128,
            'deepship':1024, #512
            'ksf-c':1024,    #384
            'voc-c':1024,    #128
            'hcc':3096
        }
        multilabel_dataset = {
            'audioset': True, 
            'esc50': False, 
            'k400': False, 
            'speechcommands': True,
            'deepship':False,
            'ksf-c':False,
            'voc-c':False,
            'hcc':True
            }
        aug_param={
            'esc50':[48,192,0.],
            'deepship':[48,192,0.],
            'ksf-c':[48,192,0.],#96
            'voc-c':[48,192,0.],#48
            'hcc':[48,192,0.]
        }
        audio_conf_train = {'num_mel_bins': 128, 
                        'target_length': target_length[conf.dataset], 
                        'freqm': aug_param[conf.dataset][0],
                        'timem': aug_param[conf.dataset][1],
                        'mixup': aug_param[conf.dataset][2],
                        'dataset': conf.dataset,
                        'mode':'train',
                        'mean':norm_stats[conf.dataset][0],
                        'std':norm_stats[conf.dataset][1],
                        'noise':False,
                        'multilabel':multilabel_dataset[conf.dataset],
                        'sample_rate':16000,
                        }
        audio_conf_val = {'num_mel_bins': 128, 
                        'target_length': target_length[conf.dataset], 
                        'freqm': 0,
                        'timem': 0,
                        'mixup': 0,
                        'dataset': conf.dataset,
                        'mode':'val',
                        'mean':norm_stats[conf.dataset][0],
                        'std':norm_stats[conf.dataset][1],
                        'noise':False,
                        'multilabel':multilabel_dataset[conf.dataset],
                        'sample_rate':16000,
                        }  
        wav_aug = C.get()['aug'][0]
        spec_aug = C.get()['aug'][1]
        ### without wav
        # cv_aug = C.get()['aug'][2]
        mix_aug=C.get()['aug'][2]

        aug_wav=make_wav_aug(wav_aug,audio_conf_train.get('sample_rate'))
        aug_spec=make_spec_aug(spec_aug)
        # aug_cv=make_cv_aug(cv_aug)
        aug_mix=make_mix_aug(mix_aug)
    else:
        raise ValueError('dataset=%s' % dataset)

    # total_aug = augs = None
    # if isinstance(C.get()['aug'], list):
    #     logger.debug('augmentation provided.')
    #     transform_train.transforms.insert(0, Augmentation(C.get()['aug']))
    ### 看配置文件的话是有cutout的但是并不明白为什么要cutout
    # if C.get()['cutout'] > 0:
    #     transform_train.transforms.append(CutoutDefault(C.get()['cutout']))
    ## 开始dataloader数据初始化了，得到total_trainset, testset
    if dataset == 'esc50':
        pass
        # total_trainset=AudioDataset_finetue(,dataset_json_file=conf['total_data'],audio_conf=audio_conf,label_csv=conf['label_csv'])
        # testset=AudioDataset_finetue(,dataset_json_file=conf['total_data'],audio_conf=val_audio_conf,label_csv=conf['label_csv'])
        # total_trainset=AudioDataset(transform=transform_train,dataset_json_file=conf['total_data'],audio_conf=audio_conf,label_csv=conf['label_csv'])
        # testset=AudioDataset(transform=transform_test,dataset_json_file=conf['total_data'],audio_conf=val_audio_conf,label_csv=conf['label_csv'])
    elif dataset == 'BalancedAudioSet':
        total_trainset=AudioDataset_finetue(dataset_json_file=conf['data_train'],audio_conf=audio_conf,label_csv=conf['label_csv'],aug_wav=aug_wav,aug_spec=aug_spec,aug_cv=aug_cv,aug_mix=aug_mix)
        testset=AudioDataset_finetue(dataset_json_file=conf['data_test'],audio_conf=val_audio_conf,label_csv=conf['label_csv'])
    elif dataset == 'audioset':
        total_trainset=AudioDataset_finetue(dataset_json_file=conf.data_train, audio_conf=audio_conf_train, label_csv=conf.label_csv, aug_wav=aug_wav, aug_spec=aug_spec, aug_cv=aug_cv, aug_mix=aug_mix)
        testset=AudioDataset_finetue(dataset_json_file=conf.data_eval, audio_conf=audio_conf_val,label_csv=conf.label_csv)
    elif dataset == 'voc-c':
        total_trainset=AudioDataset_finetue(dataset_json_file=conf.data_train, audio_conf=audio_conf_train, label_csv=conf.label_csv, aug_wav=aug_wav, aug_spec=aug_spec, aug_cv=aug_cv, aug_mix=aug_mix)
        testset=AudioDataset_finetue(dataset_json_file=conf.data_eval, audio_conf=audio_conf_val,label_csv=conf.label_csv)
    elif dataset == 'hcc':
        ### without wav
        # total_trainset=AudioDataset_finetue(dataset_json_file=conf.data_train, audio_conf=audio_conf_train, label_csv=conf.label_csv, aug_wav=aug_wav, aug_spec=aug_spec, aug_cv=aug_cv, aug_mix=aug_mix)
        total_trainset=AudioDataset_finetue(dataset_json_file=conf.data_train, audio_conf=audio_conf_train, label_csv=conf.label_csv, aug_wav=aug_wav, aug_spec=aug_spec, aug_cv=None, aug_mix=aug_mix)
        testset=AudioDataset_finetue(dataset_json_file=conf.data_eval, audio_conf=audio_conf_val,label_csv=conf.label_csv)
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    # if total_aug is not None and augs is not None:
    #     total_trainset.set_preaug(augs, total_aug)
    #     print('set_preaug-')

    ## 数据集划分
    train_sampler = None
    if split > 0.0:
        # sss = StratifiedShuffleSplit(n_splits=5, test_size=split, random_state=0)
        sss = ShuffleSplit(n_splits=5, test_size=split, random_state=0)
        ###
        # sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        ### audioset don't have attributes labels
        sss = sss.split(list(range(len(total_trainset))), total_trainset.labels)
        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)

        ### version problem
        # if target_lb >= 0:
        #     train_idx = [i for i in train_idx if total_trainset.targets[i] == target_lb]
        #     valid_idx = [i for i in valid_idx if total_trainset.targets[i] == target_lb]
        if target_lb >= 0:
            train_idx = [i for i in train_idx if total_trainset.labels[i] == target_lb]
            valid_idx = [i for i in valid_idx if total_trainset.labels[i] == target_lb]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)

        if multinode:
            train_sampler = torch.utils.data.distributed.DistributedSampler(Subset(total_trainset, train_idx), num_replicas=dist.get_world_size(), rank=dist.get_rank())
    else:
        valid_sampler = SubsetSampler([])

        if multinode:
            train_sampler = torch.utils.data.distributed.DistributedSampler(total_trainset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            logger.info(f'----- dataset with DistributedSampler  {dist.get_rank()}/{dist.get_world_size()}')

    ## dataloader
    ### change some parameters
    # trainloader = torch.utils.data.DataLoader(
    #     total_trainset, batch_size=batch, shuffle=True if train_sampler is None else False, num_workers=8, pin_memory=True,
    #     sampler=train_sampler, drop_last=True)
    # validloader = torch.utils.data.DataLoader(
    #     total_trainset, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True,
    #     sampler=valid_sampler, drop_last=False)
    #
    # testloader = torch.utils.data.DataLoader(
    #     testset, batch_size=batch, shuffle=False, num_workers=8, pin_memory=True,
    #     drop_last=False
    # )
    # trainloader = torch.utils.data.DataLoader(
    #     total_trainset, batch_size=conf['batch_size'], shuffle=True if train_sampler is None else False, num_workers=0, pin_memory=True,
    #     sampler=train_sampler, drop_last=True)
    # validloader = torch.utils.data.DataLoader(
    #     total_trainset, batch_size=conf['batch_size'], shuffle=False, num_workers=0, pin_memory=True,
    #     sampler=valid_sampler, drop_last=False)
    trainloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=conf.batch_size, shuffle=True if train_sampler is None else False, num_workers=0, pin_memory=True,
        sampler=train_sampler, drop_last=True)
    validloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=conf.batch_size, shuffle=False, num_workers=0, pin_memory=True,
        sampler=valid_sampler, drop_last=False)

    # testloader = torch.utils.data.DataLoader(
    #     testset, batch_size=conf['batch_size'], shuffle=False, num_workers=0, pin_memory=True,
    #     drop_last=False
    # testloader = torch.utils.data.DataLoader(
    #     testset, batch_size=conf['batch_size'], shuffle=False, num_workers=0, pin_memory=True,
    #     sampler=valid_sampler,drop_last=False
    # )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=conf.batch_size, shuffle=False, num_workers=0, pin_memory=True,
        sampler=valid_sampler,drop_last=False
    )

    return train_sampler, trainloader, validloader, testloader


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)
