import warnings
import time
warnings.filterwarnings("ignore")
import numpy as np
from tqdm import tqdm
import os
import random
import torch
import torch.utils.data as data
import sys
from torch.utils.data import DataLoader
# import xarray as xr
from glob import glob
from typing import Optional

class WenhaiPretrainDataset(data.Dataset):
    def __init__(self, status = 0):
        """
        root is the directory for *.pt
        nin is the number of the preceding information/day(s) to produce forecasts
        nout is the number of days to forecasts
        status is the flag indicating training/test/validation (0,1,2)
        """
        #super(dataset, self).__init__()
        root = '/test2/output_norm/data_S32-38/'
        root_bulk = '/test2/cuiyz/data/bulk_flux_surface_norm/'
        status_str = {0: 'Training', 1: 'Test', 2: 'Validation'}
        files = glob(root + '*.pt')
        files.sort()
        status_str = {0: 'Training', 1: 'Test', 2: 'Validation'}
        #files_bulk = glob(root_bulk + '*.pt')
        #files_bulk.sort()
        files_bulk = [root_bulk + f'{i}.pt' for i in range(len(files))]
        nfiles = len(files)
        self.mask = torch.load('/test2/cuiyz/data/mask_GLORYS_0.083d.pt').half() 
        self.loss_weight = torch.load('/test2/cuiyz/data/weight.pt').half()
        self.lead_time=1
        self.status = status
        idx1, idx2= 9496, 9496+365 
        if status == 0: 
            self.files = files[:idx1]
            self.files_bulk = files_bulk[:idx1]
            self.length = idx1-self.lead_time

        elif status == 1:
            self.files = files[idx1:]
            self.files_bulk = files_bulk[idx1:]
            self.length = idx2 - idx1 - self.lead_time

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = 1
        self.tensor=torch.load(self.files[idx])
        label_tensor=torch.load(self.files[idx+self.lead_time])
        bulk_flux = torch.load(self.files_bulk[idx]) * self.mask[0][None]
        return self.tensor, label_tensor, bulk_flux, self.mask, self.loss_weight


class WenhaiFinetuneDataset(data.Dataset):
    def __init__(self, status = 0, finetune_days=5):
        """
        root is the directory for *.pt
        nin is the number of the preceding information/day(s) to produce forecasts
        nout is the number of days to forecasts
        status is the flag indicating training/test/validation (0,1,2)
        """
        #super(dataset, self).__init__()
        root = '/test2/output_norm/data_S5-40/'
        root_bulk = '/test2/cuiyz/data/bulk_flux_surface_norm/'
        status_str = {0: 'Training', 1: 'Test', 2: 'Validation'}
        files = glob(root + '*.pt')
        files.sort()
        status_str = {0: 'Training', 1: 'Test', 2: 'Validation'}
        #files_bulk = glob(root_bulk + '*.pt')
        #files_bulk.sort()
        files_bulk = [root_bulk + f'{i}.pt' for i in range(len(files))]
        nfiles = len(files)
        self.mask = torch.load('/test2/cuiyz/data/mask_GLORYS_0.083d.pt').half()
        # self.mask_at_cuda = self.mask.half().cuda()
        self.lead_time = finetune_days
        self.status = status
        idx1, idx2 = 9496, 9496+365 
        if status == 0: 
            self.files = files[:idx1]
            self.files_bulk = files_bulk[:idx1]
            self.length = idx1-self.lead_time

        elif status == 1:
            self.files = files[idx1:]
            self.files_bulk = files_bulk[idx1:]
            self.length = idx2 - idx1 - self.lead_time

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        days = []
        for i in range(self.lead_time+1):
            days.append(torch.load(self.files[idx + i]))
        # real_day_0=torch.load(self.files[idx])
        # real_day_1=torch.load(self.files[idx+1])
        # real_day_2=torch.load(self.files[idx+2])
        # real_day_3=torch.load(self.files[idx+3])
        # real_day_4=torch.load(self.files[idx+4])
        # real_day_5=torch.load(self.files[idx+5])
        # days = [real_day_0, real_day_1, real_day_2, real_day_3, real_day_4, real_day_5]

        bulk_flux_day_0 = torch.load(self.files_bulk[idx]) * self.mask[0][None]
        #fine-tune 5天一共需要神经网络计算5次，第一次的bulk formula直接读取预加载好的，后面的才是需要计算的

        #至于大气变量。。能不能直接在这里load?感觉loader交给CPU会更快一点？

        return days,  bulk_flux_day_0, self.mask, idx


def build_pretrain_train_test_dataset():
    train_dataset = WenhaiPretrainDataset(status=0)
    test_dataset = WenhaiPretrainDataset(status=1)
    return train_dataset, test_dataset


def build_finetune_train_test_dataset(finetune_days):
    train_dataset = WenhaiFinetuneDataset(status = 0, finetune_days=finetune_days)
    test_dataset = WenhaiFinetuneDataset(status = 1, finetune_days=finetune_days)
    return train_dataset, test_dataset