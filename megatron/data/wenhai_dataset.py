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
from metpy.calc import specific_humidity_from_dewpoint
from metpy.units import units
from aerobulk.flux import noskin_np
import xarray as xr
import multiprocessing

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
        self.mask = torch.load('/test2/cuiyz/data/mask_GLORYS_0.083d.pt').bool()
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




class WenhaiEvaluateDataset(data.Dataset):
    def __init__(self, status = 0):
        """
        root is the directory for *.pt
        nin is the number of the preceding information/day(s) to produce forecasts
        nout is the number of days to forecasts
        status is the flag indicating training/test/validation (0,1,2)
        """
        root = '/test2/output_norm/data_S5-40/'
        root_9var = '/test2/use_daily/'
        status_str = {0: 'Training', 1: 'Test', 2: 'Validation'}
        files = glob(root + '*.pt')
        files.sort()
        files_d2m = glob(root_9var+'d2m/'+'*.nc')
        files_msl = glob(root_9var+'msl/'+'*.nc')
        files_mtpr = glob(root_9var+'mtpr/'+'*.nc')
        files_siconc = glob(root_9var+'siconc/'+'*.nc')
        files_ssr = glob(root_9var+'ssr/'+'*.nc')
        files_strd = glob(root_9var+'strd/'+'*.nc')
        files_t2m = glob(root_9var+'t2m/'+'*.nc')
        files_u10 = glob(root_9var+'u10/'+'*.nc')
        files_v10 = glob(root_9var+'v10/'+'*.nc')
        files_d2m.sort()
        files_msl.sort()
        files_mtpr.sort()
        files_siconc.sort()
        files_ssr.sort()
        files_strd.sort()
        files_t2m.sort()
        files_u10.sort()
        files_v10.sort()
        
        self.lead_time=0
        self.status = status
        idx1,idx2 = 9496,len(files)
        #idx1, idx2= 9496, 9496+365 
        if status == 0:  # training set 5 yrs    # training set 10 yrs
            self.files = files[:idx1]
            self.files_d2m = files_d2m[:idx1]
            self.files_msl = files_msl[:idx1]
            self.files_mtpr = files_mtpr[:idx1]
            self.files_siconc = files_siconc[:idx1]
            self.files_ssr = files_ssr[:idx1]
            self.files_strd = files_strd[:idx1]
            self.files_t2m = files_t2m[:idx1]
            self.files_u10 = files_u10[:idx1]
            self.files_v10 = files_v10[:idx1]
            self.length = idx1-self.lead_time

        elif status == 1: # test set 2 yr
            self.files = files[idx1:]
            self.files_d2m = files_d2m[idx1:]
            self.files_msl = files_msl[idx1:]
            self.files_mtpr = files_mtpr[idx1:]
            self.files_siconc = files_siconc[idx1:]
            self.files_ssr = files_ssr[idx1:]
            self.files_strd = files_strd[idx1:]
            self.files_t2m = files_t2m[idx1:]
            self.files_u10 = files_u10[idx1:]
            self.files_v10 = files_v10[idx1:]
            self.length = idx2 - idx1 - self.lead_time

        self.max = torch.load('/test2/cuiyz/data/max_GLORYS_S5-40.pt')[:,None,None]
        self.min = torch.load('/test2/cuiyz/data/min_GLORYS_S5-40.pt')[:,None,None]
        self.bulkmax = torch.load('/test2/cuiyz/data/bulk_flux_surface_max.pt')[:,None,None]
        self.bulkmin = torch.load('/test2/cuiyz/data/bulk_flux_surface_min.pt')[:,None,None]
        self.mask = torch.load('/test2/cuiyz/data/mask_GLORYS_0.083d.pt')
    def __len__(self):
        return self.length

    def isel(self, da):
        return da[0].fillna(0).values*self.mask[0].numpy()

    def mask_tensor(self, var):
        return torch.tensor(var)
 
    def calc_bulk_flux(self, tensor, idx):
        tensor = tensor.clone()
        #tensor=self.tensor.clone()
        ds_d2m = xr.open_dataset(self.files_d2m[idx])
        ds_msl = xr.open_dataset(self.files_msl[idx])
        ds_mtpr = xr.open_dataset(self.files_mtpr[idx])
        ds_siconc = xr.open_dataset(self.files_siconc[idx])
        ds_ssr = xr.open_dataset(self.files_ssr[idx])
        ds_strd = xr.open_dataset(self.files_strd[idx])
        ds_t2m = xr.open_dataset(self.files_t2m[idx])
        ds_u10 = xr.open_dataset(self.files_u10[idx])
        ds_v10 = xr.open_dataset(self.files_v10[idx])
        sst = ((self.max[46] - self.min[46])*tensor[46].type(torch.float32) + self.min[46])*self.mask[46]+273.15
        #mask = sst == 273.25
        sst[sst==273.2500] = 0
        uo = ((self.max[0] - self.min[0])*tensor[0].type(torch.float32)+self.min[0])*self.mask[0]
        vo = ((self.max[23] - self.min[23])*tensor[23].type(torch.float32)+self.min[23])*self.mask[23]
        pool = multiprocessing.Pool(processes=9)

        d2m=pool.apply_async(self.isel,(ds_d2m.d2m, ))
        mtpr=pool.apply_async(self.isel,(ds_mtpr.mtpr, ))
        ssr=pool.apply_async(self.isel,(ds_ssr.ssr, ))
        strd=pool.apply_async(self.isel,(ds_strd.strd, ))
        t2m=pool.apply_async(self.isel,(ds_t2m.t2m, ))
        u10=pool.apply_async(self.isel,(ds_u10.u10, ))
        v10=pool.apply_async(self.isel,(ds_v10.v10, ))
        try:
            msl=pool.apply_async(self.isel,(ds_msl.MSL, ))
        except:
            msl = pool.apply_async(self.isel,(ds_msl.msl,))
        sic=pool.apply_async(self.isel,(ds_siconc.siconc, ))


        pool.close()
        pool.join()

        d2m=d2m.get()
        mtpr=mtpr.get()
        ssr=ssr.get()
        strd=strd.get()
        t2m=t2m.get()
        u10=u10.get()
        v10=v10.get()
        msl=msl.get()
        sic=sic.get()

        sic = np.zeros_like(sic)
        ssr /= 3600
        strd /= 3600
        mtpr /= 1000
        sst *= self.mask[0]

        uo *= self.mask[0]
        vo *= self.mask[0]
        h2m = specific_humidity_from_dewpoint(msl * units.Pa, d2m * units.K).to('kg/kg').magnitude
        h2m = np.nan_to_num(h2m)

        def calculate_noskin_np(args):
            sst, t2m, h2m, u, v, msl = args
            return noskin_np(sst, t2m, h2m, u, v, msl, 'ncar', 2, 10, 4, False)
        
        pool = multiprocessing.Pool(processes=9)
        results = pool.map(calculate_noskin_np, [[sst.numpy()[:,i], t2m[:,i], h2m[:,i], u10[:,i]-uo[:,i].numpy(), v10[:,i]-vo[:,i].numpy(), msl[:,i]] for i in range(4320)])
        pool.close()
        pool.join()
        
        # Reshape the results back into the original shape
        qe, qh, taux, tauy, evap = zip(*results)
        qe = np.array(qe).T.reshape(sst.shape)
        qh = np.array(qh).T.reshape(sst.shape)
        taux = np.array(taux).T.reshape(sst.shape)
        tauy = np.array(tauy).T.reshape(sst.shape)
        evap = np.array(evap).T.reshape(sst.shape)

        qe=qe*(1-sic)
        qh=qh*(1-sic)
        taux=taux*(1-sic)
        tauy=tauy*(1-sic)
        evap=evap*(1-sic)
        evap /= 1000
        rho0 = 1026 # kg/m^3
        cp = 3996 # J/kg/K
        R = 0.58
        xi0 = 0.35
        xi1 = 23
        sigma = 5.67e-8
        ql = (1-torch.from_numpy(sic))*(torch.from_numpy(strd)-sigma*(sst**4))
        qs = torch.from_numpy(ssr)
        ql,qs,qh,qe,taux,tauy,evap,mtpr = map(torch.tensor,[ql,qs,qh,qe,taux,tauy,evap,mtpr])
        bulk_flux = torch.stack((ql,qs,qh,qe,taux,tauy,evap,mtpr), dim = 0)
        bulk_flux = torch.nan_to_num(bulk_flux)
        bulk_flux = (bulk_flux - self.bulkmin)/(self.bulkmax-self.bulkmin)
        bulk_flux *= self.mask[0][None]
        ds_d2m.close()
        ds_msl.close()
        ds_mtpr.close()
        ds_siconc.close()
        ds_ssr.close()
        ds_strd.close()
        ds_t2m.close()
        ds_u10.close()
        ds_v10.close()
        return bulk_flux
    
    def __getitem__(self, idx):
        self.tensor=torch.load(self.files[idx])
        bulk_flux = self.calc_bulk_flux(idx)
        return self.tensor, bulk_flux


def build_pretrain_train_test_dataset():
    train_dataset = WenhaiPretrainDataset(status=0)
    test_dataset = WenhaiPretrainDataset(status=1)
    return train_dataset, test_dataset


def build_finetune_train_test_dataset(finetune_days):
    train_dataset = WenhaiFinetuneDataset(status = 0, finetune_days=finetune_days)
    test_dataset = WenhaiFinetuneDataset(status = 1, finetune_days=finetune_days)
    return train_dataset, test_dataset


def build_evaluate_dataset():
    eval_dataset = WenhaiEvaluateDataset(status = 1)
    return eval_dataset