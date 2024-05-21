
import xarray as xr
import multiprocessing
from glob import glob 
import torch
import numpy as np
from metpy.calc import specific_humidity_from_dewpoint
from metpy.units import units
from aerobulk.flux import noskin_np

root_9var = '/test2/use_daily/'
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
files_v10.sort()#大气文件，原来是9个，现在形成bulk formula


max = torch.load('/test2/cuiyz/data/max_GLORYS.pt')#[93]
min = torch.load('/test2/cuiyz/data/min_GLORYS.pt')#[93]
mask = torch.load('/test2/cuiyz/data/mask_GLORYS_0.083d.pt')#[93, 2041, 4320]  # 和上面那个区别开来，一个在CPU，一个在CPU

bulkmax = torch.load('/test2/cuiyz/data/bulk_flux_surface_max.pt')[:,None,None]
bulkmin = torch.load('/test2/cuiyz/data/bulk_flux_surface_min.pt')[:,None,None]

def isel(da):
    return da[0].fillna(0).values*mask[0].numpy()


def calculate_noskin_np(args):
    sst, t2m, h2m, u, v, msl = args
    return noskin_np(sst, t2m, h2m, u, v, msl, 'ncar', 2, 10, 4, False)

def calc_bulk_flux(tensor, idx):#注意tensor的shape是[1, 93, xx, xx]还是[93, xx, xx]
    ds_d2m = xr.open_dataset(files_d2m[idx])
    ds_msl = xr.open_dataset(files_msl[idx])
    ds_mtpr = xr.open_dataset(files_mtpr[idx])
    ds_siconc = xr.open_dataset(files_siconc[idx])
    ds_ssr = xr.open_dataset(files_ssr[idx])
    ds_strd = xr.open_dataset(files_strd[idx])
    ds_t2m = xr.open_dataset(files_t2m[idx])
    ds_u10 = xr.open_dataset(files_u10[idx])
    ds_v10 = xr.open_dataset(files_v10[idx])


    sst = ((max[46] - min[46])*tensor[0][46].type(torch.float32) + min[46])*mask[46]+273.15#[2041, 4320]
    sst[sst==273.2500] = 0
    uo = ((max[0] - min[0])*tensor[0][0].type(torch.float32)+min[0])*mask[0]
    vo = ((max[23] - min[23])*tensor[0][23].type(torch.float32)+min[23])*mask[23]


    if False:
        d2m, mtpr, ssr, strd, t2m, u10, v10, msl, sic = map(isel, [ds_d2m.d2m, ds_mtpr.mtpr, ds_ssr.ssr, ds_strd.strd, ds_t2m.t2m, ds_u10.u10, ds_v10.v10, ds_msl.MSL, ds_siconc.siconc])
        print('d2m sum', d2m.sum())#1713013600.0

    #上面这个函数4.7秒，是主要的优化热点
    #考虑多线程并发？
    else:
        pool = multiprocessing.Pool(processes=9)

        d2m=pool.apply_async(isel,(ds_d2m.d2m, ))
        mtpr=pool.apply_async(isel,(ds_mtpr.mtpr, ))
        ssr=pool.apply_async(isel,(ds_ssr.ssr, ))
        strd=pool.apply_async(isel,(ds_strd.strd, ))
        t2m=pool.apply_async(isel,(ds_t2m.t2m, ))
        u10=pool.apply_async(isel,(ds_u10.u10, ))
        v10=pool.apply_async(isel,(ds_v10.v10, ))
        try:
            msl=pool.apply_async(isel,(ds_msl.MSL, ))
        except:
            msl=pool.apply_async(isel,(ds_msl.msl, ))
        sic=pool.apply_async(isel,(ds_siconc.siconc, ))


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
    sst *= mask[0]
    uo *= mask[0]
    vo *= mask[0]

    h2m = specific_humidity_from_dewpoint(msl * units.Pa, d2m * units.K).to('kg/kg').magnitude
    #从开头到这里时间为5.3秒
    #上面这个函数0.12秒


    #start_time_profile=time.time()#--------------------------
    if False:
        qe, qh, taux, tauy, evap = noskin_np(sst.numpy(), t2m, h2m, u10-uo.numpy(), v10-vo.numpy(), msl, 'ncar', 2, 10, 4, False)
    #这个函数时间为11秒
    else:
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
        
        # qe, qh, taux, tauy, evap = np.ones_like(sst.shape), np.ones_like(sst.shape), np.ones_like(sst.shape), np.ones_like(sst.shape), np.ones_like(sst.shape)
    #end_time_profile=time.time()#--------------------------
    #print(f"profile时间: {end_time_profile - start_time_profile}秒")
    


    #从这里到结尾只有0.27秒，不用管
    qe=qe*(1-sic)
    qh=qh*(1-sic)
    taux=taux*(1-sic)
    tauy=tauy*(1-sic)
    evap=evap*(1-sic)
    evap /= 1000
    sigma = 5.67e-8
    ql = (1-torch.from_numpy(sic))*(torch.from_numpy(strd)-sigma*(sst**4))
    qs = torch.from_numpy(ssr)
    ql,qs,qh,qe,taux,tauy,evap,mtpr = map(torch.tensor,[ql,qs,qh,qe,taux,tauy,evap,mtpr])
    bulk_flux = torch.stack((ql,qs,qh,qe,taux,tauy,evap,mtpr), dim = 0)
    bulk_flux = torch.nan_to_num(bulk_flux)
    bulk_flux = (bulk_flux - bulkmin)/(bulkmax-bulkmin)
    bulk_flux = bulk_flux*mask[0][None]

    ds_d2m.close()
    ds_msl.close()
    ds_mtpr.close()
    ds_siconc.close()
    ds_ssr.close()
    ds_strd.close()
    ds_t2m.close()
    ds_u10.close()
    ds_v10.close()

    return bulk_flux.unsqueeze(dim=0).type(torch.float16)