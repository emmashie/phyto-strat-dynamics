import numpy as np 
import matplotlib.pyplot as plt
import xarray as xr
from stompy.model.delft import dfm_grid
from stompy.grid import unstructured_grid
import matplotlib.dates as mdates
import os
plt.ion()


ncdir = '/yolo_bypass/emma/sfb_dfm/runs/wy2013_evap/DFM_OUTPUT_wy2013_evap'
#ncdir = '/hpcvol1/emma/sfb_dfm/runs/wy2017-v4/DFM_OUTPUT_wy2017-v4'

#savepath = '/hpcvol1/emma/analysis/stratification_analysis'



fig,ax=plt.subplots(figsize=(10,8))
t = slice(184,273,1)
for i in range(16):
    #nc = "wy2017-v4_%04d_20160801_000000_map.nc" % (i)
    nc = "wy2013_evap_%04d_20120801_000000_map.nc" % (i)    
    #grid=dfm_grid.DFMGrid(os.path.join(ncdir,nc))
    #ds=xr.open_dataset(os.path.join(ncdir,nc))    
    grid=dfm_grid.DFMGrid(os.path.join(ncdir,nc))
    ds=xr.open_dataset(os.path.join(ncdir,nc))
    surf_salt=ds.sa1.isel(time=t, laydim=0).values
    bot_salt=ds.sa1.isel(time=t, laydim=-1).values
    waterdepth=ds.waterdepth.isel(time=t).values
    dsdz_avg = np.mean((surf_salt-bot_salt)/waterdepth,axis=0)
    coll=grid.plot_cells(values=dsdz_avg)
    coll.set_edgecolor('face')
    coll.set_clim(vmin=0,vmax=0.3)
    ax.axis('equal')
    ax.set_ylim((4135850,4175000))
    ax.set_xlim((545436,596287))

plt.colorbar(coll,label='Stratification (psu/m)')
ax.axis('off')

ncdir1 = '/hpcvol1/emma/sfb_dfm/runs/wy2013_temp/DFM_OUTPUT_wy2013_temp'
ncdir2 = '/hpcvol1/emma/sfb_dfm/runs/wy2017-v4/DFM_OUTPUT_wy2017-v4'

nc1 = 'wy2013_temp_0000_20120801_000000_his.nc'
nc2 = 'wy2017-v4_0000_20160801_000000_his.nc'

ds1 = xr.open_dataset(os.path.join(ncdir1,nc1))
ds2 = xr.open_dataset(os.path.join(ncdir2,nc2))

P24_ind = np.where(ds1.station_id.values == b'P24')[0]
P25_ind = np.where(ds1.station_id.values == b'P25')[0]
P27_ind = np.where(ds1.station_id.values == b'P27')[0]
P29_ind = np.where(ds1.station_id.values == b'P29')[0]

dsdz1 = np.squeeze(((ds1.salinity[:,P24_ind,0].values-ds1.salinity[:,P24_ind,-1].values)/ds1.waterdepth[:,P24_ind].values + 
        (ds1.salinity[:,P25_ind,0].values-ds1.salinity[:,P25_ind,-1].values)/ds1.waterdepth[:,P25_ind].values +
        (ds1.salinity[:,P27_ind,0].values-ds1.salinity[:,P27_ind,-1].values)/ds1.waterdepth[:,P27_ind].values +  
        (ds1.salinity[:,P29_ind,0].values-ds1.salinity[:,P29_ind,-1].values)/ds1.waterdepth[:,P29_ind].values)/4)
dsdz2 = np.squeeze(((ds2.salinity[:,P24_ind,0].values-ds2.salinity[:,P24_ind,-1].values)/ds2.waterdepth[:,P24_ind].values + 
        (ds2.salinity[:,P25_ind,0].values-ds2.salinity[:,P25_ind,-1].values)/ds2.waterdepth[:,P25_ind].values +
        (ds2.salinity[:,P27_ind,0].values-ds2.salinity[:,P27_ind,-1].values)/ds2.waterdepth[:,P27_ind].values +  
        (ds2.salinity[:,P29_ind,0].values-ds2.salinity[:,P29_ind,-1].values)/ds2.waterdepth[:,P29_ind].values)/4)
time1 = ds1.time.values
time2 = ds2.time.values


fig, ax = plt.subplots()
ax.plot(time1, dsdz2, color='steelblue', label='Water Year 2017')
ax.plot(time1, dsdz1, color='lightskyblue', label='Water Year 2013')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.legend(loc='best')
ax.set_ylabel('Stratification (psu/m)')