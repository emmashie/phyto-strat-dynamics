import numpy as np 
import matplotlib.pyplot as plt
import xarray as xr 
import os 
from stompy.utils import rotate_to_principal,principal_theta, rot

run_name="wy2017-v4"
begindate = "20160801"
path = "/hpcvol1/emma/sfb_dfm/runs/%s/DFM_OUTPUT_%s/"%(run_name,run_name)
hisfile = os.path.join(path, "%s_0000_%s_000000_his.nc"%(run_name,begindate))

his = xr.open_dataset(hisfile)
smb_ind = np.where(his.station_name.values==b'P27')[0]
dmb_ind = np.where(his.station_name.values==b'P32')[0]

smb = his.isel(stations=smb_ind)
dmb = his.isel(stations=dmb_ind)

dist = np.sqrt((smb.station_x_coordinate.values[0,0]-dmb.station_x_coordinate.values[0,0])**2 + (smb.station_y_coordinate.values[0,0]-dmb.station_y_coordinate.values[0,0])**2)
dsdx = (np.mean(smb.salinity.values[:,0,:],axis=-1)-np.mean(dmb.salinity.values[:,0,:],axis=-1))/dist

h = smb.waterdepth.values[:,0] 

flood_dir = 142.0 ## as per NOAA ADCP data
vel = np.asarray([np.mean(smb.x_velocity.values[:,0,:],axis=-1), np.mean(smb.y_velocity.values[:,0,:],axis=-1)]).T
theta = principal_theta(vel, positive=flood_dir)
rvel = rot(-theta,vel).T 
u_s = 0.1*rvel[0,:]

beta = 7.7*10**-4 
g = 9.8 

Rix = (beta*g*dsdx*h)/(u_s**2)

f = open("%s_rix.csv" % run_name, "w")

f.write("time,dsdx,h,u,rix\n")

for i in range(len(Rix)):
    f.write("%s," % str(smb.time[i].values))
    f.write("%f," % dsdx[i])
    f.write("%f," % h[i])
    f.write("%f," % rvel[0,i])
    f.write("%f\n" % Rix[i])

f.close()