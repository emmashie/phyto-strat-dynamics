import numpy as np 
import matplotlib.pyplot as plt 
import os 
import pandas as pd 
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from datetime import datetime 
import scipy

#google_drive_path = "/mnt/c/Google_Drive"
#google_drive_path = "/Users/emmashienuss/Google_Drive"
#filepath = "1_Nutrient_Share/1_Projects_NUTRIENTS/06_FY20_NMS_Projects/FY20_Phyto_Blooms_Controls/3_Project_Work_Analysis/aggregating_data/"
filepath = "data"
file = "sb_data.csv"

flood_only=True

dat = pd.read_csv(os.path.join(filepath,file))
dat['Date'] = pd.to_datetime(dat['Date'])

if flood_only==True:
    find = np.where(dat['cruise_tidal_velocity']>0)[0]
    #dat = dat.copy().iloc[find].reset_index()

dates = dat.Date.values
dsdz = -(dat.sta24_dsdz.values+dat.sta25_dsdz.values+dat.sta27_dsdz.values+dat.sta29_dsdz.values)/4
chl = (dat.sta24_chl.values+dat.sta25_chl.values+dat.sta27_chl.values+dat.sta29_chl.values)/4

## calculate u* - friction velocity
## u* = sqrt(tau/rho)
## "A general rule is that the shear velocity is about 1/10 of the mean flow velocity.""
## from (https://www.fxsolver.com/browse/formulas/Friction+velocity+%28shear+velocity%29)

u_s = dat.tidal_velocity.values*0.1

### omega -- tidal frequency 
omega = 1/(12.4206012*3600)

### H water depth 
H = 11 

### salinity expansion coefficient
beta = 7.7*10**-4

### gravity
g = 9.8 

### salinity gradient 
## starting with 27-32 ds/dx
dsdx = np.abs(dat.dsdx_27_32.values)/1000 ## psu/km to psu/m

### unsteadiness parameter
Un = omega*H/u_s

### simpson number 
Si = (beta*g*dsdx*H**2)/(u_s**2)

### flow metric
qu3 = dat.Alameda_cfs.values/dat.tidal_velocity.values**3

### playing with what was the max Si or qu3 preceeding chlorophyll data
max_Si = np.zeros(len(Si))
max_qu3 = np.zeros(len(qu3))
min_Si = np.zeros(len(Si))
window = 3

for i in range(window,len(chl)):
    if np.isfinite(chl[i])==True:
        max_Si[i] = np.nanmax(Si[i-window:i])
        max_qu3[i] = np.nanmax(qu3[i-window:i])
        min_Si[i] = np.nanmin(Si[i-window:i])

def condition_ind(condition, variable, quantile=0.75, var_q=0.9):
    qcondition = condition[np.where(np.isfinite(variable))[0]]
    q = np.quantile(qcondition[np.isfinite(qcondition)],quantile)
    ind = np.where(condition>q)[0]
    sub_var = variable[ind]
    q2 = np.quantile(variable[np.isfinite(variable)],var_q)
    hvar = np.where(sub_var>q2)[0]
    p = len(hvar)/len(sub_var)
    return ind, p


def pdf(condition, variable, chl, q1=0, q2=1, step=0.1):
    quants1 = np.arange(q1,q2+0.1,step)
    quants2 = np.arange(q1,q2,step) 
    #quants1_mid = (quants1[1:]+quants1[:-1])/2    
    quants1_mid = quants1+0.05    
    p = np.zeros((len(quants1_mid),len(quants2)))
    value1 = np.zeros(len(quants1)+1)
    value2 = np.zeros(len(quants2))
    for i in range(1,len(quants1)):
        for j in range(len(quants2)):
            qcondition = condition[np.where(np.isfinite(chl))[0]]
            q0 = np.quantile(qcondition[np.isfinite(qcondition)],quants1[i-1])
            q1 = np.quantile(qcondition[np.isfinite(qcondition)],quants1[i])
            value1[i-1] = q0
            value1[i] = q1
            print(quants1[i-1], quants1[i])
            #print(q0,q1)
            ind = np.where((condition>=q0)&(condition<q1))[0]
            sub_var = variable[ind]
            qvar = np.quantile(variable[np.isfinite(variable)],quants2[j])
            value2[j] = qvar
            #print(qvar)
            hvar = np.where(sub_var>qvar)[0]
            p[i-1,j] = len(hvar)/len(sub_var)
            print(quants1[i], quants2[j], len(hvar), len(sub_var))
    return quants1_mid, quants2, value1, value2, p


if 1: ### both quantiles varying 
    var1 = min_Si 
    x_label = '$Ri_x$ quantile'
    var2 = chl
    y_label = 'Chl quantile'
    step = 0.1
    quants1, quants2, values1, values2, p = pdf(var1, var2, chl, q1=0, q2=1, step=step)
    # remove extra zero column 
    p = p[0:-1,:]
    quants1 = quants1[:-1]


    dataplot = np.append(p,p[:,-1][:,np.newaxis],axis=1)
    dataplot = np.append(dataplot,dataplot[-1,:][np.newaxis,:],axis=0)
    fig, ax = plt.subplots(figsize=(4,4),dpi=160)
    cb = ax.pcolormesh(np.append(quants1,np.array([1])),
                       np.append(quants2,np.array([1])),
                       dataplot.T,vmin=0,vmax=1)
    ct = ax.contour(np.append(quants1,np.array([1]))+step/2,
                    np.append(quants2,np.array([1]))+step/2,
                       dataplot.T,
                    levels=[0.1,0.2,0.4,0.6,0.8,0.9,1],colors='k')
    ct.levels = [0.1,0.2,0.4,0.6,0.8,0.9,1]
    ax.clabel(ct,ct.levels,fmt='%r',inline=True,fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim([quants1[1],quants1[-1]])
    ax.set_ylim([quants2[0]+step/2,quants2[-1]+step/2])
    fig.colorbar(cb,ax=ax,label='Probability')


    ## create radial basis function
    dqs1 = np.hstack([quants1]) ## use quantiles
    dqs2 = np.hstack([quants2]) ## use quantiles
    #dqs1 = np.hstack([values1[1:]]) ## use values
    #dqs2 = np.hstack([values2]) ## use values
    dqs1, dqs2 = np.meshgrid(dqs1, dqs2)
    dp = dqs1*0
    dp = p.T

    def rbf(r,epsilon=0.5):
        return r*r*scipy.log(r)

    good = np.isnan(dp)==False
    q1a,q1b=np.meshgrid(dqs1[good],dqs1[good])
    q2a,q2b=np.meshgrid(dqs2[good],dqs2[good])
    A=rbf(np.sqrt( (q1a-q1b)**2 + (q2a-q2b)**2))
    A[np.isnan(A)]=0.0
    coefs=np.linalg.solve(A,dp[good])

    #q1,q2=np.meshgrid(np.arange(np.min(values1),np.max(values1),.0005),np.arange(np.min(values2),np.max(values2),.1))
    q1,q2=np.meshgrid(np.arange(0,1,.01),np.arange(0,1,.01))
    q1a,q1b=np.meshgrid(dqs1[good],q1.flatten())
    q2a,q2b=np.meshgrid(dqs2[good],q2.flatten())
    A=rbf(np.sqrt( (q1a-q1b)**2 + (q2a-q2b)**2))
    A[np.isnan(A)]=0.0
    cont_p = np.dot(A,coefs).reshape(q1.shape)

    fig, ax = plt.subplots(figsize=(5,4),dpi=160)
    cb = ax.pcolor(q1,q2,cont_p)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.colorbar(cb,ax=ax,label='Probability')
    ct = ax.contour(np.append(quants1,np.array([1]))+step/2,
                    np.append(quants2,np.array([1]))+step/2,
                       dataplot.T,
                    levels=[0.1,0.2,0.4,0.6,0.8,0.9,1],colors='k')
    ct.levels = [0.1,0.2,0.4,0.6,0.8,0.9,1]
    ax.clabel(ct,ct.levels,fmt='%r',inline=True,fontsize=8)
    ax.set_xlim([quants1[1],quants1[-1]])
    ax.set_ylim([quants2[0]+step/2,quants2[-1]+step/2])    
    fig.tight_layout()



    def prob(q1,q2):
        quants1 = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.05,
                           0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.05, 0.15,
                           0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.05, 0.15, 0.25,
                           0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.05, 0.15, 0.25, 0.35,
                           0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.05, 0.15, 0.25, 0.35, 0.45,
                           0.55, 0.65, 0.75, 0.85, 0.95, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55,
                           0.65, 0.75, 0.85, 0.95, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65,
                           0.75, 0.85, 0.95, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75,
                           0.85, 0.95, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85,
                           0.95])
        quants2 = np.array([0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.1, 0.1, 0.1,
                           0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                           0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                           0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.5,
                           0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6,
                           0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
                           0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9,
                           0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        coefs = np.array([ -1.33261121,  -0.37979314,  -0.61516971,   0.58401462,
                          -0.27227876,  -0.74125522,  -0.48948123,  -0.19523555,
                           0.62552186,  -0.91076862,   2.00243371,  -0.24177169,
                           2.31567172,  -2.50457412,   0.02072829,   5.72814133,
                          -7.03718417,   4.93070756,  -1.70757821,  -1.45860639,
                           0.45282697,  -1.24595952,  -4.00485686,   7.62016792,
                          -9.3644737 ,   3.36732825,   2.25659875,   0.98124795,
                          -5.09099303,   5.20471553,   1.43559922,   0.57683245,
                          -3.66330814,   6.15299268,  -1.93613148,  -0.76269032,
                          -1.1777783 ,   0.5667428 ,  -0.87649641,  -0.6982582 ,
                          -0.82905545,  -3.34819704,   3.10230506,  -0.93884601,
                           0.5303743 ,  -1.78766584,   4.67709399,  -3.1743464 ,
                           5.13921924,  -3.95839631,   2.32473206,   5.34792052,
                          -6.84907557,   2.02010078,   3.03726184,  -8.41711875,
                           3.85779107,   1.60825065,  -8.14520375,   6.03553693,
                          -9.42160631,   2.58482468,   0.27771879,   0.87226979,
                           1.58636147,   0.97281291,  -4.77532219,   8.87393552,
                          -2.66465045,  -0.70817299,   6.12115126,  -2.64302443,
                           7.48094738,  -8.52503078,   6.11948162, -11.2105359 ,
                          13.71379769, -14.42733406,   5.66784233,  -0.33079916,
                          -2.76162666,  -6.36309051,   2.47416878,  -5.11004525,
                           8.49543693,  -4.13129662,   2.08828366,  -0.86521091,
                           1.73383114,  -2.52477786,   5.87943287,  -2.95184984,
                           3.72443149,  -1.46612246,   0.77334781,  -2.30873476,
                          -1.11323802,   3.69288928,  -3.74550575,   3.18823424])
        q1a,q1b=np.meshgrid(quants1,q1)
        q2a,q2b=np.meshgrid(quants2,q2)
        A=rbf(np.sqrt( (q1a-q1b)**2 + (q2a-q2b)**2))
        p = np.dot(A, coefs)
        return p



