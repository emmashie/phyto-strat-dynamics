import numpy as np 
import matplotlib.pyplot as plt 
import os 
import pandas as pd 
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from datetime import datetime 
import scipy

#google_drive_path = "/mnt/c/Google_Drive"
google_drive_path = "/Users/emmashienuss/Google_Drive"
filepath = "1_Nutrient_Share/1_Projects_NUTRIENTS/06_FY20_NMS_Projects/FY20_Phyto_Blooms_Controls/3_Project_Work_Analysis/aggregating_data/"
file = "sb_data.csv"

flood_only=True

dat = pd.read_csv(os.path.join(google_drive_path,filepath,file))
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
    quants1 = np.arange(q1,q2,step)
    quants2 = np.arange(q1,q2,step) 
    quants1_mid = (quants1[1:]+quants1[:-1])/2    
    p = np.zeros((len(quants1_mid),len(quants2)))
    value1 = np.zeros(len(quants1))
    value2 = np.zeros(len(quants2))
    for i in range(1,len(quants1)):
        for j in range(len(quants2)):
            qcondition = condition[np.where(np.isfinite(chl))[0]]
            q0 = np.quantile(qcondition[np.isfinite(qcondition)],quants1[i-1])
            q1 = np.quantile(qcondition[np.isfinite(qcondition)],quants1[i])
            value1[i-1] = q0
            value1[i] = q1
            #print(q0,q1)
            ind = np.where((condition>=q0)&(condition<q1))[0]
            sub_var = variable[ind]
            qvar = np.quantile(variable[np.isfinite(variable)],quants2[j])
            value2[j] = qvar
            #print(qvar)
            hvar = np.where(sub_var>qvar)[0]
            p[i-1,j] = len(hvar)/len(sub_var)
            #print(quants1[i], quants2[j], len(hvar), len(sub_var))
    return quants1_mid, quants2, value1, value2, p


if 1: ### both quantiles varying 
    var1 = min_Si 
    x_label = '$Ri_x$ quantile'
    var2 = chl
    y_label = 'Chl quantile'
    step = 0.1
    quants1, quants2, values1, values2, p = pdf(var1, var2, chl, q1=0, q2=1, step=step)

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

    q1,q2=np.meshgrid(np.arange(0,1,.01),np.arange(0,1,.01))
    q1a,q1b=np.meshgrid(dqs1[good],q1.flatten())
    q2a,q2b=np.meshgrid(dqs2[good],q2.flatten())
    A=rbf(np.sqrt( (q1a-q1b)**2 + (q2a-q2b)**2))
    A[np.isnan(A)]=0.0
    cont_p = np.dot(A,coefs).reshape(q1.shape)

    fig, ax = plt.subplots(figsize=(4,4),dpi=160)
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
        quants1 = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.05, 0.15,
                           0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.05, 0.15, 0.25, 0.35,
                           0.45, 0.55, 0.65, 0.75, 0.85, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55,
                           0.65, 0.75, 0.85, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75,
                           0.85, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.05,
                           0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.05, 0.15, 0.25,
                           0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.05, 0.15, 0.25, 0.35, 0.45,
                           0.55, 0.65, 0.75, 0.85, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65,
                           0.75, 0.85])
        quants2 = np.array([0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.1, 0.1, 0.1, 0.1,
                           0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                           0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4,
                           0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                           0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7,
                           0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
                           0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        coefs = np.array([-1.91317929e+00, -2.30043504e-01, -6.96980325e-01,  5.39240433e-01,
                           -3.25537943e-01, -8.11633058e-01, -5.33654436e-01, -3.46365793e-01,
                           -6.62099257e-02,  2.10413632e+00, -1.71052243e-02,  2.35240112e+00,
                           -2.43688704e+00,  8.39713962e-02,  5.76026499e+00, -6.85760330e+00,
                            4.50453604e+00, -1.22761378e+00,  3.53710790e-01, -1.17800940e+00,
                           -4.04549375e+00,  7.60899447e+00, -9.37158256e+00,  3.31327507e+00,
                            2.44003678e+00, -3.62584649e-02, -2.10103553e+00,  1.37294604e+00,
                            6.68846085e-01, -3.67943773e+00,  6.15946706e+00, -1.92572014e+00,
                           -7.94305201e-01, -1.02679711e+00,  3.93781948e-03, -4.40976076e-01,
                           -8.98093763e-01, -3.26792190e+00,  3.08181872e+00, -9.37207599e-01,
                            5.36489972e-01, -1.82235687e+00,  4.80832002e+00, -3.59661784e+00,
                            4.56391461e+00,  2.25674895e+00,  5.42816730e+00, -6.86696297e+00,
                            2.02348723e+00,  3.04652645e+00, -8.45808703e+00,  4.06681706e+00,
                            5.82981609e-01, -5.11970468e+00, -9.49301117e+00,  2.66164377e+00,
                            2.56170346e-01,  8.71425939e-01,  1.58908919e+00,  9.32014971e-01,
                           -4.62080252e+00,  8.12631069e+00, -1.49275788e+00,  6.05132739e+00,
                           -2.55380637e+00,  7.47460144e+00, -8.50724438e+00,  6.14186524e+00,
                           -1.12099994e+01,  1.38257717e+01, -1.45663250e+01,  5.30745696e+00,
                           -2.86229026e+00, -6.32209195e+00,  2.40182185e+00, -5.16735075e+00,
                            8.42814123e+00, -4.24643592e+00,  2.06058763e+00, -1.42711133e+00,
                            5.86410734e-01,  5.76173376e+00, -2.82272550e+00,  3.74955426e+00,
                           -1.42364646e+00,  8.20157836e-01, -2.28908594e+00, -9.06086042e-01,
                            2.81032654e+00, -1.20462226e-01])
        q1a,q1b=np.meshgrid(quants1,q1)
        q2a,q2b=np.meshgrid(quants2,q2)
        A=rbf(np.sqrt( (q1a-q1b)**2 + (q2a-q2b)**2))
        p = np.dot(A, coefs)
        return p



