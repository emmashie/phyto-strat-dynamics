import numpy as np 
import matplotlib.pyplot as plt 
import os 
import pandas as pd 
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from datetime import datetime, date 
import scipy
from matplotlib.dates import DateFormatter

def rbf(r,epsilon=0.5):
    return r*r*scipy.log(r)

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


datapath = "data/"
wy2017 = "wy2017-v4_rix.csv" 
wy2013 = "wy2013_temp_rix.csv"
obs = "sb_data.csv"

### constants
### salinity expansion coefficient
beta = 7.7*10**-4
### gravity
g = 9.8 

mod17 = pd.read_csv(os.path.join(datapath,wy2017))
mod17['time'] = pd.to_datetime(mod17['time'])
time17 = mod17.resample('D', on='time').agg('mean').index
dsdx17 = mod17.resample('D', on='time').agg('mean')['dsdx'].values
h17 = mod17.resample('D', on='time').agg('mean')['h'].values
u17 = mod17.resample('D', on='time').agg('max')['u'].values
rix17 = (beta*g*np.abs(dsdx17)*h17**2)/(0.1*u17)**2
min_rix17 = np.zeros(len(rix17))
window = 3
for i in range(window,len(rix17)):
    min_rix17[i] = np.nanmin(rix17[i-window:i])


mod13 = pd.read_csv(os.path.join(datapath,wy2013))
mod13['time'] = pd.to_datetime(mod13['time'])
time13 = mod13.resample('D', on='time').agg('mean').index
dsdx13 = mod13.resample('D', on='time').agg('mean')['dsdx'].values
h13 = mod13.resample('D', on='time').agg('mean')['h'].values
u13 = mod13.resample('D', on='time').agg('max')['u'].values
rix13 = (beta*g*np.abs(dsdx13)*h13**2)/(0.1*u13)**2
min_rix13 = np.zeros(len(rix13))
window = 3
for i in range(window,len(rix13)):
    min_rix13[i] = np.nanmin(rix13[i-window:i])


dat = pd.read_csv(os.path.join(datapath,obs))
dat['Date'] = pd.to_datetime(dat['Date'])
dates = dat.Date.values
dates13_ind = np.where((dates>np.datetime64('2012-08-01'))&(dates<np.datetime64('2013-10-01')))[0]
dates17_ind = np.where((dates>np.datetime64('2016-08-01'))&(dates<np.datetime64('2017-10-01')))[0]
dsdz = -(dat.sta24_dsdz.values+dat.sta25_dsdz.values+dat.sta27_dsdz.values+dat.sta29_dsdz.values)/4
chl = (dat.sta24_chl.values+dat.sta25_chl.values+dat.sta27_chl.values+dat.sta29_chl.values)/4
## calculate u* - friction velocity
## u* = sqrt(tau/rho)
## "A general rule is that the shear velocity is about 1/10 of the mean flow velocity.""
## from (https://www.fxsolver.com/browse/formulas/Friction+velocity+%28shear+velocity%29)
u_s = dat.tidal_velocity.values*0.1
### H water depth 
H = 11 
### salinity gradient 
## starting with 27-32 ds/dx
dsdx = np.abs(dat.dsdx_27_32.values)/1000 ## psu/km to psu/m
### simpson number 
Si = (beta*g*dsdx*H**2)/(u_s**2)
### playing with what was the max Si or qu3 preceeding chlorophyll data
max_Si = np.zeros(len(Si))
min_Si = np.zeros(len(Si))
window = 3

for i in range(window,len(chl)):
    if np.isfinite(chl[i])==True:
        max_Si[i] = np.nanmax(Si[i-window:i])
        min_Si[i] = np.nanmin(Si[i-window:i])

min_Si13 = min_Si[dates13_ind]
min_Si17 = min_Si[dates17_ind]

if 0:
	fig, ax = plt.subplots(nrows=3, sharex=True)
	ax[0].hist(np.log10(min_Si[np.where(min_Si!=0)]), bins=100, label='Observed $Ri_x$ 1990-2019')
	ax[0].legend(loc='best')
	ax[1].hist(np.log10(min_rix13[np.where(min_rix13!=0)]), bins=100, label='Modeled $Ri_x$ WY 2013 (Dry Year)')
	ax[1].legend(loc='best')
	ax[2].hist(np.log10(min_rix17[np.where(min_rix17!=0)]), bins=100, label='Modeled $Ri_x$ WY 2017 (Wet Year)')
	ax[2].legend(loc='best')
	ax[2].set_xlabel('log($Ri_x$)')


## convert modeled rix to quantiles 
q = np.arange(0,1,0.001)
rix_val = np.asarray([np.quantile(min_Si[np.isfinite(chl)],q[i]) for i in range(len(q))])

valid_min_Si = min_Si[np.isfinite(chl)]
qobs = np.asarray([q[np.argmin(np.abs(rix_val-valid_min_Si[i]))] for i in range(len(valid_min_Si))])
q13 = np.asarray([q[np.argmin(np.abs(rix_val-min_rix13[i]))] for i in range(len(min_rix13))])
q17 = np.asarray([q[np.argmin(np.abs(rix_val-min_rix17[i]))] for i in range(len(min_rix17))])

prob13_chl50 = np.asarray([prob(q13[i], .5)[0] for i in range(len(q13))])
prob13_chl75 = np.asarray([prob(q13[i], .75)[0] for i in range(len(q13))])
prob13_chl90 = np.asarray([prob(q13[i], .90)[0] for i in range(len(q13))])


prob17_chl50 = np.asarray([prob(q17[i], .5)[0] for i in range(len(q17))])
prob17_chl75 = np.asarray([prob(q17[i], .75)[0] for i in range(len(q17))])
prob17_chl90 = np.asarray([prob(q17[i], .90)[0] for i in range(len(q17))])

if 0:
	fig, ax = plt.subplots(nrows=2, sharey=True, figsize=(7,5))
	ax[0].plot(time13, prob13_chl75, color='teal')
	ax[0].set_xlim([date(2012, 10, 1), date(2013, 10, 1)])
	ax[0].set_ylabel('Probability (Chl > 75%tile)')
	ax[0].set_title('Water Year 2013 (Dry)')
	ax0 = ax[0].twinx()
	ax0.set_ylim((0,18))
	ax0.set_ylabel('Chl [mg/$m^3$]')
	ax0.plot(dates[dates13_ind], chl[dates13_ind], 'o', color='grey', alpha=0.8)
	ax0.plot(dates[dates13_ind], np.ones(len(dates13_ind))*np.quantile(chl[np.isfinite(chl)],0.75), '--', color='grey', alpha=0.6)
	ax[1].plot(time17, prob17_chl75, color='teal')
	ax[1].set_ylabel('Probability (Chl > 75%tile)')
	ax[1].set_xlim([date(2016, 10, 1), date(2017, 10, 1)])
	ax[1].set_title('Water Year 2017 (Wet)')
	ax1 = ax[1].twinx()
	ax1.plot(dates[dates17_ind], chl[dates17_ind], 'o', color='grey', alpha=0.8)
	ax1.plot(dates[dates17_ind], np.ones(len(dates17_ind))*np.quantile(chl[np.isfinite(chl)],0.75), '--', color='grey', alpha=0.6)
	ax1.set_ylabel('Chl [mg/$m^3$]')
	ax1.set_ylim((0,18))
	fig.tight_layout()


	c1 = 'indianred'
	c2 = 'lightseagreen'
	fig, ax = plt.subplots(nrows=3, sharex=True)
	ax[0].plot(time13, prob13_chl50, label='WY 2013 (Dry Year)', color=c1)
	ax[0].plot(time13, prob17_chl50, label='WY 2017 (Wet Year)', color=c2)
	ax[0].set_xlim([date(2012, 10, 1), date(2013, 10, 1)])	
	ax[0].legend(loc='lower right')
	ax[0].set_title('Probability of Observing >0.5 Quantile Level Chlorophyll')
	ax[1].plot(time13, prob13_chl75, color=c1)
	ax[1].plot(time13, prob17_chl75, color=c2)
	ax[1].set_xlim([date(2012, 10, 1), date(2013, 10, 1)])
	ax[1].set_ylabel('Probability')
	ax[1].set_title('Probability of Observing >0.75 Quantile Level Chlorophyll')
	ax[2].plot(time13, prob13_chl90, color=c1)
	ax[2].plot(time13, prob17_chl90, color=c2)
	ax[2].set_xlim([date(2012, 10, 1), date(2013, 10, 1)])
	ax[2].set_title('Probability of Observing >0.9 Quantile Level Chlorophyll')
	date_form = DateFormatter("%b")
	ax[2].xaxis.set_major_formatter(date_form)




