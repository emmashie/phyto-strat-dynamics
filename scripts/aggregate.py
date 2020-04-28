import numpy as np 
import matplotlib.pyplot as plt 
import os 
import pandas as pd 
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from datetime import datetime 

google_drive_path = "/mnt/c/Google_Drive"
usgs_path = "1_Nutrient_Share/1_Projects_NUTRIENTS/06_FY20_NMS_Projects/FY20_PhyotoBlooms_Controls/3_Project_Work_Analysis/aggregating_data"
flow_path = "1_Nutrient_Share/2_Data_NUTRIENTS/Aggregated_South_Bay_Data/Discharge_Data"
tidal_path = "1_Nutrient_Share/2_Data_NUTRIENTS/Aggregated_South_Bay_Data/Tidal"
salt_path = "1_Nutrient_Share/1_Projects_NUTRIENTS/06_FY20_NMS_Projects/FY20_PhyotoBlooms_Controls/3_Project_Work_Analysis/exploring_S_daily"
fleb_path = "1_Nutrient_Share/2_Data_NUTRIENTS/Aggregated_South_Bay_Data"

tide_file = 'smb_u_extended.csv'
alameda_file = 'Alameda_Creek_Discharge.csv'
delta_file = 'Delta_Dayflow.csv'
usgs_file = 'usgs-sfb_data-processed.csv'
salt_file = 'sal_dSdx_daily.csv'
fleb_file = 'South_Bay_Data.csv'

savepath = "1_Nutrient_Share/1_Projects_NUTRIENTS/06_FY20_NMS_Projects/FY20_PhyotoBlooms_Controls/3_Project_Work_Analysis/aggregating_data"


alameda = pd.read_csv(os.path.join(google_drive_path,flow_path,alameda_file), comment = '#', header = 0)
alameda = alameda[['Date','Discharge_cfs']]
alameda['Date'] = pd.to_datetime(alameda['Date'])

delta = pd.read_csv(os.path.join(google_drive_path,flow_path,delta_file))
delta['day'] = pd.to_datetime(delta['day'])
delta.columns = ['Date', 'Delta_Dayflow']


tides = pd.read_csv(os.path.join(google_drive_path,tidal_path,tide_file))
tides.columns = ['time', 'tidal_velocity']
tides['time'] = pd.to_datetime(tides['time'])
tides['tidal_velocity'] = tides['tidal_velocity'] 
tides = tides.resample('D', on='time').agg('max') ## maximum flood velocity
tides['time'] = pd.to_datetime(tides.index)
tides.columns = ['Date', 'tidal_velocity']

salt = pd.read_csv(os.path.join(google_drive_path,salt_path,salt_file))
datestr = ['%s %s' % (str(salt.year[i]), str(salt.doy[i])) for i in range(len(salt))]
salt['dec_date'] = datestr
salt = salt[['dec_date', 'sal22', 'sal24', 'sal25', 'sal27', 'sal29', 'sal30', 'sal32', 
             'dsdx_22_24', 'dsdx_24_25', 'dsdx_25_27', 'dsdx_27_29', 'dsdx_29_30', 
             'dsdx_30_32', 'dsdx_27_32']]
salt.columns = ['Date', 'sal22', 'sal24', 'sal25', 'sal27', 'sal29', 'sal30', 'sal32', 
                'dsdx_22_24', 'dsdx_24_25', 'dsdx_25_27', 'dsdx_27_29', 'dsdx_29_30', 
                'dsdx_30_32', 'dsdx_27_32']
salt['Date'] = pd.to_datetime(salt['Date'], format='%Y %j')

usgs = pd.read_csv(os.path.join(google_drive_path,usgs_path,usgs_file))
usgs = usgs[['Date', 'StationNumber', 'Depth', 'Distance', 'dchl', 'cchl']]
usgs.columns = ['time', 'StationNumber', 'Depth', 'Distance', 'dchl', 'cchl']

dep = 4.0
sta22 = usgs.copy().iloc[np.where(usgs.StationNumber==22)[0]].reset_index()
sta22 = sta22.copy().iloc[np.where(sta22.Depth<dep)[0]].reset_index()
sta22['chl_comb'] = [0]*len(sta22)
sta22['chl_comb'] = [x if ~np.isnan(x) else y for x,y in zip(sta22['dchl'],sta22['cchl'])]
sta22['time'] = pd.to_datetime(sta22['time'])-np.timedelta64(8,'h')
sta22 = sta22.resample('D', on='time').mean()
sta22['time'] = pd.to_datetime(sta22.index)
sta22['time'] = sta22['time'].dt.date
sta22['time'] = pd.to_datetime(sta22['time'])
sta22 = sta22[['time', 'chl_comb']]
sta22.columns = ['Date', 'sta22_chl']

sta24 = usgs.copy().iloc[np.where(usgs.StationNumber==24)[0]].reset_index()
sta24 = sta24.copy().iloc[np.where(sta24.Depth<dep)[0]].reset_index()
sta24['chl_comb'] = [0]*len(sta24)
sta24['chl_comb'] = [x if ~np.isnan(x) else y for x,y in zip(sta24['dchl'],sta24['cchl'])]
sta24['time'] = pd.to_datetime(sta24['time'])-np.timedelta64(8,'h')
sta24 = sta24.resample('D', on='time').mean()
sta24['time'] = pd.to_datetime(sta24.index)
sta24['time'] = sta24['time'].dt.date
sta24['time'] = pd.to_datetime(sta24['time'])
sta24 = sta24[['time', 'chl_comb']]
sta24.columns = ['Date', 'sta24_chl']

sta25 = usgs.copy().iloc[np.where(usgs.StationNumber==25)[0]].reset_index()
sta25 = sta25.copy().iloc[np.where(sta25.Depth<dep)[0]].reset_index()
sta25['chl_comb'] = [0]*len(sta25)
sta25['chl_comb'] = [x if ~np.isnan(x) else y for x,y in zip(sta25['dchl'],sta25['cchl'])]
sta25['time'] = pd.to_datetime(sta25['time'])-np.timedelta64(8,'h')
sta25 = sta25.resample('D', on='time').mean()
sta25['time'] = pd.to_datetime(sta25.index)
sta25['time'] = sta25['time'].dt.date
sta25['time'] = pd.to_datetime(sta25['time'])
sta25 = sta25[['time', 'chl_comb']]
sta25.columns = ['Date', 'sta25_chl']

sta27 = usgs.copy().iloc[np.where(usgs.StationNumber==27)[0]].reset_index()
sta27 = sta27.copy().iloc[np.where(sta27.Depth<dep)[0]].reset_index()
sta27['chl_comb'] = [0]*len(sta27)
sta27['chl_comb'] = [x if ~np.isnan(x) else y for x,y in zip(sta27['dchl'],sta27['cchl'])]
sta27['time'] = pd.to_datetime(sta27['time'])-np.timedelta64(8,'h')
sta27 = sta27.resample('D', on='time').mean()
sta27['time'] = pd.to_datetime(sta27.index)
sta27['time'] = sta27['time'].dt.date
sta27['time'] = pd.to_datetime(sta27['time'])
sta27 = sta27[['time', 'chl_comb']]
sta27.columns = ['Date', 'sta27_chl']

sta29 = usgs.copy().iloc[np.where(usgs.StationNumber==29)[0]].reset_index()
sta29 = sta29.copy().iloc[np.where(sta29.Depth<dep)[0]].reset_index()
sta29['chl_comb'] = [0]*len(sta29)
sta29['chl_comb'] = [x if ~np.isnan(x) else y for x,y in zip(sta29['dchl'],sta29['cchl'])]
sta29['time'] = pd.to_datetime(sta29['time'])-np.timedelta64(8,'h')
sta29 = sta29.resample('D', on='time').mean()
sta29['time'] = pd.to_datetime(sta29.index)
sta29['time'] = sta29['time'].dt.date
sta29['time'] = pd.to_datetime(sta29['time'])
sta29 = sta29[['time', 'chl_comb']]
sta29.columns = ['Date', 'sta29_chl']

sta30 = usgs.copy().iloc[np.where(usgs.StationNumber==30)[0]].reset_index()
sta30 = sta30.copy().iloc[np.where(sta30.Depth<dep)[0]].reset_index()
sta30['chl_comb'] = [0]*len(sta30)
sta30['chl_comb'] = [x if ~np.isnan(x) else y for x,y in zip(sta30['dchl'],sta30['cchl'])]
sta30['time'] = pd.to_datetime(sta30['time'])-np.timedelta64(8,'h')
sta30 = sta30.resample('D', on='time').mean()
sta30['time'] = pd.to_datetime(sta30.index)
sta30['time'] = sta30['time'].dt.date
sta30['time'] = pd.to_datetime(sta30['time'])
sta30 = sta30[['time', 'chl_comb']]
sta30.columns = ['Date', 'sta30_chl']

sta32 = usgs.copy().iloc[np.where(usgs.StationNumber==32)[0]].reset_index()
sta32 = sta32.copy().iloc[np.where(sta32.Depth<dep)[0]].reset_index()
sta32['chl_comb'] = [0]*len(sta32)
sta32['chl_comb'] = [x if ~np.isnan(x) else y for x,y in zip(sta32['dchl'],sta32['cchl'])]
sta32['time'] = pd.to_datetime(sta32['time'])-np.timedelta64(8,'h')
sta32 = sta32.resample('D', on='time').mean()
sta32['time'] = pd.to_datetime(sta32.index)
sta32['time'] = sta32['time'].dt.date
sta32['time'] = pd.to_datetime(sta32['time'])
sta32 = sta32[['time', 'chl_comb']]
sta32.columns = ['Date', 'sta32_chl']

usgs = pd.read_csv(os.path.join(google_drive_path,usgs_path,usgs_file))
usgs = usgs[['Date', 'StationNumber', 'Depth', 'sal']]
usgs.columns = ['time', 'StationNumber', 'Depth', 'salt']

top = 0.5 
bot = 11

s22 = usgs.copy().iloc[np.where(usgs.StationNumber==22)[0]].reset_index()
s22['time'] = pd.to_datetime(s22['time'])-np.timedelta64(8,'h')
s22['time'] = s22['time'].dt.date
udates = np.unique(s22.time)
top_z = np.asarray([s22.Depth[np.where(s22.time==d)[0]][np.argmin(np.abs(s22.Depth[np.where(s22.time==d)[0]]-top))] for d in udates])
bot_z = np.asarray([s22.Depth[np.where(s22.time==d)[0]][np.argmin(np.abs(s22.Depth[np.where(s22.time==d)[0]]-bot))] for d in udates])
top_s = np.asarray([s22['salt'][np.where(s22.time==d)[0]][np.argmin(np.abs(s22.Depth[np.where(s22.time==d)[0]]-top))] for d in udates])
bot_s = np.asarray([s22['salt'][np.where(s22.time==d)[0]][np.argmin(np.abs(s22.Depth[np.where(s22.time==d)[0]]-bot))] for d in udates])
dsdz = (top_s-bot_s)/np.abs(top_z-bot_z)
dsdz_22 = pd.DataFrame({'Date':udates,'s22_dsdz':dsdz})
dsdz_22['Date'] = pd.to_datetime(dsdz_22['Date'])

s24 = usgs.copy().iloc[np.where(usgs.StationNumber==24)[0]].reset_index()
s24['time'] = pd.to_datetime(s24['time'])-np.timedelta64(8,'h')
s24['time'] = s24['time'].dt.date
udates = np.unique(s24.time)
top_z = np.asarray([s24.Depth[np.where(s24.time==d)[0]][np.argmin(np.abs(s24.Depth[np.where(s24.time==d)[0]]-top))] for d in udates])
bot_z = np.asarray([s24.Depth[np.where(s24.time==d)[0]][np.argmin(np.abs(s24.Depth[np.where(s24.time==d)[0]]-bot))] for d in udates])
top_s = np.asarray([s24['salt'][np.where(s24.time==d)[0]][np.argmin(np.abs(s24.Depth[np.where(s24.time==d)[0]]-top))] for d in udates])
bot_s = np.asarray([s24['salt'][np.where(s24.time==d)[0]][np.argmin(np.abs(s24.Depth[np.where(s24.time==d)[0]]-bot))] for d in udates])
dsdz = (top_s-bot_s)/np.abs(top_z-bot_z)
dsdz_24 = pd.DataFrame({'Date':udates,'s24_dsdz':dsdz})
dsdz_24['Date'] = pd.to_datetime(dsdz_24['Date'])

s25 = usgs.copy().iloc[np.where(usgs.StationNumber==25)[0]].reset_index()
s25['time'] = pd.to_datetime(s25['time'])-np.timedelta64(8,'h')
s25['time'] = s25['time'].dt.date
udates = np.unique(s25.time)
top_z = np.asarray([s25.Depth[np.where(s25.time==d)[0]][np.argmin(np.abs(s25.Depth[np.where(s25.time==d)[0]]-top))] for d in udates])
bot_z = np.asarray([s25.Depth[np.where(s25.time==d)[0]][np.argmin(np.abs(s25.Depth[np.where(s25.time==d)[0]]-bot))] for d in udates])
top_s = np.asarray([s25['salt'][np.where(s25.time==d)[0]][np.argmin(np.abs(s25.Depth[np.where(s25.time==d)[0]]-top))] for d in udates])
bot_s = np.asarray([s25['salt'][np.where(s25.time==d)[0]][np.argmin(np.abs(s25.Depth[np.where(s25.time==d)[0]]-bot))] for d in udates])
dsdz = (top_s-bot_s)/np.abs(top_z-bot_z)
dsdz_25 = pd.DataFrame({'Date':udates,'s25_dsdz':dsdz})
dsdz_25['Date'] = pd.to_datetime(dsdz_25['Date'])

s27 = usgs.copy().iloc[np.where(usgs.StationNumber==27)[0]].reset_index()
s27['time'] = pd.to_datetime(s27['time'])-np.timedelta64(8,'h')
s27['time'] = s27['time'].dt.date
udates = np.unique(s27.time)
top_z = np.asarray([s27.Depth[np.where(s27.time==d)[0]][np.argmin(np.abs(s27.Depth[np.where(s27.time==d)[0]]-top))] for d in udates])
bot_z = np.asarray([s27.Depth[np.where(s27.time==d)[0]][np.argmin(np.abs(s27.Depth[np.where(s27.time==d)[0]]-bot))] for d in udates])
top_s = np.asarray([s27['salt'][np.where(s27.time==d)[0]][np.argmin(np.abs(s27.Depth[np.where(s27.time==d)[0]]-top))] for d in udates])
bot_s = np.asarray([s27['salt'][np.where(s27.time==d)[0]][np.argmin(np.abs(s27.Depth[np.where(s27.time==d)[0]]-bot))] for d in udates])
dsdz = (top_s-bot_s)/np.abs(top_z-bot_z)
dsdz_27 = pd.DataFrame({'Date':udates,'s27_dsdz':dsdz})
dsdz_27['Date'] = pd.to_datetime(dsdz_27['Date'])

s29 = usgs.copy().iloc[np.where(usgs.StationNumber==29)[0]].reset_index()
s29['time'] = pd.to_datetime(s29['time'])-np.timedelta64(8,'h')
s29['time'] = s29['time'].dt.date
udates = np.unique(s29.time)
top_z = np.asarray([s29.Depth[np.where(s29.time==d)[0]][np.argmin(np.abs(s29.Depth[np.where(s29.time==d)[0]]-top))] for d in udates])
bot_z = np.asarray([s29.Depth[np.where(s29.time==d)[0]][np.argmin(np.abs(s29.Depth[np.where(s29.time==d)[0]]-bot))] for d in udates])
top_s = np.asarray([s29['salt'][np.where(s29.time==d)[0]][np.argmin(np.abs(s29.Depth[np.where(s29.time==d)[0]]-top))] for d in udates])
bot_s = np.asarray([s29['salt'][np.where(s29.time==d)[0]][np.argmin(np.abs(s29.Depth[np.where(s29.time==d)[0]]-bot))] for d in udates])
dsdz = (top_s-bot_s)/np.abs(top_z-bot_z)
dsdz_29 = pd.DataFrame({'Date':udates,'s29_dsdz':dsdz})
dsdz_29['Date'] = pd.to_datetime(dsdz_29['Date'])

s30 = usgs.copy().iloc[np.where(usgs.StationNumber==30)[0]].reset_index()
s30['time'] = pd.to_datetime(s30['time'])-np.timedelta64(8,'h')
s30['time'] = s30['time'].dt.date
udates = np.unique(s30.time)
top_z = np.asarray([s30.Depth[np.where(s30.time==d)[0]][np.argmin(np.abs(s30.Depth[np.where(s30.time==d)[0]]-top))] for d in udates])
bot_z = np.asarray([s30.Depth[np.where(s30.time==d)[0]][np.argmin(np.abs(s30.Depth[np.where(s30.time==d)[0]]-bot))] for d in udates])
top_s = np.asarray([s30['salt'][np.where(s30.time==d)[0]][np.argmin(np.abs(s30.Depth[np.where(s30.time==d)[0]]-top))] for d in udates])
bot_s = np.asarray([s30['salt'][np.where(s30.time==d)[0]][np.argmin(np.abs(s30.Depth[np.where(s30.time==d)[0]]-bot))] for d in udates])
dsdz = (top_s-bot_s)/np.abs(top_z-bot_z)
dsdz_30 = pd.DataFrame({'Date':udates,'s30_dsdz':dsdz})
dsdz_30['Date'] = pd.to_datetime(dsdz_30['Date'])

s32 = usgs.copy().iloc[np.where(usgs.StationNumber==32)[0]].reset_index()
s32['time'] = pd.to_datetime(s32['time'])-np.timedelta64(8,'h')
s32['time'] = s32['time'].dt.date
udates = np.unique(s32.time)
top_z = np.asarray([s32.Depth[np.where(s32.time==d)[0]][np.argmin(np.abs(s32.Depth[np.where(s32.time==d)[0]]-top))] for d in udates])
bot_z = np.asarray([s32.Depth[np.where(s32.time==d)[0]][np.argmin(np.abs(s32.Depth[np.where(s32.time==d)[0]]-bot))] for d in udates])
top_s = np.asarray([s32['salt'][np.where(s32.time==d)[0]][np.argmin(np.abs(s32.Depth[np.where(s32.time==d)[0]]-top))] for d in udates])
bot_s = np.asarray([s32['salt'][np.where(s32.time==d)[0]][np.argmin(np.abs(s32.Depth[np.where(s32.time==d)[0]]-bot))] for d in udates])
dsdz = (top_s-bot_s)/np.abs(top_z-bot_z)
dsdz_32 = pd.DataFrame({'Date':udates,'s32_dsdz':dsdz})
dsdz_32['Date'] = pd.to_datetime(dsdz_32['Date'])

fleb_dat = pd.read_csv(os.path.join(google_drive_path,fleb_path,fleb_file))
fleb_dat = fleb_dat[['time','sta27_tidal_vel']]
fleb_dat['time'] = pd.to_datetime(fleb_dat['time'])
fleb_dat.columns = ['Date', 'cruise_tidal_velocity']

time = pd.DataFrame(pd.date_range(start = '1/1/1990', end = '1/1/2019')) 
time.columns = ['time'] 
time.index = time.time 
time.columns = ['Date']

df = pd.merge(time, alameda, how='left', on='Date')
df = pd.merge(df, delta, how='left', on='Date')
df = pd.merge(df, tides, how='left', on='Date')
df = pd.merge(df, salt, how='left', on='Date')
df = pd.merge(df, sta22, how='left', on='Date')
df = pd.merge(df, sta24, how='left', on='Date')
df = pd.merge(df, sta25, how='left', on='Date')
df = pd.merge(df, sta27, how='left', on='Date')
df = pd.merge(df, sta29, how='left', on='Date')
df = pd.merge(df, sta30, how='left', on='Date')
df = pd.merge(df, sta32, how='left', on='Date')
df = pd.merge(df, dsdz_22, how='left', on='Date')
df = pd.merge(df, dsdz_24, how='left', on='Date')
df = pd.merge(df, dsdz_25, how='left', on='Date')
df = pd.merge(df, dsdz_27, how='left', on='Date')
df = pd.merge(df, dsdz_29, how='left', on='Date')
df = pd.merge(df, dsdz_30, how='left', on='Date')
df = pd.merge(df, dsdz_32, how='left', on='Date')
df = pd.merge(df, fleb_dat, how='left', on='Date')


df.columns = ['Date', 'Alameda_cfs', 'Delta_Dayflow', 'tidal_velocity', 'sta22_salt',
              'sta24_salt', 'sta25_salt', 'sta27_salt', 'sta29_salt', 'sta30_salt', 'sta32_salt', 
              'dsdx_22_24','dsdx_24_25', 'dsdx_25_27', 'dsdx_27_29', 'dsdx_29_30', 'dsdx_30_32',
              'dsdx_27_32', 'sta22_chl', 'sta24_chl', 'sta25_chl', 'sta27_chl','sta29_chl', 
              'sta30_chl', 'sta32_chl', 'sta22_dsdz', 'sta24_dsdz','sta25_dsdz', 'sta27_dsdz', 
              'sta29_dsdz', 'sta30_dsdz', 'sta32_dsdz', 'cruise_tidal_velocity']

df.to_csv(os.path.join(google_drive_path, savepath, 'sb_data.csv'))