# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:49:15 2020

@author: slashchevam
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path
from scipy import signal as sg

import h5py 


# %%
Sess = 'Bl6_177_2020-02-29_17-12-05'
Sess = 'Bl6_177_2020-02-27_14-36-07'
Sess = 'Bl6_177_2020-03-01_14-49-02'

wd = os.path.join('C:\\Users\\slashchevam\\Desktop\\NPx\\digitals_example', Sess)

#wd = '//gs/projects/OWVinckNatIm/NPx_recordings/Bl6_177_2020-02-27_14-36-07/experiment1/recording1/events/Rhythm_FPGA-102.0/TTL_1'
#wd_cont = '//gs/projects/OWVinckNatIm/NPx_recordings/Bl6_177_2020-02-27_14-36-07/experiment1/recording1/continuous/Rhythm_FPGA-102.0'

#cont_ts = np.load(os.path.join(wd_cont, "timestamps.npy"))
#ts_first = cont_ts[0]
#del cont_ts

ts_first = 6354944 # for Bl6_177_2020-02-27_14-36-07
sr = 30000

#channels = np.load(os.path.join(wd, "channels.npy"))
channel_states = np.load(os.path.join(wd, "channel_states.npy"))
ts = np.load(os.path.join(wd, "timestamps.npy"))
#words = np.load(os.path.join(wd, "full_words.npy"))
ts = ts - ts_first

camera_ind = [channel_states == 1][0]
camera_time = ts[camera_ind]/sr


diff = camera_time[1:] - camera_time[:-1]
val = diff[diff > 0.05]

np.where(diff == val[0])[0][0]
np.where(diff == val[1])

camera_time_cont = camera_time[np.where(diff == val[0])[0][0]:np.where(diff == val[1])[0][0]]

pupil_export = {'time': camera_time_cont[0:len(pupil_area)], 'pupil_area': pupil_area}
pupil_area_df = pd.DataFrame(pupil_export, columns=['time', 'pupil_area'])

name = Sess + '_pupil.csv'
pupil_area_df.to_csv(name, index=False)
# %%



# %%

pupil_folder = 'C:\\Users\\slashchevam\\Desktop\\NPx\\videos'
os.chdir(pupil_folder)

pupil_table = pd.read_csv(os.path.join(pupil_folder, "Bl6_177_200227_0001DLC_resnet50_Pupils_NPxApr22shuffle1_300000.csv"),  
                          sep=",", 
                          index_col=False)

pupil_table = pupil_table[2:]
pupil_table.columns = ['ind', 
                       'upx', 'upy', 'upscore', 
                       'rightx', 'righty', 'rightscore', 
                       'downx', 'downy', 'downscore', 
                       'leftx', 'lefty', 'leftscore']

pupil_table = pupil_table.astype('float')

# Adjust manually for your data
kernel = 15 # must be odd

w = np.abs(pupil_table['rightx'] - pupil_table['leftx'])
plt.plot(w)
w_medianf = sg.medfilt(w, kernel_size = kernel)
plt.plot(w_medianf)

h = np.abs(pupil_table['upy'] - pupil_table['downy'])
plt.plot(h)
h_medianf = sg.medfilt(h, kernel_size = kernel)
plt.plot(h_medianf)

av_rad = (w_medianf + h_medianf)/4
pupil_area = np.pi * av_rad**2

plt.plot(pupil_area)



