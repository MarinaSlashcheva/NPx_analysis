# -*- coding: utf-8 -*-
"""
Tryout of the NPx recording analysis

"""

# %% Load packages etc.
import numpy as np
import pandas as pd
import os
import os.path
import h5py 
import scipy.io

from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import NWBFile
from pynwb import TimeSeries
from pynwb import NWBHDF5IO
from pynwb import get_manager


# %% Define folders and other common parameters
# upload information about all recorded session

Sess = 'Bl6_177_2020-02-27_14-36-07'
sr = 30000
system = 'windows' # or 'linux'

if system == 'linux':
    SaveDir = '/mnt/gs/departmentN4/Marina/NPx_python/'
    RawDataDir = '/mnt/gs/projects/OWVinckNatIm/NPx_recordings/'
    PAthToAnalyzed = '/experiment1/recording1/continuous/Neuropix-PXI-100.0/'
    MatlabOutput = '/mnt/gs/projects/OWVinckNatIm/NPx_processed/Lev0_condInfo/'

    PathToUpload = RawDataDir + Sess + PAthToAnalyzed

if system == 'windows':
    SaveDir = r'C:\Users\slashchevam\Desktop\NPx_Bl6_177_2020-02-27_14-36-07\Results'
    RawDataDir = r'C:\Users\slashchevam\Desktop\NPx_Bl6_177_2020-02-27_14-36-07\Bl6_177_2020-02-27_14-36-07'
    MatlabOutput = RawDataDir

    PathToUpload = RawDataDir
# %% Upload all the necessary data

spike_stamps = np.load(os.path.join(PathToUpload, "spike_times.npy"))
spike_times = spike_stamps / sr
spike_clusters = np.load(os.path.join(PathToUpload, "spike_clusters.npy"))
cluster_group = pd.read_csv(os.path.join(PathToUpload, "cluster_group.tsv"),  sep="\t")
cluster_info = pd.read_csv(os.path.join(PathToUpload, "cluster_info.tsv"),  sep="\t")

# Select spikes from good clusters only
# Have to add the depth of the clusters
good_clus = cluster_group[cluster_group['group'] == 'good']
good_clus_info = cluster_info[cluster_group['group'] == 'good']
print("Found", len(good_clus), ' of good clusters') # has depth info

good_spikes_ind = [x in good_clus['cluster_id'].values for x in spike_clusters]
spike_clus_good = spike_clusters[good_spikes_ind]
spike_times_good = spike_times[good_spikes_ind]
#spike_stamps_good = spike_stamps[good_spikes_ind]

good_clus_info['area'] = good_clus_info['depth'] > np.max(good_clus_info['depth']) - 900
good_clus_info['area'] = good_clus_info['area'].replace(True, 'V1')
good_clus_info['area'] = good_clus_info['area'].replace(False, 'HPC')

del spike_clusters, spike_times, spike_stamps, good_spikes_ind
# %%
# Now reading digitals from condInfo
# This has to be checked carefully again, especially for few stimuli in the session and blocks

class condInfo:
    pass

if system == 'linux':
    mat = scipy.io.loadmat(os.path.join((MatlabOutput + Sess), 'condInfo_01.mat'))
else:
    mat = scipy.io.loadmat(os.path.join(PathToUpload, 'condInfo_01.mat'))

SC_stim_labels = mat['StimClass'][0][0][0][0]
SC_stim_present = np.where(mat['StimClass'][0][0][1][0] == 1)[0]
SC_stim_labels_present = SC_stim_labels[SC_stim_present]

cond = [condInfo() for i in range(len(SC_stim_labels_present))]

for stim in range(len(SC_stim_labels_present)):
    cond[stim].name = SC_stim_labels_present[stim][0]
    cond[stim].time = mat['StimClass'][0][0][3][0, SC_stim_present[stim]][0][0][0][2]
    cond[stim].timestamps =  mat['StimClass'][0][0][3][0, SC_stim_present[stim]][0][0][0][3]
    cond[stim].trl_list = mat['StimClass'][0][0][3][0, SC_stim_present[stim]][1]
    
    cond[stim].conf =  mat['StimClass'][0][0][2][0, SC_stim_present[stim]]
    
    if SC_stim_labels_present[stim][0] == 'natural_images':
        img_order = []
        for i in range(len(cond[stim].conf[0][0][0][8][0])):
            img_order.append(cond[stim].conf[0][0][0][8][0][i][1][0][0])
        cond[stim].img_order = img_order

# Keep it for later just in case
#SC_trl_time = mat['StimClass'][0][0][3][0,5][0][0][0][2]
#SC_trl_timestamps = mat['StimClass'][0][0][3][0,5][0][0][0][3]
#SC_trl_list = mat['StimClass'][0][0][3][0,5][1]
#SC_stim_cfg =  mat['StimClass'][0][0][2][0, 9]

# This is how class can be turned into dict
# vars(cond[0])
a = cond[1].__dict__ 
a.keys()

# %% Trying to create NWB files

# Need to upload excel table and extract all relevant info from there

start_time = datetime(2020, 2, 27, 14, 36, 7, tzinfo=tzlocal())
nwbfile = NWBFile(session_description=Sess, identifier='NWB123', session_start_time=start_time, 
                  experimenter = 'Marina Slashcheva', lab = 'Martin Vinck, ESI')

# Did not add it for the moment
# test_ts = TimeSeries(name='test_timeseries', data=data, unit='m', timestamps=timestamps)

# Add units
nwbfile.add_unit_column('location', 'the anatomical location of this unit') # to be added and CHECKED
nwbfile.add_unit_column('depth', 'depth on the NPx probe')
nwbfile.add_unit_column('channel', 'channel on the NPx probe')
nwbfile.add_unit_column('fr', 'average FR according to KS')

for un in good_clus_info['id']:
    info_tmp = good_clus_info[good_clus_info['id'] == un]
    spike_times_tmp = spike_times_good[spike_clus_good == un]
    
    nwbfile.add_unit(id = un, spike_times = np.transpose(spike_times_tmp)[0], 
                     location = info_tmp['area'].values[0], depth = info_tmp['depth'].values[0], 
                     channel = info_tmp['ch'].values[0], fr = info_tmp['fr'].values[0])
    del spike_times_tmp

# Add epochs 
for ep in range(len(cond)):
    if cond[ep].name == 'spontaneous_brightness':
        nwbfile.add_epoch(cond[ep].time[0][0], cond[ep].time[0][1], cond[ep].name)
    if cond[ep].name == 'natural_images':
        nwbfile.add_epoch(cond[ep].time[0][0], cond[ep].time[-1][1], cond[ep].name)

# Add trials
nwbfile.add_trial_column(name='stimset', description='the visual stimulus type during the trial')
nwbfile.add_trial_column(name='img_id', description='image ID for Natural Images')

for ep in range(len(cond)):
    if cond[ep].name == 'spontaneous_brightness':
        nwbfile.add_trial(start_time = cond[ep].time[0][0], stop_time = cond[ep].time[0][1], 
                          stimset = (cond[ep].name).encode('utf8'), img_id = ('gray').encode('utf8'))
        
    if cond[ep].name == 'natural_images':
        for tr in range(len(cond[ep].time)):
            nwbfile.add_trial(start_time = cond[ep].time[tr][0], stop_time = cond[ep].time[tr][1], 
                              stimset = (cond[ep].name).encode('utf8'), img_id = (str(cond[ep].img_order[tr])).encode('utf8'))


# Write NWB file
os.chdir(RawDataDir)
name_to_save = Sess + '.nwb'
io = NWBHDF5IO(name_to_save, manager=get_manager(), mode='w')
io.write(nwbfile)
io.close()

del nwbfile


# %%
# Reading the NWB data
f = NWBHDF5IO(name_to_save, 'r')
data_nwb = f.read()




f.close()




















