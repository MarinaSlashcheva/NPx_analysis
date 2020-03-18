# -*- coding: utf-8 -*-
"""
Tryout of the NPx recording analysis

"""

# %% Load packages etc.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import os.path
import h5py 
import scipy.io

from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import TimeSeries, NWBHDF5IO, NWBFile, get_manager 
from pynwb.file import Subject
import deepdish as dd


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

excel_info = pd.read_excel((RawDataDir + '\\Recordings_Marina_NPx.xlsx'), sheet_name=Sess)

# Select spikes from good clusters only
# Have to add the depth of the clusters
good_clus = cluster_group[cluster_group['group'] == 'good']
good_clus_info = cluster_info[cluster_group['group'] == 'good']
print("Found", len(good_clus), ' of good clusters') # has depth info

good_spikes_ind = [x in good_clus['cluster_id'].values for x in spike_clusters]
spike_clus_good = spike_clusters[good_spikes_ind]
spike_times_good = spike_times[good_spikes_ind]
spike_stamps_good = spike_stamps[good_spikes_ind]

good_clus_info['area'] = good_clus_info['depth'] > np.max(good_clus_info['depth']) - 900
good_clus_info['area'] = good_clus_info['area'].replace(True, 'V1')
good_clus_info['area'] = good_clus_info['area'].replace(False, 'HPC')

del spike_clusters, spike_times, spike_stamps, good_spikes_ind
# %%
# Now reading digitals from condInfo
# This has to be checked carefully again, especially for few stimuli in the session and 

# cond class contains the following:
#   'spontaneous_brightness': dict_keys(['name', 'time', 'timestamps', 'trl_list', 'conf'])
#   'natural_images': dict_keys(['name', 'time', 'timestamps', 'trl_list', 'conf', 'img_order', 'img_name'])

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
        cond[stim].img_name = cond[stim].conf[0][0][0][10][0] # currently not used but might be needed later

# This is how class can be turned into dict
# vars(cond[0])
a = cond[1].__dict__ 
a.keys()


# %% Trying to create NWB files

# Need to upload excel table and extract all relevant info from there

start_time = datetime(2020, 2, 27, 14, 36, 7, tzinfo=tzlocal())
nwb_subject = Subject(description = "Pretty nice girl",
                      sex = 'F', species = 'mouse',
                      subject_id=excel_info['Mouse'].values[0],
                      genotype = excel_info['Genotype'].values[0]
                      )

nwbfile = NWBFile(session_description= "NPx recording of Natural images and spontaneous activity", 
                  session_id = Sess,
                  identifier='NWB123', 
                  session_start_time=start_time, 
                  experimenter = 'Marina Slashcheva', 
                  institution = 'ESI, Frankfurt',
                  lab = 'Martin Vinck',
                  notes = ' | '.join([x for x in list(excel_info['Note'].values) if str(x) != 'nan']),
                  protocol = ' | '.join([x for x in list(excel_info['experiment'].values) if str(x) != 'nan']),
                  data_collection = 'Ref: {}, Probe_angle: {}, , Depth: {}, APcoord: {}, MLcoord: {}, Recday: {}, Hemi: {}'.format(excel_info['refCh'].values[0], 
                                          excel_info['Angle_probe'].values[0], excel_info['Depth'].values[0], 
                                          excel_info['anteroposterior'].values[0], excel_info['mediolateral'].values[0],
                                          excel_info['Recday'].values[0], excel_info['Hemisphere'].values[0]),
                  subject = nwb_subject
                  )

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
# Images names can be also added here
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

#def ms_raster():
    

def make_trials(dat): # takes NWB file with trials and units
    #trl_list = len(dat.trials)*[None]
    #trl_list = [] 
    trl_list = {}
    for tr in range(10): #(len(dat.trials)):
        tmp_dict = {}
        for un in range(len(dat.units)):
            
            tmp_ind = (dat.units[un]['spike_times'].values[0] >= dat.trials[tr]['start_time'].values[0]) & (dat.units[un]['spike_times'].values[0] < dat.trials[tr]['stop_time'].values[0])
            tmp_dict[str(dat.units[un].index.values[0])] = dat.units[un]['spike_times'].values[0][tmp_ind]
       
        #trl_list[tr] = tmp_dict
        #trl_list.append(tmp_dict)
        trl_list[str(tr)] = tmp_dict
        print('Trial ', tr)
    return trl_list


# %%
# Reading the NWB data
os.chdir(RawDataDir)
f = NWBHDF5IO((Sess + '.nwb'), 'r')
data_nwb = f.read()


trials_short = make_trials(data_nwb)

trials = make_trials(data_nwb)


with h5py.File("mytestfile.hdf5", "w") as file:
    dset = file.create_dataset("mydataset", (100,), dtype='i')
    
z = h5py.File('mydataset.hdf5', 'a')
grp = z.create_group("subgroup")



dd.io.save('trl.h5', trials_short)

filename = 'trl.h5'
h5 = h5py.File(filename, 'r')
h5['0']['6'][:]

for it in h5['1'].items():
    print(it[:])

tr = 4
list_units = []
list_spikes = []
list_depth = []

for key in h5[str(tr)].keys():
    list_spikes.append(h5[str(tr)][key][:])
    list_units.append(int(key))

list_spikes_ord =  pd.Series(data=list_spikes,index=list_units).sort_index().tolist()
#(np.array(list_units)).argsort()
    
fig = plt.figure()  # an empty figure with no axes
fig, (ax0) = plt.subplots(1,1, figsize=(16,6))
ax0.eventplot(list_spikes_ord)
ax0.set_xlim(data_nwb.trials[tr]['start_time'].values[0], data_nwb.trials[tr]['stop_time'].values[0])

    



            
f.close()

# %% Option for multiprocessing
from multiprocessing import Pool
import workers

def worker(tr):
    tmp_dict = {}
    for un in range(len(data_nwb.units)):
            
        tmp_ind = (data_nwb.units[un]['spike_times'].values[0] >= data_nwb.trials[tr]['start_time'].values[0]) & (data_nwb.units[un]['spike_times'].values[0] < data_nwb.trials[tr]['stop_time'].values[0])
        tmp_dict[str(data_nwb.units[un].index.values[0])] = data_nwb.units[un]['spike_times'].values[0][tmp_ind]
    return tmp_dict   
    print('Trial ', tr)


if __name__ == '__main__':
    with Pool(os.cpu_count()) as p:
        print(p.map(workers.worker, [0,1,2,3,4,5]))
        





