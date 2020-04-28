# -*- coding: utf-8 -*-

# %% Load packages etc.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path
import h5py 
import scipy.io
import seaborn as sb
import sys

from scipy import stats as st
from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import TimeSeries, NWBHDF5IO, NWBFile, get_manager 
from pynwb.file import Subject
from itertools import chain

from random import randint

sys.path.append(r'C:\Users\slashchevam\Desktop\NPx\NPx_analysis')
sys.path.append('/mnt/pns/departmentN4/Marina/NPx_python/NPx_analysis')
from NPx_preprocessing_module import *

# %%

# Choose the session
Sess = 'Bl6_177_2020-02-29_17-12-05'
Sess = 'Bl6_177_2020-02-27_14-36-07'
Sess = 'Bl6_177_2020-03-01_14-49-02'


pupil_folder = 'C:\\Users\\slashchevam\\Desktop\\NPx\\videos'
os.chdir(pupil_folder)

pupil_name = Sess + '_pupil.csv'
pupil_table = pd.read_csv(os.path.join(pupil_folder, pupil_name), index_col=False)


if sys.platform == 'win32':
    SaveDir = os.path.join(r'C:\Users\slashchevam\Desktop\NPx\Results', Sess)

if sys.platform == 'linux':
    SaveDir = os.path.join('/mnt/gs/departmentN4/Marina/NPx_python/', Sess)
    
if not os.path.exists(SaveDir):
    os.makedirs(SaveDir)
    
os.chdir(SaveDir)

# Create new NWB and HDF5 files, if they do not exist yet
# Creating hdf5 takes some time!!! 

# start_time = datetime(2020, 2, 27, 14, 36, 7, tzinfo=tzlocal())
# create_nwb_file(Sess, start_time)
# create_hdf5_file(Sess)

# Upload NWB and HDF5 files
f = NWBHDF5IO((Sess + '.nwb'), 'r')
data_nwb = f.read()

data_hdf = h5py.File((Sess + '_trials.hdf5'), 'r')


# Close files
#data_hdf.close()
#f.close()

# %%
# Add proper path for that! 
psth_per_unit_NatIm(Sess, 50) # session title and bin number

raster_spontaneous(Sess, 10, pupil_table) # session title and duration of one plot in sec


# %% Thie is the code for plotting rasters per unit, only 917 trials per unit overlayed with phst of the same data [1]

iti = []
trials = data_nwb.trials[:][data_nwb.trials[:]['stimset'] == 'natural_images'.encode('utf8')]
for t in range(1,917):
    iti.append((trials.iloc[t]['start_time'] - trials.iloc[t-1]['stop_time']) + 0.200)

iti_ordered = pd.Series(data=iti,index= range(2,len(iti)+2)).sort_values()

for un in data_hdf.keys():

    tr_list_raster = []
    for i in range(len(iti_ordered)):
        tr_tmp = iti_ordered.index[i]
        stop = trials.loc[tr_tmp]['stop_time']
        start = stop-0.800
        stop_previous = start-iti_ordered[tr_tmp]
        timevec = data_hdf[un]['spike_times'][(data_hdf[un]['spike_times'] >= stop_previous) & (data_hdf[un]['spike_times'] < stop)]
        tr_list_raster.append(timevec - start)
    
    
    fig = plt.figure()  # an empty figure with no axes
    fig, ax = plt.subplots(1,1, figsize=(18,8)) 
    ax.eventplot(tr_list_raster)
    ax.axvline(linewidth=1, color='r')
    ax.set_xlim(-0.500, 0.800)
    ax.set_title('Raster plot')
    ax.set_ylabel('trial, sorted by ITI')
    ax.set_xlabel('Time, sec')
    color = 'tab:blue'
    ax.tick_params(axis='y', labelcolor=color)

    bins = 100
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:grey'
    ax2.set_ylabel('psth', color=color)  # we already handled the x-label with ax1
    ax2.hist(list(chain.from_iterable(tr_list_raster)), bins=bins, range=[-0.500, 0.800], density=True, alpha=0.2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #nplt.show()

    folder = 'rasters_units'
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    plot_title = os.path.join(SaveDir, folder, (un+'.png'))
    plt.savefig(plot_title)
    plt.close('all')
    
    print(un)

# %% Now I do something similar, but per trial! (raster plot of all neurons per trial) [2]
    
iti = []
trials = data_nwb.trials[:][data_nwb.trials[:]['stimset'] == 'natural_images'.encode('utf8')]
for t in range(1,150):
    iti.append((trials.iloc[t]['start_time'] - trials.iloc[t-1]['stop_time']) + 0.200)

iti_ordered = pd.Series(data=iti,index= range(2,len(iti)+2)).sort_values()

units_v1_sorted = data_nwb.units[:].sort_values(by = 'depth')[data_nwb.units[:]['location'] == 'V1']

for tr in range(len(iti_ordered)):
    un_list_raster = []
    
    tr_tmp = iti_ordered.index[tr]
    
    # defiines which periods to take for trial (tr)
    stop = trials.loc[tr_tmp]['stop_time']
    stop_plot = stop + 0.200
    start = stop-0.800
    stop_previous = start-iti_ordered[tr_tmp]
    begin_plot = start - 0.600
    
    for i in units_v1_sorted.index.values: #data_hdf.keys():
        un = str(i)
        timevec = data_hdf[un]['spike_times'][(data_hdf[un]['spike_times'] >= begin_plot) & (data_hdf[un]['spike_times'] < stop_plot)]
        un_list_raster.append(timevec - start)
            
    fig = plt.figure()  # an empty figure with no axes
    fig, ax = plt.subplots(1,1, figsize=(18,8)) 
    ax.eventplot(un_list_raster)
    #ax.yticks(units_v1_sorted['depth'].values)
    #ax.axvline(linewidth=1, color='r')
    ax.axvspan(begin_plot - start, stop_previous - start, alpha=0.1, color='grey')
    ax.axvspan(start - start, stop - start, alpha=0.1, color='blue')
    
    ax.set_xlim(-0.600, 1)
    title = 'Trial ' + str(tr_tmp) + '; Image num ' + str(data_nwb.trials[:].loc[tr_tmp]['img_id']) + '; only V1 neurons (no order)'
    ax.set_title(title)
    ax.set_ylabel('Neurons')
    ax.set_xlabel('Time, sec')
    color = 'tab:blue'
    ax.tick_params(axis='y', labelcolor=color)

    bins = 100
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:grey'
    ax2.set_ylabel('psth', color=color)  # we already handled the x-label with ax1
    ax2.hist(list(chain.from_iterable(un_list_raster)), bins=bins, range=[-0.600, 1.000], density=True, alpha=0.1, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #plt.show()

    folder = 'rasters_by_trials'
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    plot_title = os.path.join(SaveDir, folder, str(tr_tmp) +'.png')
    plt.savefig(plot_title)
    plt.close('all')
    
    print(tr)

# %% Plotting rasters with 4 repetitions per plot [3]

trials = data_nwb.trials[:][data_nwb.trials[:]['stimset'] == 'natural_images'.encode('utf8')]
units_v1_sorted = data_nwb.units[:].sort_values(by = 'depth')[data_nwb.units[:]['location'] == 'V1']

folder = 'rasters_by_trials_repetitions'
if not os.path.exists(folder):
    os.makedirs(folder)

im = 0
plotted = []
pupil=1

while im <= 100:
    imnum = randint(2, 917)
    if imnum in plotted:
        continue
    
    trial_repet = data_nwb.trials[:][data_nwb.trials[:]['img_id'] == str(imnum).encode('utf8')]
    
    fig, axs = plt.subplots(len(trial_repet), 1, figsize=(18,14))
    
    for t, ax in enumerate(fig.axes):
        
        un_list_raster = []
            
        # defiines which periods to take for trial (tr)
        stop = trial_repet.iloc[t]['stop_time']
        stop_plot = stop + 0.200
        start = stop-0.800
        stop_previous = start- (start - data_nwb.trials[:].iloc[(trial_repet.index[t] - 1)]['stop_time'])
        begin_plot = start - 0.600
            
        
        for i in units_v1_sorted.index.values: #data_hdf.keys():
            un = str(i)
            timevec = data_hdf[un]['spike_times'][(data_hdf[un]['spike_times'] >= begin_plot) & (data_hdf[un]['spike_times'] < stop_plot)]
            un_list_raster.append(timevec - start)
            
        if pupil == 1:
            pupil_size = pupil_table['pupil_area'][(pupil_table['time'] >= start-0.600) & (pupil_table['time'] <= start+1)]
            pupil_time = pupil_table['time'][(pupil_table['time'] >= start - 0.600) & (pupil_table['time'] <= start+1)] - start
            pupil_size_rescaled = np.interp(pupil_size, (pupil_size.min(), pupil_size.max()), (1, len(units_v1_sorted)))
                        
        ax.eventplot(un_list_raster)
        ax.axvspan(begin_plot - start, stop_previous - start, alpha=0.1, color='grey')
        ax.axvspan(start - start, stop - start, alpha=0.1, color='blue')
        if pupil == 1:
            ax.plot(pupil_time, pupil_size_rescaled, color = 'green', alpha=0.5)
        
        ax.set_xlim(-0.600, 1)
        title = 'Trial ' + str(trial_repet.index[t]) + '; Image number ' + str(imnum) + '; Previous image num: ' + str(data_nwb.trials[:].iloc[(trial_repet.index[t] - 1)]['img_id'])
        ax.set_title(title)
        ax.set_ylabel('Neurons')
        ax.set_xlabel('Time, sec')
        color = 'tab:blue'
        ax.tick_params(axis='y', labelcolor=color)
    
        bins = 100
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:grey'
        ax2.set_ylabel('psth', color=color)  # we already handled the x-label with ax1
        ax2.hist(list(chain.from_iterable(un_list_raster)), bins=bins, range=[-0.600, 1.000], density=True, alpha=0.1, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 2.5)
        
        fig.tight_layout()    
        
    plot_title = os.path.join(SaveDir, folder, str(imnum) +'.png')
    plt.savefig(plot_title)
    plt.close('all')
    
    print(im)

    plotted.append(imnum)
    im = im + 1
    

# %% Plotting 4 trials that follow the same image [4]
# This is to check if 3-5hz oscillations are triggered by certain images
    
# %% Plotting rasters with 4 repetitions per plot [3]

trials = data_nwb.trials[:][data_nwb.trials[:]['stimset'] == 'natural_images'.encode('utf8')]
units_v1_sorted = data_nwb.units[:].sort_values(by = 'depth')[data_nwb.units[:]['location'] == 'V1']

folder = 'rasters_by_trials_following_same_image'
if not os.path.exists(folder):
    os.makedirs(folder)

pupil = 1
im = 0
plotted = []

while im <= 100:
    imnum = randint(2, 918)
    if imnum in plotted:
        continue
    
    trial_repet = data_nwb.trials[:][data_nwb.trials[:]['img_id'] == str(imnum).encode('utf8')]
    next_trials = data_nwb.trials[:].loc[trial_repet.index.values+1]

    
    fig, axs = plt.subplots(len(trial_repet), 1, figsize=(18,16))
    
    for t, ax in enumerate(fig.axes):
        
        un_list_raster = []
            
        # defiines which periods to take for trial (tr)
        stop_next = next_trials.iloc[t]['stop_time']
        stop_plot = stop_next + 0.200 ## 
        start_next = stop_next - 0.800 #zero
        
        stop_first = trial_repet.iloc[t]['stop_time']
        start_first = stop_first - 0.800 
        begin_plot = start_first - 0.300 ## 
        
        for i in units_v1_sorted.index.values: #data_hdf.keys():
            un = str(i)
            timevec = data_hdf[un]['spike_times'][(data_hdf[un]['spike_times'] >= begin_plot) & (data_hdf[un]['spike_times'] < stop_plot)]
            un_list_raster.append(timevec - start_next)
        
        if pupil == 1:
            pupil_size = pupil_table['pupil_area'][(pupil_table['time'] >= start_next-1.5) & (pupil_table['time'] <= start_next+1)]
            pupil_time = pupil_table['time'][(pupil_table['time'] >= start_next-1.5) & (pupil_table['time'] <= start_next+1)] - start_next
            pupil_size_rescaled = np.interp(pupil_size, (pupil_size.min(), pupil_size.max()), (1, len(units_v1_sorted)))
        
                
        ax.eventplot(un_list_raster)
        ax.axvspan(start_first - start_next, stop_first - start_next, alpha=0.1, color='grey')
        ax.axvspan(start_next - start_next, stop_next- start_next, alpha=0.1, color='blue')
        if pupil == 1:
            ax.plot(pupil_time, pupil_size_rescaled, color = 'green', alpha=0.5)
        
        ax.set_xlim(-1.5, 1)
        title = 'Trial ' + str(trial_repet.index[t]) + ', img ' + str(imnum) + '--> Trial ' + str(next_trials.index[t]) + ', img ' + (next_trials.iloc[t]['img_id']).decode('utf-8')

        ax.set_title(title)
        ax.set_ylabel('Neurons')
        ax.set_xlabel('Time, sec')
        color = 'tab:blue'
        ax.tick_params(axis='y', labelcolor=color)
    
        bins = 100
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:grey'
        ax2.set_ylabel('psth', color=color)  # we already handled the x-label with ax1
        ax2.hist(list(chain.from_iterable(un_list_raster)), bins=bins, range=[-1.5, 1], density=True, alpha=0.1, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 2.5)
        
        fig.tight_layout()
        
    plot_title = os.path.join(SaveDir, folder, str(imnum) +'.png')
    plt.savefig(plot_title)
    plt.close('all')
    
    print(im)

    plotted.append(imnum)
    im = im + 1

# %% 

def get_norm_spike_counts(Sess, bin_dur=0.05):
    
    ResDir = os.path.join(r'C:\Users\slashchevam\Desktop\NPx\Results', Sess)
    if sys.platform == 'win32':
        SaveDir = os.path.join(r'C:\Users\slashchevam\Desktop\NPx\Results', Sess)

    if sys.platform == 'linux':
        SaveDir = os.path.join('/mnt/gs/departmentN4/Marina/NPx_python/', Sess)

    os.chdir(ResDir)

    f = NWBHDF5IO((Sess + '.nwb'), 'r')
    data_nwb = f.read()
    data_hdf = h5py.File((Sess + '_trials.hdf5'), 'r')
    

    dur = max(data_nwb.trials[:]['stop_time']) - min(data_nwb.trials[:]['start_time'])
    n_bins = int(dur/bin_dur)
    bins_vec = np.linspace(min(data_nwb.trials[:]['start_time']), max(data_nwb.trials[:]['stop_time']), num = n_bins)
    
    spike_counts = np.zeros((len(data_hdf.keys()), n_bins-1))
    
    unit_order = []
    index = 0
    for un in data_hdf:
        counts, bin_edges = np.histogram(data_hdf[un]['spike_times'][:], bins=bins_vec)
        spike_counts[index, :] = counts
        unit_order.append(un)
        
        index = index+1
    
    bins_vec = bins_vec[0:len(counts)]
    tr_list = np.empty(len(bins_vec)) * np.nan
    for tr in data_nwb.trials[:].index.values:
        tr_list[(bins_vec >= data_nwb.trials[:].iloc[tr]['start_time']) & (bins_vec <= data_nwb.trials[:].iloc[tr]['stop_time'])] = tr
        
    spike_counts_norm = st.zscore(spike_counts, axis=None)
    data_hdf.close()
    f.close()
    return spike_counts_norm, tr_list, unit_order


    
#    for i in range(len(unit_order)):
#        grp_name = '/' + unit_order[i] + '/zscore_spike_counts_'+ str(bin_dur)
#        grp = data_hdf.require_group(grp_name)
#        grp.require_dataset(grp_name, data = spike_counts_norm[i, :], shape=(spike_counts_norm[i, :].shape), dtype=spike_counts_norm[i, :].dtype, exact=False)
#        
#        grp_name1 = '/' + unit_order[i] + '/spike_counts_'+ str(bin_dur)
#        grp1 = data_hdf.create_group(grp_name1)
#        grp1.create_dataset(grp_name, data = spike_counts[i, :]) 
#        
#        grp_name2 = '/' + unit_order[i] + '/trialnum_bins'
#        grp2 = data_hdf.create_group(grp_name2)
#        grp2.create_dataset(grp_name, data = tr_list) 
#        
        

[spike_counts_norm, tr_list, unit_order] = get_norm_spike_counts(Sess) # 0.05s bin size by default
    
# %%
# Quantify mean pupil size within each trial, then correlate it agains mean FR or variance
    
# Normalize spike trains

trials = data_nwb.trials[:][data_nwb.trials[:]['stimset'] == 'natural_images'.encode('utf8')]
units_v1_sorted = data_nwb.units[:].sort_values(by = 'depth')[data_nwb.units[:]['location'] == 'V1']

pupil_mean = []
pupil_trend = []
spikecount_mean = []
spikecount_variance = []
spikecount_entropy = []

for tr in trials.index.values:
    pupil_size = pupil_table['pupil_area'][(pupil_table['time'] >=  (trials.loc[tr]['start_time']+0.2)) & (pupil_table['time'] <=  trials.loc[tr]['stop_time'])]
    pupil_mean.append(np.mean(pupil_size))
    dif = np.diff(pupil_size)  
    #if sum(dif[dif> 0]) > abs(-4.951480042058392):
    if sum(dif[dif> 0]) > sum(abs(dif[dif< 0])):
        pupil_trend.append(1)
    else:
        pupil_trend.append(-1) 
    
    psth = []
    for u in units_v1_sorted.index.values:
        
        spikes = data_hdf[str(u)]['time_to_onset'][data_hdf[str(u)]['trial_num'][:] == tr] 
        spikes = spikes[spikes >= 0]
        psth.append(spikes)
    
    psth = list(chain.from_iterable(psth))
    
    counts, bin_edges = np.histogram(psth, bins=int(0.8/0.025))
    
    spikecount_mean.append(np.mean(counts))
    spikecount_variance.append(np.var(counts)) #The variance is the average of the squared deviations from the mean
    spikecount_entropy.append(st.entropy(counts))


st.pearsonr(pupil_mean, spikecount_mean)
st.pearsonr(pupil_mean, spikecount_variance)
st.pearsonr(pupil_mean, spikecount_entropy)

slope, intercept, r, p, stderr = st.linregress(pupil_trend, spikecount_entropy)
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}, p={p:.6f}'

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(pupil_trend, spikecount_entropy, linewidth=0, marker='.', label='Data points', color='green')
ax.plot(pupil_trend, intercept + slope *np.asarray(pupil_trend), linewidth = 2,label=line, color='orange')
ax.set_xlabel('pupil_trend')
ax.set_ylabel('spikecount_entropy')
ax.legend(facecolor='white')
#plt.show()
plt.savefig('cat_regression_spikecount_entropy_25ms')



fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(pupil_mean, spikecount_mean, linewidth=0, marker='.', label='Data points', c = pupil_trend)
ax.set_xlabel('pupil_trend')
ax.set_ylabel('spikecount_entropy')
plt.savefig('scatter')




# %% 

spikes_norm = normalize_spike_trains_spont(Sess, 0.05)
spikes_norm = spikes_norm[0]

spikes_norm_nc = spikes_norm.T
spikes_spont_norm = pd.DataFrame(data=spikes_norm_nc)

pearsoncorr = spikes_spont_norm.corr(method='pearson')
plt.subplots(figsize=(20,15))
sb.heatmap(pearsoncorr, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r'
            #annot=True,
            )



from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(spikes)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
















































    
# %%
data_hdf.close()
f.close()
