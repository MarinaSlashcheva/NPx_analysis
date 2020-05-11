# Module for the preprocessing of NPx recordings obtained in Vinck lab, ESI
# Date: 23.03.2020

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import gridspec
import matplotlib.patches as mpatches

import os
import os.path
import h5py 
import scipy.io
from scipy import stats as st
import sys
import time

from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import TimeSeries, NWBHDF5IO, NWBFile, get_manager 
from pynwb.file import Subject

# %%



# %%

def create_nwb_file(Sess, start_time):
    
    sr = 30000 #30kHz
    if sys.platform == 'win32':
        SaveDir = os.path.join(r'C:\Users\slashchevam\Desktop\NPx\Results', Sess)
        RawDataDir = r'C:\Users\slashchevam\Desktop\NPx'
        ExcelInfoPath = RawDataDir
          
        PathToUpload =  os.path.join(RawDataDir , Sess)
    
    if sys.platform == 'linux':
        SaveDir = os.path.join('/mnt/gs/departmentN4/Marina/NPx_python/', Sess)
        RawDataDir = '/mnt/gs/projects/OWVinckNatIm/NPx_recordings/'
        PAthToAnalyzed = '/experiment1/recording1/continuous/Neuropix-PXI-100.0/'
        MatlabOutput = '/mnt/gs/projects/OWVinckNatIm/NPx_processed/Lev0_condInfo/'
        ExcelInfoPath = '/mnt/gs/departmentN4/Marina/'
          
        PathToUpload = RawDataDir + Sess + PAthToAnalyzed
    
    # Upload all the data
    spike_stamps = np.load(os.path.join(PathToUpload, "spike_times.npy"))
    spike_times = spike_stamps / sr
    spike_clusters = np.load(os.path.join(PathToUpload, "spike_clusters.npy"))
    cluster_group = pd.read_csv(os.path.join(PathToUpload, "cluster_group.tsv"),  sep="\t")
    cluster_info = pd.read_csv(os.path.join(PathToUpload, "cluster_info.tsv"),  sep="\t")
    
    if len(cluster_group) != len(cluster_info):
        print('Cluster group (manual labeling) and claster info do not match!')
        
    #excel_info = pd.read_excel((ExcelInfoPath + '\\Recordings_Marina_NPx.xlsx'), sheet_name=Sess)
    excel_info = pd.read_excel(os.path.join(ExcelInfoPath + '\\Recordings_Marina_NPx.xlsx'), sheet_name=Sess)


    # Select spikes from good clusters only
    # Have to add the depth of the clusters
    good_clus_info = cluster_info[cluster_info['group'] == 'good'] # has depth info
    good_clus = good_clus_info[['id', 'group']]
    print("Found", len(good_clus), ' good clusters') 
    
    good_spikes_ind = [x in good_clus['id'].values for x in spike_clusters]
    spike_clus_good = spike_clusters[good_spikes_ind]
    spike_times_good = spike_times[good_spikes_ind]
    # spike_stamps_good = spike_stamps[good_spikes_ind]
    
    good_clus_info['area'] = good_clus_info['depth'] > np.max(good_clus_info['depth']) - 1000
    good_clus_info['area'] = good_clus_info['area'].replace(True, 'V1')
    good_clus_info['area'] = good_clus_info['area'].replace(False, 'HPC')
    
    del spike_clusters, spike_times, spike_stamps, good_spikes_ind
    
    
    # Now reading digitals from condInfo
    # This has to be checked carefully again, especially for few stimuli in the session 
    
    # cond class contains the following:
    #   'spontaneous_brightness': dict_keys(['name', 'time', 'timestamps', 'trl_list', 'conf'])
    #   'natural_images': dict_keys(['name', 'time', 'timestamps', 'trl_list', 'conf', 'img_order', 'img_name'])
    
    class condInfo:
        pass
    
    if sys.platform == 'linux':
        mat = scipy.io.loadmat(os.path.join((MatlabOutput + Sess), 'condInfo_01.mat'))
    if sys.platform == 'win32':
        mat = scipy.io.loadmat(os.path.join(PathToUpload, 'condInfo_01.mat'))
    
    SC_stim_labels = mat['StimClass'][0][0][0][0]
    SC_stim_present = np.where(mat['StimClass'][0][0][1][0] == 1)[0]
    SC_stim_labels_present = SC_stim_labels[SC_stim_present]
    
    cond = [condInfo() for i in range(len(SC_stim_labels_present))]

    for stim in range(len(SC_stim_labels_present)):
        cond[stim].name = SC_stim_labels_present[stim][0]
        cond[stim].stiminfo = mat['StimClass'][0][0][3][0, SC_stim_present[stim]][0][0][0][1] # image indices are here
        
        # sorting out digitals for spontaneous activity
        # Need this loop in case there are few periods of spont, recorded like separate blocks
        if SC_stim_labels_present[stim][0] == 'spontaneous_brightness':
            cond[stim].time = []
            cond[stim].timestamps = []
            for block in range(len(mat['StimClass'][0][0][3][0, SC_stim_present[stim]][0][0])):
                print(block)
                cond[stim].time.append(mat['StimClass'][0][0][3][0, SC_stim_present[stim]][0][0][block][2])
                cond[stim].timestamps.append(mat['StimClass'][0][0][3][0, SC_stim_present[stim]][0][0][block][3])
        
        cond[stim].trl_list = mat['StimClass'][0][0][3][0, SC_stim_present[stim]][1]
        cond[stim].conf =  mat['StimClass'][0][0][2][0, SC_stim_present[stim]]# config is very likely wrong and useless
    
        # sorting out digitals for natural images
        if SC_stim_labels_present[stim][0] == 'natural_images':
            cond[stim].time = mat['StimClass'][0][0][3][0, SC_stim_present[stim]][0][0][0][2]
            cond[stim].timestamps =  mat['StimClass'][0][0][3][0, SC_stim_present[stim]][0][0][0][3]
            img_order = []
            for i in range(len(cond[stim].stiminfo)):
                img_order.append(int(cond[stim].stiminfo[i][2]))
            cond[stim].img_order = img_order
            cond[stim].img_name = cond[stim].conf[0][0][0][10][0] # currently not used but might be needed later

    
    # Now create NWB file
    start_time = start_time # datetime(2020, 2, 27, 14, 36, 7, tzinfo=tzlocal())
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
    
    # Did not add it for the moment, later add running as a timeseries and add to HDF5 as binary parameter
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
            #if len(cond[ep].time) > 1:
            for bl in range(len(cond[ep].time)):
                nwbfile.add_epoch(cond[ep].time[bl][0][0], cond[ep].time[bl][0][1], cond[ep].name)
            #else:    
            #    nwbfile.add_epoch(cond[ep].time[0][0], cond[ep].time[0][1], cond[ep].name)
        
        if cond[ep].name == 'natural_images':
            nwbfile.add_epoch(cond[ep].time[0][0], cond[ep].time[-1][1], cond[ep].name)
    
    # Add trials
    # Images names can be also added here
    nwbfile.add_trial_column(name='start', description='start time relative to the stimulus onset')
    nwbfile.add_trial_column(name='stimset', description='the visual stimulus type during the trial')
    nwbfile.add_trial_column(name='img_id', description='image ID for Natural Images')
    
    for ep in range(len(cond)):
        if cond[ep].name == 'spontaneous_brightness':
            #if len(cond[ep].time) > 1:
            for tr in range(len(cond[ep].time)):
                nwbfile.add_trial(start_time = cond[ep].time[tr][0][0], stop_time = cond[ep].time[tr][0][1],
                                  start = cond[ep].time[tr][0][2],
                                  stimset = (cond[ep].name).encode('utf8'), 
                                  img_id = ('gray').encode('utf8'))
                    
#            else:
#                nwbfile.add_trial(start_time = cond[ep].time[0][0], stop_time = cond[ep].time[0][1],
#                                  start = cond[ep].time[0][2],
#                                  stimset = (cond[ep].name).encode('utf8'), 
#                                  img_id = ('gray').encode('utf8'))                   
            
        if cond[ep].name == 'natural_images':
            for tr in range(len(cond[ep].time)):
                nwbfile.add_trial(start_time = cond[ep].time[tr][0], stop_time = cond[ep].time[tr][1],
                                  start = cond[ep].time[tr][2],
                                  stimset = (cond[ep].name).encode('utf8'), 
                                  img_id = (str(cond[ep].img_order[tr])).encode('utf8'))
    
    
    # Write NWB file
    os.chdir(SaveDir)
    name_to_save = Sess + '.nwb'
    io = NWBHDF5IO(name_to_save, manager=get_manager(), mode='w')
    io.write(nwbfile)
    io.close()
    
    del nwbfile
    
# %%    
def create_hdf5_file(Sess):
    #   Create HDF5 file wth the following srtucture:
    #   /unit_id
    #   /unit_id/spike_times
    #   /unit_id/spike_stamps
    #   /unit_id/time_to_onset
    #   /unit_id/trial_num

    sr = 30000
    if sys.platform == 'win32':
        SaveDir = os.path.join(r'C:\Users\slashchevam\Desktop\NPx\Results', Sess)

    if sys.platform == 'linux':
        SaveDir = os.path.join('/mnt/gs/departmentN4/Marina/NPx_python/', Sess)

    os.chdir(SaveDir)
    f = NWBHDF5IO((Sess + '.nwb'), 'r')
    data_nwb = f.read()
        
    hdf5_name = Sess + '_trials.hdf5'
    file = h5py.File(hdf5_name, 'a')
    
    for un in range(len(data_nwb.units[:].index.values)):
        grp = file.create_group(str(data_nwb.units[:].index.values[un]))
        
        #spike_stamps_tmp = spike_stamps_good[spike_clus_good == data_nwb.units[:].index.values[un]]
        trial_num = np.empty((len(data_nwb.units[un]['spike_times'].values[0]))) * np.nan
        trialonset_time = np.empty((len(data_nwb.units[un]['spike_times'].values[0]))) * np.nan
        
        for tr in data_nwb.trials[:].index.values:
            tmp_vec = (data_nwb.units[un]['spike_times'].values[0] >= data_nwb.trials[tr]['start_time'].values[0]) & (data_nwb.units[un]['spike_times'].values[0] < data_nwb.trials[tr]['stop_time'].values[0])
            trial_num[tmp_vec] = tr
            
            if data_nwb.trials[tr]['stimset'].values[0] == ('natural_images').encode('utf8'):
                val = data_nwb.units[un]['spike_times'].values[0][tmp_vec]
                start_t = data_nwb.trials[tr]['start_time'].values[0] + 0.200
                trialonset_time[tmp_vec] = val - start_t
    
            if data_nwb.trials[tr]['stimset'].values[0] == ('spontaneous_brightness').encode('utf8'):
                val = data_nwb.units[un]['spike_times'].values[0][tmp_vec]
                start_t = data_nwb.trials[tr]['start_time'].values[0]
                trialonset_time[tmp_vec] = val - start_t
            
        grp.create_dataset('spike_times', data =  data_nwb.units[un]['spike_times'].values[0])
        #grp.create_dataset('spike_stamps', data = spike_stamps_tmp)
        grp.create_dataset('spike_stamps', data = (data_nwb.units[un]['spike_times'].values[0]*sr).astype(int))
        grp.create_dataset('trial_num', data = trial_num)
        grp.create_dataset('time_to_onset', data = trialonset_time)
        
        print('Processed ', un+1, ' units')
        
    #file.visit(print)
    file.close()
    
    
    
# %%
def FR_barplot(nwbfile):
    
    y_pos = np.arange(len(nwbfile.units[:]))
    col_vec = (np.zeros((len(nwbfile.units[:])))).astype('str')
    col_vec[nwbfile.units[:]['location'].values == 'V1'] = 'darkblue'
    col_vec[nwbfile.units[:]['location'].values == 'HPC'] = 'blue'
    
    plt.bar(y_pos, nwbfile.units[:]['fr'].values, color = col_vec)
    plt.xticks(y_pos, nwbfile.units[:].index.values)
    plt.ylabel('Average FR')
    plt.xlabel('Unit ID')

    plt.show()   




def psth_per_unit_NatIm(Sess, bins):
# Creating PSTH (simple histogram)
    ResDir = os.path.join(r'C:\Users\slashchevam\Desktop\NPx\Results', Sess)
    if sys.platform == 'win32':
        SaveDir = os.path.join(r'C:\Users\slashchevam\Desktop\NPx\Results', Sess)

    if sys.platform == 'linux':
        SaveDir = os.path.join('/mnt/gs/departmentN4/Marina/NPx_python/', Sess)

    os.chdir(ResDir)

    f = NWBHDF5IO((Sess + '.nwb'), 'r')
    data_nwb = f.read()
    data_hdf = h5py.File((Sess + '_trials.hdf5'), 'r')

    nat_im_trials = data_nwb.trials[:].index.values[data_nwb.trials[:]['stimset'].values == ('natural_images').encode('utf8')]
    nat_im = nat_im_trials[(data_nwb.trials[nat_im_trials]['img_id'].values).astype(int) <= 900]
    gray_im = nat_im_trials[(data_nwb.trials[nat_im_trials]['img_id'].values).astype(int) > 900]
    
    for key in data_hdf.keys():
        print('psth of unit', key)
        
        # spikes for 
        spikes_for_psth_nat_im = [x in nat_im for x in data_hdf[key]['trial_num'][:]]
        spike_t_psth_nat_im = data_hdf[key]['time_to_onset'][:][spikes_for_psth_nat_im]
        
        spikes_for_psth_gray_im = [x in gray_im for x in data_hdf[key]['trial_num'][:]]
        spike_t_psth_gray_im = data_hdf[key]['time_to_onset'][:][spikes_for_psth_gray_im]
        
        psth_tmp = []
        for i in range(100):
            vec = [data_hdf[key]['trial_num'][:] == nat_im[i]]
            psth_tmp.append(data_hdf[key]['time_to_onset'][:][vec])

        fig, [ax0, ax1] = plt.subplots(2,1, figsize = (6,6), sharex=True, )
        _, bins, _ = ax0.hist(spike_t_psth_nat_im, bins=bins, range=[-0.2, 0.8], density=True,alpha=0.7, color = 'green')
        _ = ax0.hist(spike_t_psth_gray_im, bins=bins, alpha=0.5, density=True, color='grey')

#        fig, ax = plt.subplots()
#        n, bins, patches = ax.hist(spike_t_psth, num_bins, density=1)
        
        ax0.set_ylabel('Spike count')
        ax0.set_xlim((-0.2, 0.8))
        title = 'Unit ' + key + ', ' + data_nwb.units[:][data_nwb.units[:].index == int(key)]['location'].values[0] + ', FR (hz): ' + str(data_nwb.units[:].loc[int(key)]['fr'])
        ax0.set_title(title)
        green_patch = mpatches.Patch(color='green', label='Nat images')
        gray_patch = mpatches.Patch(color='gray', label='Contol (gray)', alpha=0.7)
        ax0.legend(handles=[green_patch, gray_patch])
        
        ax1.eventplot(psth_tmp, colors='black', linewidths =1.5)
        ax1.set_xlabel('Time, s')
        ax1.set_ylabel('Trials (first 100)')
        ax1.axvline(x=0, ymin=0, ymax=100, alpha=0.7, c='black', ls='--', lw='0.5')
        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()
    
        folder = 'psth_per_unit_v2'
        if not os.path.exists(folder):
            os.makedirs(folder)
    
        savetitle = os.path.join(SaveDir, folder, (key+'.png'))
        plt.savefig(savetitle)
        plt.close('all')
        
    data_hdf.close()
    f.close()
    
    
    
    
def raster_spontaneous(Sess, dur, pupil): 
    
    """
    Function to plot raster plots for spontaneous activity
    dur - the chunk of the data (in sec) to plot
    
    """
    
    if sys.platform == 'win32':
        SaveDir = os.path.join(r'C:\Users\slashchevam\Desktop\NPx\Results', Sess)

    if sys.platform == 'linux':
        SaveDir = os.path.join('/mnt/gs/departmentN4/Marina/NPx_python/', Sess)

    os.chdir(SaveDir)

    f = NWBHDF5IO((Sess + '.nwb'), 'r')
    data_nwb = f.read()
    data_hdf = h5py.File((Sess + '_trials.hdf5'), 'r')
    
    spont_ind_trials = data_nwb.trials[:].index.values[data_nwb.trials[:]['stimset'].values == ('spontaneous_brightness').encode('utf8')]
    
    # rescaling pupil area as a percentage of maximum pupil area
    pupil_area_rescaled = np.interp(pupil['pupil_area'], (pupil['pupil_area'].min(), pupil['pupil_area'].max()), ((pupil['pupil_area'].min()*100)/pupil['pupil_area'].max(), 100))
    
    #pupil_area_rescaled = [(min(pupil['pupil_area']*100)/max(pupil['pupil_area']) for x in pupil['pupil_area'].values]

    
    for ind in spont_ind_trials:
        print(ind)
        
        start = data_nwb.trials[ind]['start_time'].values[0]
        stop = data_nwb.trials[ind]['stop_time'].values[0]
        total_dur = stop-start
        
        spont_dic = {}
        for key in data_hdf.keys():
            spont_dic[key] = data_hdf[key]['spike_times'][:][data_hdf[key]['trial_num'][:] == ind]
        spont_table = list(spont_dic.values())
        
        sorted_values = data_nwb.units[:][['depth', 'location']].sort_values(by='depth')
        sorting_ind = sorted_values.index.values
        sorted_areas = sorted_values['location'].values # for coloring the plot
        dict_order = np.transpose([int(i) for i in list(spont_dic.keys())])
        
        # deepest neurons are on the top 
        spont_table_ordered = []
        for v in sorting_ind:
            spont_table_ordered.append(spont_table[np.where(dict_order == v)[0][0]])
            
        del spont_dic, spont_table
            
        init = 0
        step = dur #seconds
        image_num = 1
        
        # Separate color for each brain area
        areas = set(sorted_areas)
        col = ['C{}'.format(i) for i in range(len(areas))]
        col_list = sorted_areas.copy()
        for c in range(len(col)):
            col_list[sorted_areas == list(areas)[c]] = col[c]
            
    
        while init <= total_dur:
            
            fig = plt.figure(figsize=(16,8))
            gs = gridspec.GridSpec(2,1, height_ratios=[2, 6], hspace = 0) 
            
            temp_pupil = pupil_area_rescaled[(pupil['time'] >= start+init) & (pupil['time'] <= start+init+step)]
            temp_time = pupil['time'][(pupil['time'] >= start+init) & (pupil['time'] <= start+init+step)]
            
            ax0 = plt.subplot(gs[0])
            ax0.plot(temp_time, temp_pupil, color = 'green', alpha = 0.75)
            ax0.set_xlim(start+init, start+init+step)
            ax0.set_ylabel('Pupil area, %', fontsize=18)
            ax0.set_ylim(0, 100)
            ax0.tick_params(axis='y', which='major', labelsize=18)
            
            # raster plot
            temp_raster = []
            for val in spont_table_ordered:
                temp_raster.append(val[(val >= start+init) & (val <= start+init+step)])
            
            ax1 = plt.subplot(gs[1])
            ax1.eventplot(temp_raster, colors = col_list)
            lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in col]
            ax1.legend(lines, list(areas), fontsize = 20)
            ax1.set_xlim(start+init, start+init+step)
            ax1.set_ylabel('Neuron', fontsize=18)
            ax1.set_xlabel('Time, sec', fontsize=18)
            ax1.tick_params(axis='both', which='major', labelsize=18)

            ax0.label_outer()
            
            folder = 'rasters_spontaneous' + str(ind)
            if not os.path.exists(folder):
                os.makedirs(folder)
        
            plot_title = os.path.join(SaveDir, folder, (str(image_num)+'.png'))
            plt.savefig(plot_title)
            plt.close('all')
            
            init = init + step
            print(image_num)
            image_num = image_num + 1
        
    data_hdf.close()
    f.close()
    
    
    
    
def get_norm_spike_counts_spont(Sess, binsize=0.05):
    # binsize in sec: 0.05 s
    
    if sys.platform == 'win32':
        SaveDir = os.path.join(r'C:\Users\slashchevam\Desktop\NPx\Results', Sess)

    if sys.platform == 'linux':
        SaveDir = os.path.join('/mnt/gs/departmentN4/Marina/NPx_python/', Sess)

    os.chdir(SaveDir)

    f = NWBHDF5IO((Sess + '.nwb'), 'r')
    data_nwb = f.read()
    data_hdf = h5py.File((Sess + '_trials.hdf5'), 'r')
    
    spont_ind_trials = data_nwb.trials[:].index.values[data_nwb.trials[:]['stimset'].values == ('spontaneous_brightness').encode('utf8')]        
    units_v1_sorted = data_nwb.units[:].sort_values(by = 'depth')[data_nwb.units[:]['location'] == 'V1']    
    
    print("Found ", len(units_v1_sorted), 'units in V1')
    
    data_epochs = []
    
    for ind in spont_ind_trials:
        # Calculate the duration of spont epoch to get the number of bins for spike counts
        dur = data_nwb.trials[:].loc[ind]['stop_time'] - data_nwb.trials[:].loc[ind]['start_time']
        bins_n = int(dur/binsize)
        bins_vec = np.linspace(data_nwb.trials[:].loc[ind]['start_time'], data_nwb.trials[:].loc[ind]['stop_time'], num = bins_n)
        
        spike_traces = np.zeros((len(units_v1_sorted), bins_n-1))
        
        index = 0
        for i in units_v1_sorted.index.values: #data_hdf.keys():
            un = str(i)
            spikes_tmp = data_hdf[un]['spike_times'][data_hdf[un]['trial_num'][:] == ind]

            counts, bin_edges = np.histogram(spikes_tmp, bins=bins_vec)
            spike_traces[index, :] = counts
            
            index = index+1
            
        # z-score the entire dataset with the same mean and std
        spike_traces_norm = st.zscore(spike_traces, axis=1)
        
        data_epochs.append(spike_traces_norm)    
        print('Finished spontaneous epoch', ind)
    
    data_hdf.close()
    f.close()
    return data_epochs
    
    
def get_norm_spike_counts(Sess, bin_dur=0.200):
    # for now it is only for V1 units
    
    ResDir = os.path.join(r'C:\Users\slashchevam\Desktop\NPx\Results', Sess)
    if sys.platform == 'win32':
        SaveDir = os.path.join(r'C:\Users\slashchevam\Desktop\NPx\Results', Sess)

    if sys.platform == 'linux':
        SaveDir = os.path.join('/mnt/gs/departmentN4/Marina/NPx_python/', Sess)

    os.chdir(ResDir)

    f = NWBHDF5IO((Sess + '.nwb'), 'r')
    data_nwb = f.read()
    data_hdf = h5py.File((Sess + '_trials.hdf5'), 'r')
    
    print('Bin size is', bin_dur, 'sec')
    
    units_v1_sorted = data_nwb.units[:].sort_values(by = 'depth')[data_nwb.units[:]['location'] == 'V1']    
    
    nat_im_trials = data_nwb.trials[:][data_nwb.trials[:]['stimset'].values == ('natural_images').encode('utf8')]
    num_evoked = int(1/bin_dur) * len(nat_im_trials)
    bin_edges_evoked = np.linspace(-0.2, 0.8, int(1/bin_dur)+1)

    spont_trials = data_nwb.trials[:][data_nwb.trials[:]['stimset'].values == ('spontaneous_brightness').encode('utf8')]
    num_spont = 0
    spont_dur = 0 
    for i in spont_trials.index.values:
        dur = spont_trials.loc[i]['stop_time'] - spont_trials.loc[i]['start_time']
        num_bins = int(dur/bin_dur)
        spont_dur = spont_dur + dur
        num_spont = num_spont + num_bins
        
        
    start = time.time()
    spike_counts = np.zeros((len(units_v1_sorted), num_evoked + num_spont))
    tr_list = np.empty(num_evoked + num_spont) * np.nan
   
    tr_cursor = 0
    for tr in data_nwb.trials[:].index.values:
        if data_nwb.trials[:].loc[tr]['stimset'] == ('natural_images').encode('utf8'):
            un_count = 0
            for un in units_v1_sorted.index.values:
                dat_tmp = data_hdf[str(un)]['time_to_onset'][:][data_hdf[str(un)]['trial_num'][:] == tr]
                counts, bin_edges = np.histogram(dat_tmp, bins=bin_edges_evoked)
                
                spike_counts[un_count, tr_cursor: (tr_cursor + int(1/bin_dur))] = counts
                tr_list[tr_cursor: (tr_cursor + int(1/bin_dur))] = tr
                
                un_count = un_count + 1
            tr_cursor = tr_cursor + int(1/bin_dur)
            
        if data_nwb.trials[:].loc[tr]['stimset'] == ('spontaneous_brightness').encode('utf8'):
            un_count = 0
            for un in units_v1_sorted.index.values:
                dat_tmp = data_hdf[str(un)]['spike_times'][:][data_hdf[str(un)]['trial_num'][:] == tr]
                
                dur = spont_trials.loc[tr]['stop_time'] - spont_trials.loc[tr]['start_time']
                
                bin_edges_spont = np.linspace(spont_trials.loc[tr]['start_time'], spont_trials.loc[tr]['stop_time'], int(dur/bin_dur)+1)
                
                counts, bin_edges = np.histogram(dat_tmp, bins=bin_edges_spont)
                
                spike_counts[un_count, tr_cursor: (tr_cursor + int(dur/bin_dur)) ] = counts
                tr_list[tr_cursor: (tr_cursor + int(dur/bin_dur))] = tr
                
                un_count = un_count + 1
            tr_cursor = tr_cursor + int(dur/bin_dur)
            
        if tr%100 == 0:
            print('trials processed:', tr, '/', len(data_nwb.trials[:].index.values))
                
    end = time.time()
    print('time elapsed:' , end - start)
    
    spike_counts_norm = st.zscore(spike_counts, axis=1)
    
    unit_order = units_v1_sorted.index.values
    
    data_hdf.close()
    f.close()
    return spike_counts_norm, tr_list, unit_order    
    
    
    
    
    
    