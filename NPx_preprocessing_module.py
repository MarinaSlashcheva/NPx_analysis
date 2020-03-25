# Module for the preprocessing of NPx recordings obtained in Vinck lab, ESI
# Date: 23.03.2020

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path
import h5py 
import scipy.io
import sys

from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import TimeSeries, NWBHDF5IO, NWBFile, get_manager 
from pynwb.file import Subject

# %%



# %%

def create_nwb_file(Sess, start_time):
    
    sr = 30000
    if sys.platform == 'win32':
        SaveDir = os.path.join(r'C:\Users\slashchevam\Desktop\NPx\Results', Sess)
        RawDataDir = r'C:\Users\slashchevam\Desktop\NPx'
        ExcelInfoPath = RawDataDir
          
        PathToUpload =  os.path.join(RawDataDir , Sess)
    
    if sys.platform != 'win32':
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
    
    excel_info = pd.read_excel((ExcelInfoPath + '\\Recordings_Marina_NPx.xlsx'), sheet_name=Sess)
    
    # Select spikes from good clusters only
    # Have to add the depth of the clusters
    good_clus = cluster_group[cluster_group['group'] == 'good']
    good_clus_info = cluster_info[cluster_group['group'] == 'good']
    print("Found", len(good_clus), ' good clusters') # has depth info
    
    good_spikes_ind = [x in good_clus['cluster_id'].values for x in spike_clusters]
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
    
    if sys.platform != 'win32':
        mat = scipy.io.loadmat(os.path.join((MatlabOutput + Sess), 'condInfo_01.mat'))
    if sys.platform == 'win32':
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

    if sys.platform != 'win32':
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

    if sys.platform != 'win32':
        SaveDir = os.path.join('/mnt/gs/departmentN4/Marina/NPx_python/', Sess)

    os.chdir(ResDir)

    f = NWBHDF5IO((Sess + '.nwb'), 'r')
    data_nwb = f.read()
    data_hdf = h5py.File((Sess + '_trials.hdf5'), 'r')

    nat_im_trials = data_nwb.trials[:].index.values[data_nwb.trials[:]['stimset'].values == ('natural_images').encode('utf8')]
    
    for key in data_hdf.keys():
        print('psth of unit', key)
        
        spikes_for_psth = [x in nat_im_trials for x in data_hdf[key]['trial_num'][:]]
        spike_t_psth = data_hdf[key]['time_to_onset'][:][spikes_for_psth]
        
        num_bins = bins
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(spike_t_psth, num_bins, density=1)
        
        ax.set_xlabel('Time, s')
        ax.set_ylabel('Spiking')
        ax.set_xlim((-0.2, 1))
        title = 'Unit ' + key + ' from ' + data_nwb.units[:][data_nwb.units[:].index == int(key)]['location'].values[0]
        ax.set_title(title)
        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()
    
        folder = 'psth_per_unit'
        if not os.path.exists(folder):
            os.makedirs(folder)
    
        savetitle = os.path.join(SaveDir, folder, (key+'.png'))
        plt.savefig(savetitle)
        plt.close('all')
        
    data_hdf.close()
    f.close()
    
    
    
    
    
    
def raster_spontaneous(Sess, dur): 
    
    """
    Function to plot raster plots for spontaneous activity
    dur - the chunk of the data (in sec) to plot
    
    """
    
    if sys.platform == 'win32':
        SaveDir = os.path.join(r'C:\Users\slashchevam\Desktop\NPx\Results', Sess)

    if sys.platform != 'win32':
        SaveDir = os.path.join('/mnt/gs/departmentN4/Marina/NPx_python/', Sess)

    os.chdir(SaveDir)

    f = NWBHDF5IO((Sess + '.nwb'), 'r')
    data_nwb = f.read()
    data_hdf = h5py.File((Sess + '_trials.hdf5'), 'r')
    
    spont_ind_trials = data_nwb.trials[:].index.values[data_nwb.trials[:]['stimset'].values == ('spontaneous_brightness').encode('utf8')]
    for ind in spont_ind_trials:
        print(ind)
        
        start = data_nwb.trials[ind]['start_time'].values[0]
        stop = data_nwb.trials[ind]['stop_time'].values[0]
        total_dur = stop-start
        
        spont_dic = {}
        for key in data_hdf.keys():
            spont_dic[key] = data_hdf[key]['spike_times'][:][data_hdf[key]['trial_num'][:] == ind]
        spont_table = list(spont_dic.values())
        
        sorting_ind = ((data_nwb.units[:]['depth']).sort_values()).index.values
        dict_order = np.transpose([int(i) for i in list(spont_dic.keys())])
        
        # deepest neurons are on the top 
        spont_table_ordered = []
        for v in sorting_ind:
            spont_table_ordered.append(spont_table[np.where(dict_order == v)[0][0]])
            
        del spont_dic, spont_table
            
        init = 0
        step = dur #seconds
        image_num = 1
    
        while init <= total_dur:
    
            fig = plt.figure()  # an empty figure with no axes
            fig, ax = plt.subplots(1,1, figsize=(18,8)) 

            # raster plot
            temp_raster = []
            for val in spont_table_ordered:
                temp_raster.append(val[(val >= start+init) & (val <= start+init+step)])
    
            ax.eventplot(temp_raster)
            ax.set_xlim(start+init, start+init+step)
            ax.set_title('Raster plot')
            ax.set_ylabel('Neuron')
            ax.set_xlabel('Time, sec')
    
            #ax4 = ax3.twinx()
            #color = 'tab:red'
            #ax4.set_ylabel('iFR', color=color)  # we already handled the x-label with ax1
            #ax4.plot(np.arange(start, stop, bin_size/1000), np.mean(list(spont_visp_iFR.values()), 
            #         axis=0)[0:int(dur*1000/bin_size)], color=color)
            #ax4.tick_params(axis='y', labelcolor=color)
            #fig.tight_layout() 
            
            folder = 'rasters_spontaneous'
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