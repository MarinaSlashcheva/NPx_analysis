# -*- coding: utf-8 -*-

# Camera test
sr=30000

facemap_file = 'test_proc.npy'
camera = np.load(os.path.join(r'\\esi-svfshpx\\projects\\OWVinckNatIm\\Recordings\\Camera', facemap_file), allow_pickle=True).item()
camera_motion = camera['motion'][1]

del camera
plt.plot(camera_motion[1:500])


test = 'camtestlast_2020-06-15_12-08-44'
#stim_dir = os.path.join(r'\\gs\projects\OWVinckNatIm\NPx_processed\Lev0_condInfo', test)
stim_dir = os.path.join('/gs/projects/OWVinckNatIm/NPx_processed/Lev0_condInfo', test)

mat = scipy.io.loadmat(os.path.join(stim_dir, 'condInfo_01.mat'))
    
SC_stim_labels = mat['StimClass'][0][0][0][0]
SC_stim_present = np.where(mat['StimClass'][0][0][1][0] == 1)[0]
SC_stim_labels_present = SC_stim_labels[SC_stim_present]
    
stim_time = mat['StimClass'][0][0][3][0, SC_stim_present[0]][0][0][0][2]
stim_time[:, 0] = stim_time[:, 0] - stim_time[:, 2]

dur_stim = stim_time[-1, 1] - stim_time[0,0] # 425.2902
sel_frame = np.logical_and(camera_time < stim_time[-1, 1],camera_time > stim_time[0,0])
camera_sel = camera_time[sel_frame]
camera_sel.shape

camera_gray = np.where(camera_motion[13]==0)[0]
trial_begin = np.where(np.diff(camera_gray)>1)[0]
trial_end = np.where(np.diff(np.flip(camera_gray))<-1)[0]
trial_end - trial_begin

# Finding first ts of the rec
#first_ts_dir = os.path.join(r'\\gs\projects\OWVinckNatIm\NPx_recordings', test, r'experiment1\recording1\continuous\Rhythm_FPGA-102.0')
first_ts_dir = os.path.join('/gs/projects/OWVinckNatIm/NPx_recordings', test, 'experiment1/recording1/continuous/Rhythm_FPGA-102.0')

cont_ts = np.load(os.path.join(first_ts_dir, "timestamps.npy"))
ts_first = cont_ts[0]
del cont_ts


# Digitals
#digital_dir = os.path.join(r'\\gs\projects\OWVinckNatIm\NPx_recordings', test, r'experiment1\recording1\events\Rhythm_FPGA-102.0\TTL_1')
digital_dir = os.path.join('/gs/projects/OWVinckNatIm/NPx_recordings', test, 'experiment1/recording1/events/Rhythm_FPGA-102.0/TTL_1')

channel_states = np.load(os.path.join(digital_dir, "channel_states.npy"))
ts_camera = np.load(os.path.join(digital_dir, "timestamps.npy"))

ts_camera = ts_camera - ts_first

camera_ind = [channel_states == 1][0]
camera_time = ts_camera[camera_ind]/sr

diff = camera_time[1:] - camera_time[:-1]
val = diff[diff > 0.05]
np.where(diff == val[0])[0][0]
np.where(diff == val[1])[0][0]


# Resampling the time vector (shrinking it basically to the number of frames)

time_borders_ind = np.where(np.diff(camera_time)>0.05)[0]
time_borders = [camera_time[time_borders_ind[0]+1], camera_time[time_borders_ind[1]]]
print(time_borders_ind, '\n', time_borders, '\n', 'Number of time stamps is ', time_borders_ind[1] - time_borders_ind[0])

##### add 
num_frames = 41636
print((time_borders_ind[1] - time_borders_ind[0]) - num_frames, 'frames are missing')
new_time = np.linspace(time_borders[0], time_borders[1], num = num_frames)








