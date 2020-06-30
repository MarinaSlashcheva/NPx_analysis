# %% Loaing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path
from scipy import signal as sg

import h5py 
import re


# %%
sr = 30000
video_folder = '/mnt/hpx/projects/OWVinckNatIm/Recordings/Camera/'
video_files = os.listdir(video_folder)

proc_sessions_folder = '/gs/departmentN4/Marina/NPx/Results/'
proc_sessions = os.listdir(proc_sessions_folder)

for Sess in proc_sessions:
    print('Start with session', Sess)
    
    video_title = Sess[0:8] + Sess[10:12] + Sess[13:15] + Sess[16:18]
    
    ## First let's fin dd camera triggers for the specific session
    
    # Finding first ts of the rec
    first_ts_dir = os.path.join('/gs/projects/OWVinckNatIm/NPx_recordings', Sess, 'experiment1/recording1/continuous/Rhythm_FPGA-102.0')
    cont_ts = np.load(os.path.join(first_ts_dir, "timestamps.npy"))
    ts_first = cont_ts[0]
    del cont_ts
    
    # Then load digitals and select camera triggers
    digital_dir = os.path.join('/gs/projects/OWVinckNatIm/NPx_recordings', Sess, 'experiment1/recording1/events/Rhythm_FPGA-102.0/TTL_1')
    
    channel_states = np.load(os.path.join(digital_dir, "channel_states.npy"))
    ts_camera = np.load(os.path.join(digital_dir, "timestamps.npy"))
    ts_camera = ts_camera - ts_first
    camera_ind = [channel_states == 1][0]
    camera_time = ts_camera[camera_ind]/sr # full, uncut camera time
    
    
    # Now let's load output of facemap - facial movement
    for f in video_files:
        if re.match(video_title, f) and f.endswith('.npy'):
            print('Found facemap file', f)
    
            face = np.load(os.path.join(video_folder, f), allow_pickle=True).item()
            face_abs_motion = face['motion'][1]
            face_SVD = face['motSVD'][1][:, 0:3] # this have 3 out of 500 components
    
            del face
    
    # Output of DLC
    for f in video_files:
        if re.match(video_title, f) and f.endswith('.csv'):
            print('Found DLC file', f)
            
            pupil_table = pd.read_csv(os.path.join(video_folder, f),  
                                  sep=",", 
                                  index_col=False)
    
            pupil_table = pupil_table[2:]
            pupil_table.columns = ['ind', 
                                   'upx', 'upy', 'upscore', 
                                   'rightx', 'righty', 'rightscore', 
                                   'downx', 'downy', 'downscore', 
                                   'leftx', 'lefty', 'leftscore']
    
            pupil_table = pupil_table.astype('float')
            
            if len(pupil_table) == len(face_abs_motion):
                print('Number of frames in facemap and DLC matches', len(face_abs_motion))
            else:
                print('Number of frames in facemap and DLC does not match')
    
            # Filter the signal - Adjust manually for your data
            kernel = 15 # must be odd
            
            w = np.abs(pupil_table['rightx'] - pupil_table['leftx'])
            w_medianf = sg.medfilt(w, kernel_size = kernel)
            
            h = np.abs(pupil_table['upy'] - pupil_table['downy'])
            h_medianf = sg.medfilt(h, kernel_size = kernel)
            
            av_rad = (w_medianf + h_medianf)/4
            pupil_area = np.pi * av_rad**2
            print('DLC pupil diameter for session', Sess)
            plt.plot(pupil_area)
        else:
            pupil_area = None
    
    # Resampling the time vector (shrinking it basically to the number of frames)
    manual_check = []
    print('Number of camera frames before cutting is', len(camera_time))
    print('Initial difference is ', len(camera_time) - len(face_abs_motion))
    
    time_borders_ind = np.where(np.diff(camera_time)>0.05)[0]
    time_borders = [camera_time[time_borders_ind[0]+1], camera_time[time_borders_ind[1]]]
    
    if len(time_borders_ind) > 2:
        print('BEACHTUNG! Check timestamps manually')
        manual_check.append(Sess)
    
    print(time_borders_ind, '\n', time_borders, '\n', 'Number of time stamps after cut is ', time_borders_ind[1] - time_borders_ind[0])
    print('Difference after cutting camera ts is ', (time_borders_ind[1] - time_borders_ind[0]) - len(face_abs_motion))
    
    num_frames = len(face_abs_motion)
    print((time_borders_ind[1] - time_borders_ind[0]) - num_frames, 'frames are missing')
    new_camera_time = np.linspace(time_borders[0], time_borders[1], num = num_frames)
    print('New camera time is reconstructed')
    
    # Saving the data frame
    SaveDir = os.path.join('/mnt/gs/departmentN4/Marina/NPx/Results', Sess)
    os.chdir(SaveDir)
    
    if pupil_area is not None:
        video_export = {'time': new_camera_time, 'pupil_area': pupil_area, 'abs_motion':face_abs_motion,
                        'SVD1': face_SVD[:,0], 'SVD2': face_SVD[:,1], 'SVD3': face_SVD[:,2]}
        video_df = pd.DataFrame(video_export, columns=['time', 'pupil_area', 'abs_motion', 'SVD1', 'SVD2', 'SVD3'])
    else:
        video_export = {'time': new_camera_time, 'abs_motion': face_abs_motion,
                        'SVD1': face_SVD[:,0], 'SVD2': face_SVD[:,1], 'SVD3': face_SVD[:,2]}
        video_df = pd.DataFrame(video_export, columns=['time', 'abs_motion', 'SVD1', 'SVD2', 'SVD3'])   
        
    name = Sess + '_video.csv'
    video_df.to_csv(name, index=False)
    print('Saved file for session', Sess, '\n', '--------------------------------', '\n', '\n')




# %%
i = 0
step = 1000
while i < 10000:
    plt.plot(pupil_area[i : i+step])
    plt.show()
    plt.plot(face_abs_motion[i : i+step])
    filt = sg.medfilt(face_abs_motion[i : i+step], kernel_size = 5)
    plt.plot(filt)
    plt.show()
    print(np.corrcoef(pupil_area[i : i+step], filt))
    #plt.plot(face_SVD)
    print(i, i+step)
    i = i + step
    