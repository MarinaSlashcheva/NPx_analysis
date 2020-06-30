# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:05:59 2020

@author: slashchevam
"""
Sess = 'Bl6_177_2020-02-27_14-36-07'


# load face motion data from facemap
facemap_file = 'Bl6_177_200227_0001_proc.npy'
face = np.load(os.path.join(r'\\esi-svfshpx\\projects\\OWVinckNatIm\\Recordings\\Camera', facemap_file), allow_pickle=True).item()

face_abs_motion = face['motion'][1]
face_SVD = face['motSVD'][1] # this have 500 components

del face


# load pupil area

pupil_folder = 'C:\\Users\\slashchevam\\Desktop\\NPx\\videos'
os.chdir(pupil_folder)

pupil_name = Sess + '_pupil.csv'
pupil_table = pd.read_csv(os.path.join(pupil_folder, pupil_name), index_col=False)


plt.plot(pupil_table['pupil_area'][1:1000], color='blue')
plt.plot(face_abs_motion[1:1000], color='red')

plt.plot(face_SVD[1:1000, 0], color='orange')






Sess = 'Bl6_177_2020-02-27_14-36-07'


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


camera_time[-1]- camera_time[0]  # 5552.751566666666
1/0.0407333 # 24.549938256905286
1/0.04071529232047709 #24.560796275974823


diff = camera_time[1:] - camera_time[:-1]
val = diff[diff > 0.05]

np.where(diff == val[0])[0][0]
np.where(diff == val[1])

camera_time_cont = camera_time[np.where(diff == val[0])[0][0]+1 : np.where(diff == val[1])[0][0]+1]



camera_time_cont[-1]- camera_time_cont[0] #5551.5732333333335
