# -*- coding: utf-8 -*-

# %% Load packages etc.
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

sys.path.append(r'C:\Users\slashchevam\Desktop\NPx\NPx_analysis')
from NPx_preprocessing_module import *

# %%

# Choose the session
Sess = 'Bl6_177_2020-02-29_17-12-05'
Sess = 'Bl6_177_2020-02-27_14-36-07'

if sys.platform == 'win32':
    SaveDir = os.path.join(r'C:\Users\slashchevam\Desktop\NPx\Results', Sess)

if sys.platform != 'win32':
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


# Add proper path for that! 
psth_per_unit_NatIm(Sess, 100)

# %%Raster plot for spontaneous activity
raster_spontaneous(Sess, 10) #dur in sec


sum(data_nwb.units[:]['location'].values == 'HPC')



# %% 
data_hdf.close()
f.close()