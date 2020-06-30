# -*- coding: utf-8 -*-
"""
Created on Tue May  5 23:06:48 2020

@author: slashchevam
"""

# %%
# Quantify mean pupil size within each trial, then correlate it agains mean FR or variance
    
# Normalize spike trains
bin_dur = 0.05
[spike_counts_norm, tr_list, unit_order] = get_norm_spike_counts(Sess, bin_dur = bin_dur) # 0.05s bin size by default

trials = data_nwb.trials[:][data_nwb.trials[:]['stimset'] == 'natural_images'.encode('utf8')]
units_v1_sorted = data_nwb.units[:].sort_values(by = 'depth')[data_nwb.units[:]['location'] == 'V1']

def partition(array):
  return {i: (array == i).nonzero()[0] for i in np.unique(array)}

unique_val = partition(trials['img_id'].values)

# predefine array for noise corr
noise_corr = np.zeros(shape=(len(unit_order), len(unit_order), len(trials)))

pupil_mean = []
pupil_trend = []

tr_count = 0
for tr in trials.index.values:
    pupil_size = pupil_table['pupil_area'][(pupil_table['time'] >=  (trials.loc[tr]['start_time']+0.2)) & (pupil_table['time'] <=  trials.loc[tr]['stop_time'])]
    pupil_mean.append(np.mean(pupil_size))
    dif = np.diff(pupil_size)  
    #if sum(dif[dif> 0]) > abs(-4.951480042058392):
    if sum(dif[dif> 0]) > sum(abs(dif[dif< 0])):
        pupil_trend.append(1)
    else:
        pupil_trend.append(-1) 
        
    # computing correlations
    
    counts_per_trial = (spike_counts_norm[:, tr_list == tr]).T
    counts_per_trial = pd.DataFrame(data=counts_per_trial[4:, :]) # check this carefylly!!! take bins only after stim ON

    pearsoncorr = counts_per_trial.corr(method='pearson')
    noise_corr[:, :, tr_count] = pearsoncorr
    tr_count = tr_count + 1
    
    if tr_count%100 == 0:
           print('trials processed:', tr, '/', len(data_nwb.trials[:].index.values))

pupil_trend = np.array(pupil_trend)
pupil_mean = np.array(pupil_mean)

plt.subplots(figsize=(12,10))
sb_plot = sb.heatmap(np.nanmean(noise_corr,axis=2),  #pupil_mean < np.mean(pupil_mean)
           vmin=-1, vmax=1,
           xticklabels=pearsoncorr.columns,
           yticklabels=pearsoncorr.columns,
           cmap='RdBu_r'
           #annot=True,
           )

fig = sb_plot.get_figure()
fig.savefig('NC_all_trials_50ms_pearson_pupil_trend_up')

one_set = []
for i in unique_val.keys():
    one_set.append(unique_val[i][0])

noise_corr_onerep = noise_corr[:, :, one_set]

# Now calculate SIGNAL correlation
# first average spike counts over repetitions and then do correlation

signal_corr = np.zeros(shape=(len(unit_order), len(unit_order), len(unique_val)))
im_count = 0

for val in unique_val.keys():
    repeat_im = trials.index.values[trials['img_id'].values == val]
    
    tmp = np.zeros(shape=(len(repeat_im), len(unit_order), int(1/bin_dur)))
    for tr in range(len(repeat_im)):
        tmp[tr, :, :] = (spike_counts_norm[:, tr_list == repeat_im[tr]])
    
    counts_per_trial = (np.mean(tmp, axis=0)).T
    counts_per_trial = pd.DataFrame(data=counts_per_trial[4:, :]) # check this carefylly!!! take bins only after stim ON

    pearsoncorr = counts_per_trial.corr(method='spearman')
    signal_corr[:, :, im_count] = pearsoncorr
    im_count = im_count + 1

    if im_count%100 == 0:
           print('trials processed:', im_count, '/', len(unique_val))
           
           
plt.subplots(figsize=(12,10))
sb_plot = sb.heatmap(np.nanmean(signal_corr,axis=2),  #pupil_mean < np.mean(pupil_mean)
           vmin=-1, vmax=1,
           xticklabels=pearsoncorr.columns,
           yticklabels=pearsoncorr.columns,
           cmap='RdBu_r'
           #annot=True,
           )

fig = sb_plot.get_figure()
fig.savefig('SC_50ms_spearman')


# Signal corr VS noise_corr
# signal_corr, noise_corr_onerep

signal_mean_corr = np.zeros(shape=(1, len(unique_val)))
noise_mean_corr = np.zeros(shape=(1, len(unique_val)))

for img in range(len(unique_val)):
    np.fill_diagonal(signal_corr[:, :, img], 0)
    signal_mean_corr[0, img] = np.nansum(signal_corr[:, :, img]) / (signal_corr[:, :, img].size - len(unit_order))

    np.fill_diagonal(noise_corr_onerep[:, :, img], 0)
    noise_mean_corr[0, img] = np.nansum(noise_corr_onerep[:, :, img]) / (noise_corr_onerep[:, :, img].size - len(unit_order))

fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(signal_mean_corr, noise_mean_corr, linewidth=0, marker='.', label='Data points', color='green')
ax.set_xlabel('mean signal corr')
ax.set_ylabel('mean noise corr')
plt.savefig('mean_noise_vs_signal_corr_per_trial.png')




# Noise corr VS mean pupil size

noise_alltrials_mean = np.zeros(shape=(1, len(trials)))

for tr in range(len(trials)):
    np.fill_diagonal(noise_corr[:, :, tr], 0)
    noise_alltrials_mean[0, tr] = np.nansum(noise_corr[:, :, tr]) / (noise_corr[:, :, tr].size - len(unit_order))


fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(noise_alltrials_mean, pupil_mean, linewidth=0, marker='.', label='Data points', color='orange')
ax.set_xlabel('mean noise corr')
ax.set_ylabel('mean pupil size')
plt.savefig('mean_noise_vs_mean_pupil_per_trial.png')



#st.pearsonr(pupil_mean, spikecount_mean)
#st.pearsonr(pupil_mean, spikecount_variance)
#st.pearsonr(pupil_mean, spikecount_entropy)
#
#slope, intercept, r, p, stderr = st.linregress(pupil_trend, spikecount_entropy)
#line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}, p={p:.6f}'
#
#fig, ax = plt.subplots(figsize=(6,6))
#ax.plot(pupil_trend, spikecount_entropy, linewidth=0, marker='.', label='Data points', color='green')
#ax.plot(pupil_trend, intercept + slope *np.asarray(pupil_trend), linewidth = 2,label=line, color='orange')
#ax.set_xlabel('pupil_trend')
#ax.set_ylabel('spikecount_entropy')
#ax.legend(facecolor='white')
##plt.show()
#plt.savefig('cat_regression_spikecount_entropy_25ms')
#
#
#fig, ax = plt.subplots(figsize=(6,6))
#ax.scatter(pupil_mean, spikecount_mean, linewidth=0, marker='.', label='Data points', c = pupil_trend)
#ax.set_xlabel('pupil_trend')
#ax.set_ylabel('spikecount_entropy')
#plt.savefig('scatter')



# %% Second version where i calculated spike number per second and used all trials for each neuronal pair


bin_dur = 1
[spike_counts_norm, tr_list, unit_order] = get_norm_spike_counts(Sess, bin_dur = bin_dur) # 0.05s bin size by default

trials = data_nwb.trials[:][data_nwb.trials[:]['stimset'] == 'natural_images'.encode('utf8')]
units_v1_sorted = data_nwb.units[:].sort_values(by = 'depth')[data_nwb.units[:]['location'] == 'V1']

spike_counts_natim = spike_counts_norm[:, [x in trials.index.values for x in tr_list]]


fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(spike_counts_natim[0, :], spike_counts_natim[5, :], linewidth=0, marker='.')
ax.set_xlabel('mean signal corr')
ax.set_ylabel('mean noise corr')

pupil_area_rescaled = np.interp(pupil_table['pupil_area'], (pupil_table['pupil_area'].min(), pupil_table['pupil_area'].max()), (0, 100))

pupil_mean = []
pupil_trend = []

for tr in trials.index.values:
    pupil_size = pupil_area_rescaled[(pupil_table['time'] >=  (trials.loc[tr]['start_time'])) & (pupil_table['time'] <=  trials.loc[tr]['stop_time'])]
    pupil_mean.append(np.mean(pupil_size))
    dif = np.diff(pupil_size)  
    #if sum(dif[dif> 0]) > abs(-4.951480042058392):
    if sum(dif[dif> 0]) > sum(abs(dif[dif< 0])):
        pupil_trend.append(1)
    else:
        pupil_trend.append(-1) 

pupil_trend = np.array(pupil_trend)
pupil_mean = np.array(pupil_mean)
        
# computing correlations
    
spike_counts_natimT = spike_counts_natim.T
spike_counts_natimT = pd.DataFrame(data=spike_counts_natimT) 

pearsoncorr = spike_counts_natimT.corr(method='spearman')


plt.subplots(figsize=(12,10))
sb_plot = sb.heatmap(pearsoncorr,  #pupil_mean < np.mean(pupil_mean)
           vmin=-1, vmax=1,
           xticklabels=pearsoncorr.columns,
           yticklabels=pearsoncorr.columns,
           cmap='RdBu_r'
           #annot=True,
           )

fig = sb_plot.get_figure()
fig.savefig('NC_all_trials_1s_spearman.png')





# Now separately for trials with different pupil sizes

# Pupil size more than average
spike_counts_natimT = spike_counts_natim[:, pupil_mean < np.mean(pupil_mean)].T
spike_counts_natimT = pd.DataFrame(data=spike_counts_natimT) 

pearsoncorr = spike_counts_natimT.corr(method='spearman')

plt.subplots(figsize=(12,10))
sb_plot = sb.heatmap(pearsoncorr,  #pupil_mean < np.mean(pupil_mean)
           vmin=-1, vmax=1,
           xticklabels=pearsoncorr.columns,
           yticklabels=pearsoncorr.columns,
           cmap='RdBu_r'
           #annot=True,
           )

fig = sb_plot.get_figure()
fig.savefig('NC_still_trials_1s_spearman.png')


# Pupil size less than average
spike_counts_natimT = spike_counts_natim[:, pupil_mean > np.mean(pupil_mean)].T
spike_counts_natimT = pd.DataFrame(data=spike_counts_natimT) 

pearsoncorr = spike_counts_natimT.corr(method='spearman')

plt.subplots(figsize=(12,10))
sb_plot = sb.heatmap(pearsoncorr,  #pupil_mean < np.mean(pupil_mean)
           vmin=-1, vmax=1,
           xticklabels=pearsoncorr.columns,
           yticklabels=pearsoncorr.columns,
           cmap='RdBu_r'
           #annot=True,
           )

fig = sb_plot.get_figure()
fig.savefig('NC_aroused_trials_1s_spearman.png')


# %% Last version of noise correlation per 4 repetitions - does not seem very different from using all images

# Quantify mean pupil size within each trial, then correlate it agains mean FR or variance
    
# Normalize spike trains
bin_dur = 1
[spike_counts_norm, tr_list, unit_order] = get_norm_spike_counts(Sess, bin_dur = bin_dur) # 0.05s bin size by default
#pupil_area_rescaled = np.interp(pupil_table['pupil_area'], (pupil_table['pupil_area'].min(), pupil_table['pupil_area'].max()), 
#                                ((pupil_table['pupil_area'].min()*100)/pupil_table['pupil_area'].max(), 100))


trials = data_nwb.trials[:][data_nwb.trials[:]['stimset'] == 'natural_images'.encode('utf8')]
units_v1_sorted = data_nwb.units[:].sort_values(by = 'depth')[data_nwb.units[:]['location'] == 'V1']

def partition(array):
  return {i: (array == i).nonzero()[0] for i in np.unique(array)}

unique_val = partition(trials['img_id'].values)

spike_counts_natim = spike_counts_norm[:, [x in trials.index.values for x in tr_list]]
tr_list_natim = tr_list[[x in trials.index.values for x in tr_list]]

noisecorr = np.zeros(shape=(len(unit_order), len(unit_order), 900))

count = 0
for im in unique_val.keys():
    if int(im) > 900:
        continue
    
    same_im = spike_counts_natim[:, unique_val[im]].T
    same_im = pd.DataFrame(data=same_im) # check this carefylly!!! take bins only after stim ON

    pearsoncorr = same_im.corr(method='spearman')
    noisecorr[:, :, count] = pearsoncorr    
    
    
    count = count + 1
    
    
average_noisecorr = np.nanmean(noisecorr, axis=2)

plt.subplots(figsize=(12,10))
sb_plot = sb.heatmap(average_noisecorr,  #pupil_mean < np.mean(pupil_mean)
           vmin=-1, vmax=1,
           xticklabels=pearsoncorr.columns,
           yticklabels=pearsoncorr.columns,
           cmap='RdBu_r'
           #annot=True,
           )

fig = sb_plot.get_figure()
fig.savefig('NC_per4rep_av_1s_spearman.png')

