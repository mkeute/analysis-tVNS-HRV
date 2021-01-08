#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:05:11 2019

@author: mariuskeute
"""
do_EEG = False
do_NCC =True
do_HEP = False
import pyxdf
from os import chdir, path, listdir
exdir = '/mnt/data/tVNS_data/'
chdir(exdir)
import matplotlib.pyplot as plt
from scipy import signal, stats, interpolate
import numpy as np
import pandas as pd
import pickle
import ECGtoolbox
import math
from pathlib import Path
from os.path import isfile
import json
chdir('./recordings')
dirs = ['./L/'+ dr for dr in listdir('./L') if path.isdir('./L/' + dr)]
dirsR = ['./R/'+ dr for dr in listdir('./R') if path.isdir('./R/' + dr)]
dirs.extend(dirsR)

    
    
GA_resp_beta = []
GA_resp_cue = []
GA_beta_corrs = []


#import processed data

#%% helper functions



def segment_recording(dat, stream_indices):
    markerix = [ix for ix,val in enumerate(stream_indices) if val == 'reiz_marker_sa'][0]
    pre_trial_ix = [ix for ix,val in enumerate(dat[markerix]['time_series']) if val[0] in ['{"pre_trial": 0}', '{"stim_trigger": 0}','{"intervention_trial": 0}']][:2]
    pre_trial = dat[markerix]['time_stamps'][pre_trial_ix]
    
    stim_trial_ix = [[ix for ix,val in enumerate(dat[markerix]['time_series']) if val[0] in ['{"stim_trigger": 0}','{"intervention_trial": 0}']][0]]
    stim_trial_ix.append([ix for ix,val in enumerate(dat[markerix]['time_series']) if val[0] in ['{"post_trial": 0}']][0])
    stim_trial = dat[markerix]['time_stamps'][stim_trial_ix]
    
    post_trial_ix = [ix for ix,val in enumerate(dat[markerix]['time_series']) if val[0] in ['{"post_trial": 0}', 'run_ended']][:2]
    post_trial = dat[markerix]['time_stamps'][post_trial_ix]
    



    resp_times = np.array([dat[markerix]['time_stamps'][ix] for ix,val in enumerate(dat[markerix]['time_series']) if val[0] == 'einatmen' ])
    return pre_trial, stim_trial, post_trial, resp_times

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx



def analyze_ECG(dat, stream_indices, resp_times):

    eegix = [ix for ix,val in enumerate(stream_indices) if val == 'BrainVision RDA'][0]
    EEGchans = [dat[eegix]['info']['desc'][0]['channels'][0]['channel'][n]['label'][0] for n in range(70)]
    EEGclock = dat[eegix]['time_stamps']
    ecgix = [ix for ix,val in enumerate(EEGchans) if val == 'ECG']
    ECG = dat[eegix]['time_series'][:,ecgix]

    pre_ix1 = find_nearest(EEGclock, pre_trial[0])
    pre_ix2 = find_nearest(EEGclock, pre_trial[1])
    if np.abs(pre_trial[0] - EEGclock[pre_ix1]) < .1:
        ecg = ECG[pre_ix1:pre_ix2]
        pre_ECG = ECGtoolbox.ecg_analyzer(ecg, fs = 1000)
    else:
        print('clocks dont match!')
        pre_ECG = np.nan
        
    stim_ix1 = find_nearest(EEGclock, stim_trial[0])
    stim_ix2 = find_nearest(EEGclock, stim_trial[1])
    if np.abs(stim_trial[0] - EEGclock[stim_ix1]) < .1:
        stim_ECG = ECGtoolbox.ecg_analyzer(ECG[stim_ix1:stim_ix2], fs = 1000)
    else:
        stim_ECG = np.nan
        
    
    resp_times = resp_times[(stim_trial[0] < resp_times) & (resp_times < stim_trial[1])]
    resp_ix = [find_nearest(EEGclock, resp_times[i])-stim_ix1 for i in range(len(resp_times))]
    resplocked_HR = []
    for b in resp_ix[1:]:
        resplocked_HR.append(stim_ECG.HR[b:b+10000])
    resplocked_HR = np.array(resplocked_HR)
    
    post_ix1 = find_nearest(EEGclock, post_trial[0])
    post_ix2 = find_nearest(EEGclock, post_trial[1])
    if np.abs(post_trial[0] - EEGclock[post_ix1]) < .1:
        ecg = ECG[post_ix1:post_ix2]
        post_ECG = ECGtoolbox.ecg_analyzer(ecg, fs = 1000)
    else:
        post_ECG = np.nan
    return {'pre':pre_ECG, 'stim':stim_ECG, 'post':post_ECG},resplocked_HR




def get_clean_EEG(dat, stream_indices, resp_times,dirreport):
    from scipy import signal
    from multichannel_tools.visual_inspection import visual_inspection
    eegix = [ix for ix,val in enumerate(stream_indices) if val == 'BrainVision RDA'][0]
    EEGchans = [dat[eegix]['info']['desc'][0]['channels'][0]['channel'][n]['label'][0] for n in range(70)][:59]
    EEGclock = dat[eegix]['time_stamps']
    EEG = dat[eegix]['time_series'][:,:59]
    bandpass = signal.butter(4,(.3, 35), btype = 'pass', fs = 1000)
    notch = signal.butter(4,(48, 52), btype = 'stop', fs = 1000)
    EEG = signal.filtfilt(*bandpass, signal.filtfilt(*notch, EEG, axis = 0),axis = 0)
    chanstoexclude = visual_inspection(np.std(EEG,axis = 0), indexmode = 'exclude')
    
    dirreport['chans_excluded'].append(len(chanstoexclude))
    EEG[:,chanstoexclude] = np.nan
    EEGdat = {'EEG': EEG, 'EEGclock': EEGclock, 'EEGchans': EEGchans}
    EEGdat, dirreport = remove_artifact(EEGdat, dat, stream_indices,dirreport)
    EEGdat['EEG'] -= np.tile(np.nanmean(EEGdat['EEG'], axis = 1, keepdims = True), (1, np.shape(EEGdat['EEG'])[1])) #common avg ref.
    
    
    
    
    
    
    return EEGdat, dirreport

def segment_EEG(EEGdat, resp_times):

    eeg = EEGdat['EEG']
    clock = EEGdat['EEGclock']
    resp_ix = [find_nearest(clock, resp_times[i]) for i in range(len(resp_times))]
    
    EEGsegments = []
    for r in resp_ix:
        sg =eeg[r:r+6000,:] #from each timestamp, cut a 6 second segment
        if len(sg) == 6000:
            EEGsegments.append(sg)
    
    EEGsegments = np.squeeze(np.array(EEGsegments))
    var_ = np.nanvar(EEGsegments,axis = (1,2))
    xcl = np.where(var_ > 2*np.percentile(var_,90))[0]#visual_inspection(np.nanvar(EEGsegments,axis = (1,2)),indexmode = 'exclude')
    EEGsegments[xcl,:,:] = np.nan
    

    EEGdat['EEG'] = EEGsegments
    with open('segmented_EEG.p','wb') as pf:
        pickle.dump(EEGdat, pf)
    return EEGdat


    
def remove_artifact(EEGdat, dat, stream_indices, dirreport):
    from sklearn.decomposition import FastICA
    eeg = EEGdat['EEG']
    ica = FastICA()
    excluded = np.where(np.isnan(eeg[0,:]))[0]
    not_excluded = np.arange(np.shape(eeg)[1])
    ix = np.delete(not_excluded, excluded)
    comps = ica.fit_transform(EEGdat['EEG'][:,ix])
    
    eegix = [ix for ix,val in enumerate(stream_indices) if val == 'BrainVision RDA'][0]
    EEGchans = [dat[eegix]['info']['desc'][0]['channels'][0]['channel'][n]['label'][0] for n in range(70)]
    
    filtparams = signal.butter(4,(.3, 100), btype = "pass", fs = 1000)
    notch = signal.butter(4,(47, 53), btype = "stop", fs = 1000)
    eogix = [ix for ix,val in enumerate(EEGchans) if val == 'ORBITAL']
    VEOG = signal.filtfilt(*filtparams,signal.filtfilt(*notch,dat[eegix]['time_series'][:,eogix],axis=0),axis=0)
    
    filtparams = signal.butter(4,(.3, 100), btype = "pass", fs = 1000)

    eogix = [ix for ix,val in enumerate(EEGchans) if val == 'TEMPOR']
    HEOG = signal.filtfilt(*filtparams,signal.filtfilt(*notch,dat[eegix]['time_series'][:,eogix],axis=0),axis=0)

    filtparams = signal.butter(4,(30), btype = "high", fs = 1000)

    fEMG = [ix for ix,val in enumerate(EEGchans) if val == 'MASSETER']
    fEMG = np.abs(signal.hilbert(np.abs(signal.filtfilt(*filtparams,signal.filtfilt(*notch,dat[eegix]['time_series'][:,eogix],axis=0),axis=0))))

    heogcors = np.array([np.corrcoef(comps[:,i], HEOG.T)[0,1] for i in range(np.shape(comps)[1])])
    veogcors = np.array([np.corrcoef(comps[:,i], VEOG.T)[0,1] for i in range(np.shape(comps)[1])])
    femgcors = np.array([np.corrcoef(np.abs(signal.hilbert(comps[:,i])), fEMG.T)[0,1] for i in range(np.shape(comps)[1])])

    print(f'{np.sum((np.abs(veogcors) > .2) | (np.abs(heogcors) > .2) | (np.abs(femgcors) > .2))} components excluded')
    dirreport['comps_excluded'].append(np.sum((np.abs(veogcors) > .2) | (np.abs(heogcors) > .2) | (np.abs(femgcors) > .2)))
    comps[:,np.abs(veogcors) > .2] = 0
    comps[:,np.abs(heogcors) > .2] = 0
    comps[:,np.abs(femgcors) > .2] = 0

    
    eeg[:,ix] = ica.inverse_transform(comps)
    EEGdat['EEG'] = eeg
    # EEGdat['EEG'] = regress_out_ECG(tmp, ECG)

    return EEGdat, dirreport




def analyze_burstlocked_spectrum(EEGdat, resp_times, dirreport):
    
    if 'einatmen' in f:
        EEGdat = segment_EEG(EEGdat,resp_times+4) #begin segments at the end of inhalation
    elif 'ausatmen' in f:
        EEGdat = segment_EEG(EEGdat,resp_times+8) #begin segments at the end of exhalation
    else:
        EEGdat = segment_EEG(EEGdat,resp_times) #begin segments at the beginning of inhalation


    EEG = EEGdat['EEG']
    EEGchans = EEGdat['EEGchans']
    
    if len(resp_times) > np.shape(EEG)[0]:
        resp_times = resp_times[:-1]
    burst_ix = np.where((stim_trial[0]<resp_times) & (resp_times<stim_trial[1]))
    bl_ix = np.where(resp_times<stim_trial[0])[0]
    post_ix = np.where(resp_times>stim_trial[1])[0]


    stim_EEG = np.squeeze(EEG[burst_ix,:,:])
    retained_trials = np.where([np.invert(np.all(np.isnan(stim_EEG[i,:,:]))) for i in range(np.shape(stim_EEG)[0])])[0]
    dirreport['segments_retained'].append(len(retained_trials))
    stim_EEG = stim_EEG[retained_trials,:,:]
    retained_chans = np.where([np.invert(np.all(np.isnan(stim_EEG[:,:,i]))) for i in range(np.shape(stim_EEG)[-1])])[0]
    burst_pows = np.nan*np.zeros((len(retained_trials), 1025, 59))
    frx, burst_pows[:,:,retained_chans] = signal.welch(stim_EEG[:,:,retained_chans], fs = 1000, axis = 1, nperseg = 2048, noverlap = 1024, average="mean")
    burst_pows = np.nanmean(10*np.log10(burst_pows),axis = 0)


    bl_EEG = np.squeeze(EEG[bl_ix,:,:])
    retained_trials = np.where([np.invert(np.all(np.isnan(bl_EEG[i,:,:]))) for i in range(np.shape(bl_EEG)[0])])[0]
    bl_EEG = bl_EEG[retained_trials,:,:]
    retained_chans = np.where([np.invert(np.all(np.isnan(bl_EEG[:,:,i]))) for i in range(np.shape(bl_EEG)[-1])])[0]
    bl_pows = np.nan*np.zeros((len(retained_trials), 1025, 59))
    frx, bl_pows[:,:,retained_chans] = signal.welch(bl_EEG[:,:,retained_chans], fs = 1000, axis = 1, nperseg = 2048, noverlap = 1024, average="mean")
    bl_pows = np.nanmean(10*np.log10(bl_pows),axis = 0)
    
    for _ in range(3):
        rantrl = np.random.randint(0,np.shape(bl_EEG)[0])
        ranchan = np.random.randint(0,59)
        plt.figure()
        plt.title(d + EEGchans[ranchan])
        plt.plot(stim_EEG[rantrl,:,ranchan])
        plt.plot(bl_EEG[rantrl,:,ranchan])

    post_EEG = np.squeeze(EEG[post_ix,:,:])
    retained_trials = np.where([np.invert(np.all(np.isnan(post_EEG[i,:,:]))) for i in range(np.shape(post_EEG)[0])])[0]
    post_EEG = post_EEG[retained_trials,:,:]
    retained_chans = np.where([np.invert(np.all(np.isnan(post_EEG[:,:,i]))) for i in range(np.shape(post_EEG)[-1])])[0]
    post_pows = np.nan*np.zeros((len(retained_trials), 1025, 59))
    frx, post_pows[:,:,retained_chans] = signal.welch(post_EEG[:,:,retained_chans], fs = 1000, axis = 1, nperseg = 2048, noverlap = 1024, average="mean")
    post_pows = np.nanmean(10*np.log10(post_pows),axis = 0)


    burst_dB = burst_pows - bl_pows 
    
    delta = np.mean(burst_dB[find_nearest(frx,.5):find_nearest(frx,4),:],axis =(0)) #1.4 and 2.8 Hz
    theta = np.mean(burst_dB[find_nearest(frx,4):find_nearest(frx,8),:],axis = (0)) #4.2 - 7.1 Hz
    alpha = np.mean(burst_dB[find_nearest(frx,8):find_nearest(frx,14),:],axis =(0)) #8.5 - 12.8 Hz
    beta = np.mean(burst_dB[find_nearest(frx,14):find_nearest(frx,25),:],axis = (0)) #14.3 - 25.7 Hz
    # import my_topomap
    # my_topomap(beta, EEGchans)
    spectra = {'pre': bl_pows, 'stim': burst_pows, 'post': post_pows}
    return delta, theta, alpha, beta,spectra,EEGchans,dirreport




def get_NCC(EEGdat, ECG, time = 'pre'):
    from scipy import signal
    EEG = EEGdat['EEG']
    EEGclock = EEGdat['EEGclock']
    EEGchans = EEGdat['EEGchans']
    if time == 'pre':
        pre_ix1 = find_nearest(EEGclock, pre_trial[0])
        pre_ix2 = find_nearest(EEGclock, pre_trial[1])
    elif time == 'post':
        pre_ix1 = find_nearest(EEGclock, post_trial[0])
        pre_ix2 = find_nearest(EEGclock, post_trial[1])       
    # if np.abs(pre_trial[0] - EEGclock[pre_ix1]) < .1:
    chansel = ["Fp1", "Fp2", "F3", "Fz", "F4","FC3", "FCz", "FC4"]
    chanix = [ix for ix,val in enumerate(EEGchans) if val in chansel]
    eeg = EEG[pre_ix1:pre_ix2,chanix] 
    
    deltaepochs = []
    for r in ECG[time].r_peaks[1:-1]:
        if r < 750 or r+750 > len(eeg):
            deltaepochs.append(np.nan)
        elif np.nanmax(np.abs(eeg[r-500:r-200,:])) > 80:
            deltaepochs.append(np.nan)
  
        else:
            deltaepochs.append(eeg[r-600:r+600,:])
            
            
    IBI = np.array([np.diff(ECG[time].r_peaks/1000)[i+1] for i,v in enumerate(deltaepochs) if not np.all(np.isnan(v)) and not i+2 > len(ECG['pre'].NNi)])
    deltaepochs = np.array([d for d in deltaepochs if not np.all(np.isnan(d))])
  
    IBI = IBI[:min(len(IBI),len(deltaepochs))]
    deltaepochs = deltaepochs[:min(len(IBI),len(deltaepochs))]
  
  
    f,deltaepochs = signal.welch(deltaepochs,fs =1000, nperseg=deltaepochs.shape[1],axis=1, nfft = 2000)
    deltaepochs = 10*np.log10(deltaepochs)
    deltaepochs = np.nanmean(deltaepochs[:,find_nearest(f,0.5):find_nearest(f,5),:],axis=(1,-1))
    
    
    
    from sklearn.linear_model import LogisticRegression
    from sklearn import preprocessing
    lg = LogisticRegression()
    lgIBI = np.nan*np.zeros(len(IBI))
    lgIBI[IBI > np.percentile(IBI,66.6)] = 1
    lgIBI[IBI < np.percentile(IBI,33.3)] = 0
    lgdelta = deltaepochs[np.where(np.invert(np.isnan(lgIBI)))[0]]
  
    
    
    
    
    y=np.squeeze(lgIBI[np.where(np.invert(np.isnan(lgIBI)))[0]])
    chanwise_coef = lg.fit(preprocessing.scale(lgdelta.reshape(-1, 1)),y).coef_[0][0]
    
    
      
    from sklearn.linear_model import LinearRegression
    
      
    
          
    lm = LinearRegression()
    X = deltaepochs

    lm.fit(preprocessing.scale(X.reshape(-1, 1)),preprocessing.scale(IBI))
    R2 = lm.score(preprocessing.scale(X.reshape(-1, 1)),preprocessing.scale(IBI))
 
    chanwise_t = lm.coef_[0]
      
    return {'R2':R2,'chanwise_t':chanwise_t, 'chanwise_logodds':chanwise_coef}



def check_resp_adherence(dat,stream_indices,resp_times):
    RBix = [ix for ix,val in enumerate(stream_indices) if val == 'GDX-RB_0K2002A1'][0]
    RB = stats.zscore(dat[RBix]['time_series'])
    RBclock = dat[RBix]['time_stamps'] #10Hz
    resp_ix = [find_nearest(RBclock, resp_times[i]) for i in range(len(resp_times))]
    pace=np.tile(np.sin(np.linspace(0,2*np.pi,102)),len(resp_ix))
    RB = np.squeeze(RB[resp_ix[0]:])
    minlen = min(len(pace), len(RB))
    RB = RB[:minlen]
    pace = pace[:minlen]
    def get_PLV(x1,x2):
        sig1_hill=signal.hilbert(x1)
        sig2_hill=signal.hilbert(x2)
        theta1=np.unwrap(np.angle(sig1_hill))
        theta2=np.unwrap(np.angle(sig2_hill))
        complex_phase_diff = np.exp(np.complex(0,1)*(theta1 - theta2))
        plv = np.abs(np.sum(complex_phase_diff))/len(theta1)
        return plv
    return get_PLV(RB, pace)







#%% loop over recordings
# d=dirs[0]
ECG=[]
ECG_dict = {
              'path': [],
              'subjects': [], 
              'ear': [],
              'location': [],
              'condition': [], 
              'HR': [],
              'time': [],
              'SDNN':  [], 
              'corRSA':  [], 
              'PNN50':  [],
              'LFHF':  [],
              'HF':  [],
              'SD1SD2':  [],
              'logRSA': [],
              'RMSSD':  [],
              'resplocked_HR':[],
              'instHR': [],
              'PLV':[]
              }
EEG_dict = {
              'path': [],
              'subjects': [], 
              'ear': [],
              'location': [],
              'condition': [], 
              'time': [],
              'burstlocked_delta':[],
              'burstlocked_theta':[],
              'burstlocked_alpha':[],
              'burstlocked_beta':[],
              'HEP': [],
              'spectra':[],
              'dirreport':[]
              }

NCC = {'pre':[],'post':[]}
#d=dirs[0]
dirreport = {
    'fnames' : [],
    'comps_excluded':[],
    'chans_excluded':[],
    'segments_retained':[]}
for d in dirs:

    print(f'subj {dirs.index(d)}')
    chdir(exdir+'/recordings')
    chdir(d)
    files = [f for f in listdir() if '.xdf' in f and not '.p' in f]

#f = files[0]
    for f in files:
        print(f'file {files.index(f)}')

        dat = pyxdf.load_xdf(f, dejitter_timestamps = True, synchronize_clocks = True)[0]
            # """data structure e.g.:
            #     0 - marker
            #     1 - resp. belt
            #     2 - pupil 18-21 pupil size
            #     3 - EEG marker
            #     4 - EEG data 
            #     !!order is not consistent between datasets!!"""
                
        stream_indices = [dat[n]['info']['name'][0] for n in range(len(dat))]

        
        
        
        #% ECG
        #     # extract ECG and R peaks
        try:
            pre_trial, stim_trial, post_trial, resp_times = segment_recording(dat, stream_indices)
        except:
            continue

        
            
        ECG,resplocked_HR = analyze_ECG(dat, stream_indices, resp_times)
        for key, val in ECG.items():
            if type(val) == float:
                if np.isnan(val):
                    continue
            ECG_dict['path'].append(str(Path(f).absolute()))
            ECG_dict['subjects'].append(d.split('/')[-1])
            ECG_dict['ear'].append(d.split('/')[-2])
            ECG_dict['location'].append(f.split('-')[0])
            ECG_dict['condition'].append(f.split('-')[-1].split('_')[0])
            ECG_dict['time'].append(key)
            ECG_dict['HR'].append(np.nanmean(val.HR))
            ECG_dict['corRSA'].append(val.corRSA)
            ECG_dict['logRSA'].append(val.logRSA)
            ECG_dict['resplocked_HR'].append(resplocked_HR)

            ECG_dict['SD1SD2'].append(val.SD1SD2)
            ECG_dict['RMSSD'].append(val.HRV_time_domain['rmssd'])
            ECG_dict['SDNN'].append(val.HRV_time_domain['sdnn'])
            ECG_dict['HF'].append(val.HRV_frequency_domain['HF'])
            ECG_dict['PNN50'].append(val.HRV_time_domain['pnn50'])
            ECG_dict['LFHF'].append(val.HRV_frequency_domain['LFHF'])
            ECG_dict['instHR'].append(val.HR)
            ECG_dict["PLV"].append(check_resp_adherence(dat,stream_indices,resp_times))
            

            
        if do_NCC:
            pf=f+'clean_EEG.p'
            try:
                with open(pf, 'rb') as p:
                    EEGdat = pickle.load(p)
            except:
                EEGdat,dirreport = get_clean_EEG(dat, stream_indices, resp_times,dirreport)
            tmp=get_NCC(EEGdat, ECG, 'pre')
            tmp['d']=d
            tmp['f']=f

            NCC['pre'].append(tmp)
            
            tmp=get_NCC(EEGdat, ECG, 'post')
            tmp['d']=d
            tmp['f']=f
            NCC['post'].append(tmp)

        if do_EEG:
            pf=f+'clean_EEG.p'
            if isfile(pf):
                with open(pf, 'rb') as p:
                    EEGdat = pickle.load(p)

            else:
                EEGdat,dirreport = get_clean_EEG(dat, stream_indices, resp_times,dirreport)
                with open(pf, 'wb') as p:
                    pickle.dump(EEGdat, p)

    
            pf=f+'segmented_EEG.p'
            if False:#isfile(pf):
                with open(pf, 'rb') as p:
                    delta, theta, alpha, beta,spectra, EEGchans = pickle.load(p)
            else:
                delta, theta, alpha, beta,spectra, EEGchans, dirreport = analyze_burstlocked_spectrum(EEGdat, resp_times, dirreport)
                with open(pf, 'wb') as p:
                    pickle.dump([delta, theta, alpha, beta,spectra, EEGchans], p)
            
            
            EEG_dict['burstlocked_delta'].append(delta)
            EEG_dict['burstlocked_theta'].append(theta)
            EEG_dict['burstlocked_alpha'].append(alpha)
            EEG_dict['burstlocked_beta'].append(beta)
            EEG_dict['spectra'].append(spectra['pre'])
            EEG_dict['spectra'].append(spectra['stim'])
            EEG_dict['spectra'].append(spectra['post'])
    
    

            EEG_dict['dirreport'].append(dirreport)#

        del dat


import pickle
if do_EEG:
    with open(exdir + 'EEG_dict.p', 'wb') as pf:
        pickle.dump(EEG_dict, pf)
with open(exdir + 'ECG_dict.p', 'wb') as pf:
    pickle.dump(ECG_dict, pf)
    
with open(exdir + 'NCC.p', 'wb') as pf:
    pickle.dump(NCC, pf)
