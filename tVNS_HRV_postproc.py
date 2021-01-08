#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:38:46 2020

@author: marius
"""

from os import chdir, path, listdir
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
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


#%% functions

def plot_inst_HR(ECG_dict):
    def _get_inst_HR(time = "pre", cond = "verum"):
        if cond == "verum":
            condix = [ix for ix,val in enumerate(ECG_dict['condition']) if not val == "sham" ]
        elif cond == "sham":
            condix = [ix for ix,val in enumerate(ECG_dict['condition']) if val == "sham" ]
        else:
            raise ValueError("undefined value for condition")
                 
        timeix = [ix for ix in condix if ECG_dict['time'][ix] == time]
        HR = [val for ix,val in enumerate(ECG_dict['instHR']) if ix in timeix]
        minlen = np.min([len(v) for v in HR])
        HR = np.array([val[-minlen:] for val in HR])
        return HR
    
    def _get_bl_adjusted_HR():
        pre_verumHR = _get_inst_HR("pre", "verum")
        stim_verumHR = _get_inst_HR("stim", "verum")
        post_verumHR = _get_inst_HR("post", "verum")
        
        
        pre_shamHR = _get_inst_HR("pre", "sham")
        stim_shamHR = _get_inst_HR("stim", "sham")
        post_shamHR = _get_inst_HR("post", "sham")
        
        #cut to identical length 
        minlen = min(np.shape(post_verumHR)[1],np.shape(post_shamHR)[1])
        post_shamHR = post_shamHR[:,:minlen]
        post_verumHR = post_verumHR[:,:minlen]


        transition = min(np.shape(stim_shamHR)[1],np.shape(stim_verumHR)[1])
        stim_shamHR = stim_shamHR[:,-transition:]
        stim_verumHR = stim_verumHR[:,-transition:]
    
    
        def _bl_correct_and_smooth(preHR: np.array,stimHR: np.array,postHR: np.array):  
            stim_post = np.concatenate([stimHR.T,postHR.T])
            bl= np.tile(np.mean(preHR.T,axis = 0,keepdims = True),[np.shape(stim_post)[0],1])
            stim_post -= bl
            stim_post = np.array([np.convolve(stim_post[:,i], np.ones((20000,))/20000, mode='valid') for i in range(np.shape(stim_post)[1])])
            return stim_post
        
        sham_ = _bl_correct_and_smooth(pre_shamHR,stim_shamHR,post_shamHR)
        verum_ = _bl_correct_and_smooth(pre_verumHR,stim_verumHR,post_verumHR)
    
        return sham_, verum_, transition   
    
    
    
    
    sham_, verum_, transition = _get_bl_adjusted_HR()
    
    plt.plot(np.mean(sham_,axis = 0), color = '#F8766D', linewidth=3)
    plt.plot(np.mean(verum_,axis = 0), color = '#00BDC2', linewidth=3)
    plt.vlines(280000,-2,5)
    plt.hlines(0,-5,1e6,linestyles='--')
    plt.xlim(0,400000)
    return



    
def process_RSA(results_dict):
    stimix = [ix for ix,val in enumerate(results_dict['time']) if val == "stim" ]
    shamix = [(ix,val) for ix,val in enumerate(stimix) if results_dict['condition'][val] == 'sham'] #first index is shamix within stim, second index is shamix within whole dataset
    exhix = [(ix,val) for ix,val in enumerate(stimix) if results_dict['condition'][val] == 'ausatmen']
    inhix = [(ix,val) for ix,val in enumerate(stimix) if results_dict['condition'][val] == 'einatmen']
    inh_RSA = np.array([results_dict['resplocked_HR'][ix[0]] for ix in inhix])
    exh_RSA = np.array([results_dict['resplocked_HR'][ix[0]] for ix in exhix])
    sham_RSA = np.array([results_dict['resplocked_HR'][ix[0]] for ix in shamix])
    def _plot_RSA(exh_RSA,inh_RSA,sham_RSA,ylim=(60,80), p1=np.nan, p2 = np.nan, show_sig_thresh = np.nan):
        t=np.linspace(0,10,10000)
        fig,ax =plt.subplots()
        l1,=plt.plot(t,np.mean(exh_RSA,axis = (0)),'#609cff')
        l2,=plt.plot(t,np.mean(inh_RSA,axis = (0)),'#00b634')
        l3,=plt.plot(t,np.mean(sham_RSA,axis = (0)).T,'#f7756b')
        plt.vlines(np.arange(0,4),ymin=-100,ymax=100, colors='r')
        plt.vlines(np.arange(4,8)+.1,ymin=-100,ymax=100, colors='b')
        plt.ylabel('delta_HR / -log10(p)')

        if np.sum(np.isnan(p1)) == 0:
            ph1, =plt.plot(np.linspace(0,10,100), -np.log10(p1), color = 'k')
        if np.sum(np.isnan(p2)) == 0:
            ph2, =plt.plot(np.linspace(0,10,100), -np.log10(p2), color = 'orange')
        if show_sig_thresh == True:
            sh = plt.hlines(-np.log10(.05),xmin=-1,xmax=11, color = 'k', linestyles='--')
            inhh, = plt.fill([0,4,4,0],[10,10,9,9], color="darkgray")
            exhh, = plt.fill([4,8,8,4],[10,10,9,9], color="lightgray")
        plt.ylim(ylim)
        plt.xlim((-.5,10.5))
        if np.sum(np.isnan(p1)) > 0:
            plt.legend([l1,l2,l3],['exhalation-locked','inhalation-locked','sham'])
        else:
            plt.legend([l1,l2,l3,inhh,exhh,ph1,ph2,sh],['exhalation-locked','inhalation-locked','sham', 'inhalation', 'exhalation','-log10(p_C)', '-log10(p_I)', 'sig. threshold (p < .05)'])
            
    def _mean_center(array):
        
        bl = np.mean(array,axis=(1,2),keepdims=True)
        blmat = np.tile(bl,(1,np.shape(array)[1],np.shape(array)[2]))
        return np.mean(array - blmat,axis=1)
    
    c_exh_RSA,c_inh_RSA,c_sham_RSA = _mean_center(exh_RSA),_mean_center(inh_RSA),_mean_center(sham_RSA)
    # z_exh_RSA,z_inh_RSA,z_sham_RSA = np.mean(stats.zscore(exh_RSA,axis=-1),axis=1),np.mean(stats.zscore(inh_RSA,axis=-1),axis=1),np.mean(stats.zscore(sham_RSA,axis=-1),axis=1)

    def RSA_model(df,inhalation = False):
        pandas2ri.activate()
        r_df = pandas2ri.py2rpy(df)
        ro.r.assign("df", r_df)

        ro.r("library(lme4)")
        ro.r("library(lmerTest)")
        if inhalation == True:
            a=ro.r("print(summary(lmer(\"y~inhalation+(1|subjs) + (1+condition|subjs) + (1+location|subjs)\", data = df))$coefficients)")
            s = a[1][3], a[1][4]

        else:
            a=ro.r("print(anova(lmer(\"y~condition+(1|subjs) + (1+condition|subjs) + (1+location|subjs)\", data = df)))")
            s = a['F value']['condition'], a['Pr(>F)']['condition']
        return s
    
    def get_df(exh_RSA=c_exh_RSA, inh_RSA=c_inh_RSA, sham_RSA=c_sham_RSA, ix=None):
        subjs = np.array([results_dict['subjects'][i[1]] for i in inhix])
        location = np.array([results_dict['location'][i[1]] for i in inhix])
        ear = np.array([results_dict['ear'][i[1]] for i in inhix])
        condvec = np.concatenate([np.tile('exh',len(subjs)),np.tile('inh',len(subjs)),np.tile('sham',len(subjs))])
        inhalation = np.concatenate([np.tile('ninh',len(subjs)),np.tile('inh',len(subjs)),np.tile('ninh',len(subjs))])

        if ix == 'max':
            df_dict = {'subjs': np.tile(subjs,3), 'location': np.tile(location,3), 
                'ear': np.tile(ear,3), 'condition': condvec, 'inhalation': inhalation,
                'y': np.squeeze(np.concatenate([np.max(exh_RSA[:,:4000],axis = 1),np.max(inh_RSA[:,:4000],axis = 1),np.max(sham_RSA[:,:4000],axis = 1)])) }
            
        elif ix == 'min':
            df_dict = {'subjs': np.tile(subjs,3), 'location': np.tile(location,3), 
                'ear': np.tile(ear,3), 'condition': condvec, 'inhalation': inhalation,
                'y': np.squeeze(np.concatenate([np.min(exh_RSA[:,5000:],axis = 1),np.min(inh_RSA[:,5000:],axis = 1),np.min(sham_RSA[:,5000:],axis = 1)])) }
            
        else:

            df_dict = {'subjs': np.tile(subjs,3), 'location': np.tile(location,3), 
                       'ear': np.tile(ear,3), 'condition': condvec, 'inhalation': inhalation,
                       'y': np.squeeze(np.concatenate([exh_RSA[:,ix],inh_RSA[:,ix],sham_RSA[:,ix]])) }
            
            
        df = pd.DataFrame.from_dict(df_dict)        
        return df
    
    
    ptp_exh,ptp_inh, ptp_sham = np.ptp(c_exh_RSA,axis = 1),np.ptp(c_inh_RSA,axis = 1),np.ptp(c_sham_RSA,axis = 1)
    max_exh,max_inh, max_sham = np.max(c_exh_RSA[:,:4000],axis = 1),np.max(c_inh_RSA[:,:4000],axis = 1),np.max(c_sham_RSA[:,:4000],axis = 1)
    min_exh,min_inh, min_sham = np.min(c_exh_RSA[:,4000:8000],axis = 1),np.min(c_inh_RSA[:,4000:8000],axis = 1),np.min(c_sham_RSA[:,4000:8000],axis = 1)
    
    
    
    def ptp_plots():
        
        df1 = get_df(ptp_exh,ptp_inh,ptp_sham)
        df2 = get_df(min_exh,min_inh,min_sham)
        df3 = get_df(max_exh,max_inh,max_sham)
        df1['readout'] = 'ptp'
        df2['readout'] = 'min'
        df3['readout'] = 'max'
        frames = [df1, df2, df3]
        

        df = pd.concat(frames)
        r_df = pandas2ri.py2rpy(df)
        ro.r.assign("df", r_df)
        ro.r("""
                 library(ggplot2)
                 library(ggthemes)
                svg("./Figures/RSAptp.svg")
                print(
                  ggplot(df, aes(x = readout, y = y, color = condition)) + geom_boxplot(width = .2,position = position_dodge(width = .3),outlier.size = 0)+
                    geom_point(alpha = .2,position = position_jitterdodge(dodge.width = .3, jitter.width = .08))+
                    ylab("change to mean [bpm]")+
                    scale_color_discrete(limits = c("sham", "inh", "exh"),breaks = c("sham", "inh", "exh"), label = c("sham", "inhalation-locked", "exhalation-locked"))+
                    #scale_x_discrete(breaks = c("b_stim", "c_post"),limits = c("b_stim", "c_post"), label = c("stim", "post"))  
                    theme_clean(base_size = 20)
                )
                try(dev.off(), silent = T);try(dev.off(), silent = T);try(dev.off(), silent = T)

             """)
    stats_ptp = RSA_model(get_df(ptp_exh,ptp_inh,ptp_sham), inhalation = False)

    RSA_model(get_df(min_exh,min_inh,min_sham), inhalation = False)

    RSA_model(get_df(min_exh,min_inh,min_sham), inhalation = True)
    RSA_model(get_df(max_exh,max_inh,max_sham), inhalation = False)

    RSA_model(get_df(max_exh,max_inh,max_sham), inhalation = True)

    
    stats_over_time = [RSA_model(get_df(ix = t)) for t in np.linspace(0,9999,100).astype(int)]
    inhalation_stats = [RSA_model(get_df(ix = t),inhalation=True) for t in np.linspace(0,9999,100).astype(int)]
    

    df = get_df(ix='min')
    r_df = pandas2ri.py2rpy(df)
    ro.r.assign("df", r_df)
    ro.r('library(ggplot2)')


    _plot_RSA(c_exh_RSA, c_inh_RSA, c_sham_RSA,(-10,10), p1=np.array(stats_over_time)[:,1],p2=np.array(inhalation_stats)[:,1], show_sig_thresh=True)

    

    return stats_ptp, stats_over_time, inhalation_stats    

        

    
    
def normalize_to_bl(df, ylist):
    for y in ylist:
        if y == "PNN50":
            df[y+'rel'] = 10*np.log(df[y]/(100-df[y]))
        else:
            df[y+'rel'] = 10*np.log10(df[y])
            
        for r in range(df.shape[0]):
            blix = np.where((df["subjects"] == df["subjects"][r]) & (df["location"] == df["location"][r]) & (df["condition"] == df["condition"][r]) & (df['time'] == 'pre'))[0]
            if y == "PNN50":
                df.loc[r,y+'rel'] -= 10*np.log(np.float(df.loc[blix,y])/(100-np.float(df.loc[blix,y]))) 
            else:
                df.loc[r,y+'rel'] -= 10*np.log10(np.float(df.loc[blix,y]))
    return df

def get_bl_scores(df,ylist):
    for r in range(df.shape[0]):

        for y in ylist:
            blix = np.where((df["subjects"] == df["subjects"][r]) & (df["location"] == df["location"][r]) & (df["condition"] == df["condition"][r]) & (df['time'] == 'pre'))[0]
            df.loc[r,y+'BL'] = np.float(df.loc[blix,y])
    return df

 



def add_NCC(NCC, df):
    df['NCC_LL'] = np.nan*np.zeros(len(df))
    df['NCC_b'] = np.nan*np.zeros(len(df))
    df['NCC_R2'] = np.nan*np.zeros(len(df))
    df = df.reset_index()
    try:
        df = df.drop(["resplocked_HR", "instHR"], axis = 1)
    except:
        pass
    dv = ['HR', "RMSSD",'SDNN', 'LFHF', 'HF', 'SD1SD2', 'PNN50']
    df = normalize_to_bl(df,dv)
    df = get_bl_scores(df,dv)

    for p in NCC['pre']:
       ix = [i for i,v in enumerate(df.path) if p['d'].split('.')[1] in v and p['f'] in v and df.time[i] == 'pre']
       if len(ix) == 1:
           ix=ix[0]
       else:
           raise ValueError
       df.loc[ix, 'NCC_LL'] = p['chanwise_logodds']
       df.loc[ix, 'NCC_b'] = p['chanwise_t']
       df.loc[ix, 'NCC_R2'] = p['R2']
       df.loc[ix+1, 'NCC_LL'] = p['chanwise_logodds']
       df.loc[ix+1, 'NCC_b'] = p['chanwise_t']
       df.loc[ix+1, 'NCC_R2'] = p['R2']
       
       
    for p in NCC['post']:
       ix = [i for i,v in enumerate(df.path) if p['d'].split('.')[1] in v and p['f'] in v and df.time[i] == 'post']
       if len(ix) == 1:
           ix=ix[0]
       else:
           raise ValueError
       df.loc[ix, 'NCC_LL'] = p['chanwise_logodds']
       df.loc[ix, 'NCC_b'] = p['chanwise_t']
       df.loc[ix, 'NCC_R2'] = p['R2']
    
    return  df
    


    
def analyze_NCC(NCC):
    chansel = ["Fp1", "Fp2", "F3", "Fz", "F4","FC3", "FCz", "FC4","Cz"]
    chanix = [ix for ix,val in enumerate(EEGchans) if val in chansel]
    cor = np.array([NCC[i]["R2"] for i in range(len(NCC))])
    directionalcor = np.array([NCC[i]["chanwise_logodds"] for i in range(len(NCC))])
    directionalcor = np.nanmedian(directionalcor[:,chanix],axis=1)  
    
    
    #cor = np.array([np.nanmedian(NCC[i]['chanwise_logodds']) for i in range(len(NCC))])
    results_dict2 = {}
    for key, val in results_dict.items():
        if len(val) == len(results_dict['subjects']):
            results_dict2[key] = val
    df = pd.DataFrame.from_dict(results_dict2).drop(["spectra", "resplocked_HR", "instHR"], axis = 1)
    dv = ['HR', "RMSSD",'SDNN', 'LFHF', 'HF', 'SD1SD2', 'PNN50']
    df = normalize_to_bl(df,dv)
    df = get_bl_scores(df,dv)
    df = df.loc[df.time == "stim",:]
    df["NCC"] = cor
    df["directionalNCC"] = directionalcor

    plt.hist(cor[df.condition !="sham"],bins=15)
    plt.xlabel('neurocardiac coupling [R2]')
    plt.savefig(exdir+'Figures/R2hist.svg')
    plt.close()

    pandas2ri.activate()    
    r_df = pandas2ri.py2rpy(df)
    ro.r.assign("df", r_df)
    ro.r("library(lme4)")
    ro.r("library(lmerTest)")
    t = {}
    p = {}
    directionalt = {}
    directionalp = {}
    for d in dv:
        
        cmd = f'print(summary(lmer(\" {d}rel ~ log(NCC) +  (1|subjects) + (1+condition|subjects) + (1+location|subjects)\", data = df[df$time==\"stim\" & df$condition != \"sham\"& is.finite(df${d}rel),])))$coefficients'

        a = ro.r(cmd)
        t[d] = a[1][-2]
        p[d] = a[1][-1]
        
        cmd = f'print(summary(lmer(\" {d}rel ~ directionalNCC +  (1|subjects) + (1+condition|subjects) + (1+location|subjects)\", data = df[df$time==\"stim\" & df$condition != \"sham\"& is.finite(df${d}rel),])))$coefficients'
        a = ro.r(cmd)
        directionalt[d] = a[1][-2]
        directionalp[d] = a[1][-1]
        
        ro.r(f"""
        library(ggplot2)
        library(ggthemes)
        svg(\"{exdir}Figures/directionalNCC{d}.svg\")
        print(
        ggplot(df[df$time==\"stim\" & df$condition != \"sham\",], aes(x = directionalNCC, y={d}rel )) + geom_point(color = \"black\", alpha = .5) + geom_line(aes(group = subjects),color = \"black\", alpha = .5) + guides(color = FALSE)+
        theme_clean(base_size = 20) + stat_smooth(method = \"lm\",se=FALSE) + ylab(\"log-{d} change from baseline [dB]\") + xlab(\"median t-value for neuro-cardiac coupling\")
        )
        try(dev.off(), silent = T);try(dev.off(), silent = T);try(dev.off(), silent = T)
   
        """)
        ro.r(f"""
        library(ggplot2)
        library(ggthemes)
        svg(\"{exdir}Figures/NCC_{d}.svg\")
        print(
        ggplot(df[df$time==\"stim\" & df$condition != \"sham\",], aes(x = log(NCC), y={d}rel )) + geom_point(aes(color = subjects), alpha = .5) + geom_line(aes(group = subjects, color = subjects), alpha = .5) + guides(color = FALSE)+
        theme_clean(base_size = 20) + stat_smooth(method = \"lm\",se=FALSE) + ylab(\"log-{d} change from baseline [dB]\") + xlab(\"baseline strength of neuro-cardiac coupling [R2]\")
        )
        try(dev.off(), silent = T);try(dev.off(), silent = T);try(dev.off(), silent = T)
   
        """)
 

    

    
    
    ## put everything together
    

    for d in dv:

        ro.r(f"""

             
             m = lmer(\" {d}rel ~  directionalNCC + {d}BL +(1|subjects) \", data = df[df$time==\"stim\" & df$condition == \"sham\" & is.finite(df${d}rel),])
             print(summary(m)$coefficients)
             
                      """)
                   
                      
                   
        ro.r(f"""            
            m = lmer(\" {d}rel ~  directionalNCC + {d}BL +(1|subjects) + (1+condition|subjects) + (1+location|subjects)\", data = df[df$time==\"stim\" & df$condition != \"sham\" & is.finite(df${d}rel),])
             print(summary(m)$coefficients)
               """)
               
        ro.r(f"""
        library(ggplot2)
        library(ggthemes)
        svg(\"{exdir}Figures/BL{d}.svg\")
        print(
        ggplot(df[df$time==\"stim\" & df$condition != \"sham\",], aes(x = log({d}BL), y={d}rel )) + geom_point(aes(color = subjects), alpha = .5) + geom_line(aes(group = subjects, color = subjects), alpha = .5) + guides(color = FALSE)+
        theme_clean(base_size = 20) + stat_smooth(method = \"lm\",se=FALSE) + ylab(\"log-{d} change from baseline [dB]\") + xlab(\"baseline log-{d} [a.u.]\")
        )
        

        
        try(dev.off(), silent = T);try(dev.off(), silent = T);try(dev.off(), silent = T)
   
        """)
        
        
        

             
    pandas2ri.activate()
    r_df = pandas2ri.py2rpy(df)
    dv_ecg=['HR','SDNN', 'corRSA', 'PNN50', 'LFHF', 'HF', 'SD1SD2', 'RMSSD']
    dv = '+'.join(dv_ecg)
    ro.r.assign("df", r_df)

    ro.r("save.image(\"df.RData\")")





if __name__ == "__main__":
    exdir = '/mnt/data/Studies/tVNS_data/'
    chdir(exdir + '/recordings')

    dirs = ['./L/'+ dr for dr in listdir('./L') if path.isdir('./L/' + dr)]
    dirsR = ['./R/'+ dr for dr in listdir('./R') if path.isdir('./R/' + dr)]
    dirs.extend(dirsR)
    
        
        
    GA_resp_beta = []
    GA_resp_cue = []
    GA_beta_corrs = []
    
    #%% import preprocessed data
    with open('./ECG_dict.p', 'rb') as pf:
        ECG_dict = pickle.load(pf)
    
    with open('./EEG_dict.p', 'rb') as pf:
        EEG_dict = pickle.load(pf)
    
    
    results_dict = EEG_dict
    results_dict.update(ECG_dict)
    
    with open('./NCC.p', 'rb') as pf:
        NCC = pickle.load(pf)
    
    
    
    
    with open(exdir + 'EEGchans.p', 'rb') as pf:
        EEGchans = pickle.load(pf)
    EEGchans = EEGchans[:59]
    
    
    #turn results_dict into excel sheet to transfer to R
    results_dict2 = {}
    for key, val in results_dict.items():
        if len(val) == len(results_dict['subjects']):
            results_dict2[key] = val
    df = pd.DataFrame.from_dict(results_dict2).drop(["spectra"], axis = 1)
    
    
        
    plot_inst_HR(ECG_dict)
    plt.savefig('inst_HR.svg')    
    
    
    stats_ptp, stats_over_time, inhalation_stats =process_RSA(results_dict)
    df = add_NCC(NCC,df)
        
    df.to_csv('for_LMM.csv')    
    