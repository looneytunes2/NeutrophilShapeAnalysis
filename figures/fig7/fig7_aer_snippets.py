# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 16:51:29 2025

@author: Aaron
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CustomFunctions import utils
from scipy import stats
import seaborn as sns
import matplotlib

whichpcs = [1,7]
time_interval = 5
ntrans = 1
mind = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_Apply/'
datadir = mind + 'Data_and_Figs/'
randir = mind + 'random/'
moviedir = 'E:/Aaron/random_lls/singlecells/'


FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col = 0)
# open aers
allaers = pd.read_csv(randir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_raw_transition_aer_cf.csv', index_col = 0)

#merge aer and cf info
TotalFrame = FullFrame.merge(allaers[['aer','angular_velocity','cell']],left_on='cell',right_on='cell')
TotalFrame = TotalFrame.sort_values(['CellID','time'])

#open all the bootstrapped realizations
bsaers = pd.read_csv(randir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_{ntrans}_transition_Area_Enclosing_Rates.csv', index_col = 0)
bsgaps = pd.read_csv(randir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_{ntrans}_transition_Area_Enclosing_Rates_gaps.csv', index_col = 0)
bsaers_gaps = bsaers.merge(bsgaps, on = ['iter','real_time'])

#only use aers that are within the range of observed time of the real cells
minmaxtime = TotalFrame.groupby('CellID').time.max().min()
itertime = bsaers_gaps.groupby('iter').cumulative_time.max()
longiters = itertime[itertime>=minmaxtime].index.to_list()
bsaers_long = bsaers_gaps[bsaers_gaps.iter.isin(longiters)].copy()

###### get smoothened aer
allcells = []
for i, cell in TotalFrame.groupby('CellID'):
    ### smoothen bootstrapped AE curve
    cell,_,_ = utils.get_aer_state(cell, time_interval)
    #append that cell
    allcells.append(cell)
    
derivframe = pd.concat(allcells).reset_index(drop=True)



###### label snippets of the specified length and get snippet measurements
runlengthlist = [12,24,36] #number of frames in each snippet
aerrunlist = [] #list to append different snippet IDs to
snippetmetrics = [] #list to append snippet metrics to
for rl in runlengthlist:
    srcount = 0
    label = f'aerrun_{(rl-1)*time_interval}'
    # rl = 25 #run length in number of frames
    aertime = np.arange(1,rl)*time_interval
    cellstateruns = []

    for i, cell in derivframe.groupby('CellID'):
        ##### identify consecutive runs of different aer states
        cs, runs = utils.get_consecutive_timepoints(cell, 'time', time_interval)
        for r in runs:
            c = cs.iloc[r].copy()
            allshifts = np.arange(0,len(c),rl)
            stateruns = []
            for n in range(len(allshifts)):
                #treat all equally sized snippets normally
                #the last snippet will almost never be of the right length
                if n!= len(allshifts)-1:
                    #get snippet
                    tempc = c.iloc[allshifts[n]:allshifts[n+1]].copy()
                    tempc.loc[:,label] = srcount
                    cellstateruns.append(tempc[['cell',label]])
                    #measure snippet metrics
                    persistence = tempc.persistence.mean()
                    speed = tempc.speed.mean()

                    #fit aer
                    aerresid, aercoef = utils.fit_AER(tempc,time_interval,'aer_smooth')

                    
                    snippetmetrics.append({
                        'CellID':tempc.iloc[0].CellID,
                        'speed':speed,
                        'persistence':persistence,
                        'aercoef':aercoef,
                        'aerresid':aerresid,
                        'aerrun':label+'_'+str(srcount)})
                    
                    srcount = srcount + 1
                else:
                    tempc = c.iloc[allshifts[n]:len(c)].copy()
                    tempc.loc[:,label] = np.nan
                    cellstateruns.append(tempc[['cell',label]])
    aerrunlist.append(pd.concat(cellstateruns))
    
    
#combine and merge with other data
aerrunframe = derivframe.copy()
for a in aerrunlist:
    aerrunframe = aerrunframe.merge(a, on='cell')

#combine snippet metrics
snippetframe = pd.DataFrame(snippetmetrics)




################ NOW GET AERS FOR BOOTSTRAPPED SNIPPETS


bscells = []
for i, cell in bsaers_long.groupby('iter'):
    cell = cell.rename(columns = {'real_time':'time'})
    ### smoothen bootstrapped AE curve
    cell,_,_ = utils.get_aer_state(cell, time_interval)

    #append that bootstrap
    bscells.append(cell)
   
bsderivframe = pd.concat(bscells).reset_index(drop=True)



bsaerrunlist = [] #list to append different snippet IDs to
bssnippetmetrics = [] #list to append snippet metrics to
for rl in runlengthlist:
    srcount = 0
    label = f'aerrun_{(rl-1)*time_interval}'
    aertime = np.arange(1,rl)*time_interval
    cellstateruns = []
    for i, cell in bsderivframe.groupby('iter'):
        ##### identify consecutive runs of different aer states
        cs, runs = utils.get_consecutive_timepoints(cell, 'time', time_interval)
        for r in runs:
            c = cs.iloc[r].copy()
            allshifts = np.arange(0,len(c),rl)
            stateruns = []
            for n in range(len(allshifts)):
                #treat all equally sized snippets normally
                #the last snippet will almost never be of the right length
                if n!= len(allshifts)-1:
                    #get snippet
                    tempc = c.iloc[allshifts[n]:allshifts[n+1]].copy()
                    tempc.loc[:,label] = srcount
                    cellstateruns.append(tempc[['iter',label,'cumulative_time']])

                    #fit aer
                    aerresid, aercoef = utils.fit_AER(tempc,time_interval,'aer_smooth')

                    
                    bssnippetmetrics.append({
                        'iter':tempc.iloc[0].iter,
                        'aercoef':aercoef,
                        'aerresid':aerresid,
                        'aerrun':label+'_'+str(srcount)})
                    
                    srcount = srcount + 1
                else:
                    tempc = c.iloc[allshifts[n]:len(c)].copy()
                    tempc.loc[:,label] = np.nan
                    cellstateruns.append(tempc[['iter',label,'cumulative_time']])
    bsaerrunlist.append(pd.concat(cellstateruns))
    
#combine and merge with other data
bsaerrunframe = bsderivframe.copy()
for b in bsaerrunlist:
    bsaerrunframe = bsaerrunframe.merge(b, on=['iter','cumulative_time'])

#combine bootstrapped snippet aers
bssnippetframe = pd.DataFrame(bssnippetmetrics)




############# run ttests between the real and bs populations at the different intervals
for r, rl in enumerate(runlengthlist):

    #get data and combine sources
    tempreal = snippetframe[[f'aerrun_{(rl-1)*time_interval}' in x for x in snippetframe.aerrun]].copy()
    tempbs = bssnippetframe[[f'aerrun_{(rl-1)*time_interval}' in x for x in bssnippetframe.aerrun]].copy()
    
    print(stats.ttest_ind(tempreal.aercoef.values, tempbs.aercoef.values))



############## REAL VS BOOTSTRAPPED Mean and std AER over different time periods ##########
#define the colors to make the meshes
set1 = plt.cm.Set1
set2 = plt.cm.Set2
set3 = plt.cm.Set3
cmap = matplotlib.colors.ListedColormap(list(set3.colors)+[set2.colors[-2]] + [set1.colors[-1]])


minute_labels = [str(int(x*time_interval/60))+[' minute',' minutes',' minutes'][i] for i,x in enumerate(runlengthlist)]
fig, axes = plt.subplots(2, len(runlengthlist), figsize = (2*len(runlengthlist),5))
for r, rl in enumerate(runlengthlist):
    meanax = axes[0,r]
    stdax = axes[1,r]
    #get data and combine sources
    #add dummy column
    tempreal = snippetframe[[f'aerrun_{(rl-1)*time_interval}' in x for x in snippetframe.aerrun]].copy()
    #add dummy column
    tempreal['dummy'] = [f'aerrun_{(rl-1)*time_interval}' in x for x in tempreal.aerrun]
    tempbs = bssnippetframe[[f'aerrun_{(rl-1)*time_interval}' in x for x in bssnippetframe.aerrun]].copy()
    #get averages and standard deviations
    temprealmeans = tempreal.groupby('CellID').mean().reset_index()
    temprealstds = tempreal.groupby('CellID').std().reset_index()
    tempbsmeans = tempbs.groupby('iter').mean().reset_index()
    tempbsstds = tempbs.groupby('iter').std().reset_index()
    
    #plot the dots from real cells
    sns.swarmplot(x = 'dummy', y = 'aercoef', data = temprealmeans, hue = 'CellID', palette = cmap.colors, size = 5.5, #marker = 'o',
                   linewidth=0.5, edgecolor = '0.4', ax = meanax, zorder = 2,)

    #plot bootstrapped distribution
    sns.violinplot(data = tempbsstds, y = 'aercoef', linewidth = 0, color = '0.85', inner=None,
                    scale='count', ax = meanax, zorder = 1)
    
    
    #plot the dots from real cells
    sns.swarmplot(x = 'dummy', y = 'aercoef', data = temprealstds, hue = 'CellID', palette = cmap.colors, size = 5.5,
                   linewidth=0.5, edgecolor = '0.4', ax = stdax, zorder = 2,)

    #plot bootstrapped distribution
    sns.violinplot(data = tempbs, y = 'aercoef', linewidth = 0, color = '0.85', inner=None,
                    scale='count', ax = stdax, zorder = 1)
    

    #get rid of legend
    meanax.legend_ = None
    stdax.legend_ = None
    
    
    ###set limits
    meanax.set_ylim(0, 0.017)
    stdax.set_ylim(0, 0.07)
    
    
    #label stuff
    if r == 0:
        meanax.set_ylabel('AER Mean (PC units²/sec)', fontsize = 13)
        stdax.set_ylabel('AER Standard Deviation', fontsize = 13)
        meanax.tick_params(labelsize = 8)
        stdax.tick_params(labelsize = 8)
    else:
        #get rid of axis labels and tick labels for non-edge plots
        meanax.set_ylabel('')
        stdax.set_ylabel('')
        meanax.set_yticklabels([])
        stdax.set_yticklabels([])
        
        
    stdax.set_xlabel(minute_labels[r], fontsize = 11)
    meanax.set_xlabel('')
    meanax.set_xticks([])
    stdax.set_xticks([])
    
    
    #remove spines    
    meanax.spines['top'].set_visible(False)
    meanax.spines['right'].set_visible(False)
    stdax.spines['top'].set_visible(False)
    stdax.spines['right'].set_visible(False)    
    
plt.tight_layout()




plt.savefig(__file__.split('.')[0] + '_real_vs_bs_mean_std_aer.png', dpi = 500, bbox_inches='tight')    


