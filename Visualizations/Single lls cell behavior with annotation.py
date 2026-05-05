# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 16:47:06 2025

@author: Aaron
"""

import multiprocessing
import os
from CustomFunctions import file_management, utils, DetailedBalance, metadata_funcs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
import seaborn as sns
from scipy import stats, signal
from statsmodels.stats.multitest import multipletests



def running_mean_withna(x, N):
    means = []
    if 'full':
        for i, r in enumerate(x):
            if np.isnan(r):
                means.append(np.nan)
            elif i<N:
                #get the window to average
                wind = x[:int(i+1)]
                #remove nan
                wind = wind[~np.isnan(wind)]
                #get average
                means.append(np.mean(wind))
            else:
                #get the indicies around the target value
                first = i - N//2+N%2
                second = first + N
                wind = x[first:second]
                #remove nan
                wind = wind[~np.isnan(wind)]
                #get average
                means.append(np.mean(wind))
    elif 'valid':
        pass
    return np.array(means)





basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_Apply/'
infodir = basedir + 'processed_data/'
datadir = basedir + 'Data_and_Figs/'
cellid = '20240527_488_EGFP-CAAX_640_SPY650-DNA_cell2_01'
specificdir = basedir+'singlecells/'+cellid+'/'
savedir = basedir + 'random/'
time_interval = 5
whichpcs = [1,7]
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col = 0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())

if os.path.exists(specificdir+cellid+'_annotation.csv'):
    annotation = pd.read_csv(specificdir+cellid+'_annotation.csv')
    
    annotation = annotation.rename(columns = {'Time': 'Time(min)'})
    #add the time in seconds
    annotation['time'] = [metadata_funcs.get_sec('00:'+x) for x in annotation['Time(min)'].to_list()]
    #get the names of the data columns 
    datanames = annotation.columns[annotation.isna().any()].tolist()
    #fill NAs with zeros
    annotation.fillna(0, inplace = True)
    #change the datatype of the data columns
    atdict = {x:pd.CategoricalDtype() for x in datanames}
    annotation = annotation.astype(atdict)

#reduce the big dataframe to just the cell of interest
TotalFrame = FullFrame[FullFrame.CellID == cellid].copy()
TotalFrame = pd.merge(TotalFrame, annotation, on = 'time', how='left')


##### quickly calculate aer and cf on the raw CGPS transitions
if os.path.exists(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_transitions_separated.csv'):
    raw_trans = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_transitions_separated.csv', index_col = 0)
    #add a movie columns to separate dataframe on
    raw_trans['Movie'] = [x.split('_frame')[0] for x in raw_trans.cell.to_list()]
    ############# measure aer and cycling frequencies ###########
    #add specific scaling
    xyscaling = [centers[f'PC{whichpcs[0]}'].diff().mean(),centers[f'PC{whichpcs[1]}'].diff().mean()]
    #set the origin to the actual center
    center = [round(nbins/2)]*2
    results = []
    cells = raw_trans[raw_trans.CellID == cellid].copy()
    movielist = sorted(cells.Movie.unique(),key = lambda x: int(x.split('-')[-3]))
    for m in movielist:
        curmov = cells[cells.Movie == m]
        curmov, runs = utils.get_consecutive_timepoints(curmov, 'frame',1)
        for r in runs:
            cell = curmov.iloc[r].reset_index(drop=True)
            results.append(DetailedBalance.get_area_enclosing_rate(
                cell,
                nbins,
                xyscaling,
                center,
                ))
    allaers = pd.concat(results).reset_index(drop=True)
# allaers['cell'] = [c+f'_frame_{int(f)}' for c, f in allaers[['Movie','frame']].values]
#merge aer and cf info
TotalFrame = pd.merge(TotalFrame, allaers[['aer','angular_velocity','cell']],on = 'cell',how='left')




annots = ['front protrusion',
          'front retraction',
          'rear retraction',
          'cell moving']
behaviors = ['speed',
             'Turn_Angle',
             'aer']

fig, axes = plt.subplots(len(annots), len(behaviors), sharey=True)
statlist = []
for i, a in enumerate(annots):
    for u, b in enumerate(behaviors):
        ax = axes[i,u]
        sns.violinplot(x = b, y = a, data=TotalFrame, ax = ax, )
        #do the stats
        annon = TotalFrame.loc[TotalFrame[a]==1, b].dropna()
        annoff = TotalFrame.loc[TotalFrame[a]==0, b].dropna()
        _, pval = stats.mannwhitneyu(annon.values, annoff.values)
        statlist.append({'annotation':a, 'behavior':b, 'pval':pval})
        
plt.tight_layout()

statframe = pd.DataFrame(statlist)
reject, pvcorr = multipletests(statframe['pval'],method='bonferroni')[:2]
sigframe = statframe.iloc[reject]
# siglist = sigframe.Factor.to_list()







################ DO ANY OF THE ANNOTATIONS CORREALTE TEMPORALLY
fig, axes = plt.subplots(len(annots), len(annots))
cors = []
for i, a in enumerate(annots):
    for u, aa in enumerate(annots):
        ax = axes[i,u]
        
        one = TotalFrame[a].copy()
        two = TotalFrame[aa].copy()
        
        cor = signal.correlate(one,two,'full')

        
        # cor /= np.min(cor)
        # len(np.arange(-cell.time.max()+5,cell.time.max()-5,5)), len(cor)
        lags = signal.correlation_lags(len(one), len(two))
    
        cors.append(pd.DataFrame({'first': [a]*len(lags),
                                  'second': [aa]*len(lags),
                                   # 'time': TotalFrame.time,
                                  'correlation':cor}))
        sns.lineplot(data = cors[-1].reset_index(drop=True),
                       x=lags,
                       y = 'correlation',
                       ax = ax)
        ax.set_xlabel(a+'\n'+aa)
        ax.set_ylabel('')
plt.tight_layout()


