# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:26:03 2025

@author: Aaron
"""



import os
import pandas as pd
import numpy as np
import scipy.stats as ss
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import math
from CustomFunctions import utils

#get directories and open separated datasets
treatments = ['DMSO','Para-Nitro-Blebbistatin']
time_interval = 10 #sec/frame


#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'Para-Nitro-Blebbistatin/'
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())
#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
#limit data to the Para-Nitro-Blebbistatin experiments
TotalFrame = FullFrame[FullFrame.Experiment == 'Drug']
dates = [20240624,20240626,20240701,20241125,20241126,20241127]
TotalFrame = TotalFrame[TotalFrame.Date.isin(dates)]
TotalFrame['Treatment'] = pd.Categorical(TotalFrame.Treatment.to_list(), categories=treatments, ordered=True)



#### calculate protrusion and retraction speeds
prsplist = []
for i, cells in TotalFrame.groupby('CellID'):
    cells, runs = utils.get_consecutive_timepoints(cells, 'time', time_interval)
    for r in runs:
        tempcell = cells.iloc[r].copy()
        tempcell['protrusion_speed'] = tempcell.LengthAlongTrajectoryFront.diff()
        tempcell['retraction_speed'] = tempcell.LengthAlongTrajectoryRear.diff()
        prsplist.append(tempcell)
TotalFrame = pd.concat(prsplist).reset_index(drop=True)
###filter the data for only cells that I have 10 or more frames of
TotalFrame_filtered = TotalFrame[TotalFrame['CellID'].map(TotalFrame['CellID'].value_counts()) >= 10]

############### get list of metrics that are significant ttest of CELL AVERAGES ############
ModeFrame = TotalFrame_filtered.groupby(['Treatment','CellID']).mean().reset_index(level='Treatment')

includelist = ['Treatment','Cell_Volume','Volume_Front_Ratio','Cell_SurfaceArea','Cell_Sphericity','Cell_Aspect_Ratio',
               'LengthAlongTrajectory','LengthAlongTrajectoryFront','LengthAlongTrajectoryRear','WidthAlongTrajectory',
               'speed','directional_autocorrelation','Turn_Angle','PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10',
               'protrusion_speed','retraction_speed']#,'angular_velocity']


plist = []
for c in includelist:
    if c != 'Treatment':
        samples = [g[1].dropna() for g in ModeFrame.groupby('Treatment')[c]]
        test_stat, p_val = ss.ttest_ind(samples[0], samples[1])   
        plist.append([c, p_val])

#correct for multiple comparisons
parr = np.array(plist)[:,1].astype('float')
reject, pvcorr = multipletests(parr, method = 'fdr_bh')[:2]
siglist = list(np.array(plist)[reject,0])


ylabels = ['Instantaneous Speed (µm/sec)','Turn Angle (°)']

############### CELL AVERAGES OF SIGNIFICANT METRICS #################################
colorlist = ['#3799de','#d43131']
sns.set_palette(palette=colorlist)

scale = len(siglist)
linewid= 2

fig, axes = plt.subplots(1,scale,figsize=(scale*4*0.7,4))
for i, sig in enumerate(siglist):
    ax = axes[i]
    sns.swarmplot(x = 'Treatment', y = sig, data = ModeFrame, size = 2.5, alpha = 0.6, ax = ax)
    sns.boxplot(x = 'Treatment', y = sig, data = ModeFrame,
                boxprops={
                    'fill': False,
                    'linewidth': linewid,
                    'edgecolor': 'black'
                    },
                medianprops={
                    'linewidth': linewid,
                    'color': 'black'
                    },
                whiskerprops={
                    'linewidth': linewid,
                    'color': 'black'
                    },
                capprops={
                    'linewidth': linewid,
                    'color': 'black'
                    },
                showfliers=False, ax = ax)
    #tick stuff
    ax.set_ylabel(ylabels[i], fontsize = 16)#, labelpad=-0.5)
    ax.set_xlabel('')
    ax.set_xticklabels(['DMSO','Para-Nitro-\nBlebbistatin'])
    # ax.set_xticks([])
    # Turn off all spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
    # # axli[i].set_ylabel(''+sig, fontsize=20)
    # axli[i].set_xlabel('', fontsize=20)
    # axli[i].tick_params('y', labelsize=10)
    # #modify the labels to put bleb in two lines
    # axli[i].set_xticklabels(axli[i].get_xticklabels(), fontsize = 15)
    #remove legends
    ax.legend_ = None


plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')


# #labels
# ax.set_ylabel('Area Enclosing Rate', fontsize=24, labelpad=0)
# ax.set_xlabel('', fontsize=20)
# ax.tick_params('y', labelsize=14)
# ax.set_xticklabels(ax.get_xticklabels(), fontsize = 18)
# ax.axhline(27,xmin=0.25, xmax = 0.75,color = 'black')
# ax.text(0.5,27,'***',fontdict= {'fontsize': 14,
#                                'horizontalalignment':'center'})



