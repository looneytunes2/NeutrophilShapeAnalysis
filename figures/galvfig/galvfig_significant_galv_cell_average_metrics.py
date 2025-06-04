# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:26:03 2025

@author: Aaron
"""



import pandas as pd
import numpy as np
import scipy.stats as ss
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from CustomFunctions import utils, shparam_mod

#get directories and open separated datasets
treatments = ['Random','Galvanotaxis']
time_interval = 10 #sec/frame


#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_with_galv/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'galv/'
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())
#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
#limit data to the galv experiments
TotalFrame = FullFrame[FullFrame.Treatment.isin(treatments)]
TotalFrame['Treatment'] = pd.Categorical(TotalFrame.Treatment.to_list(), categories=treatments, ordered=True)


#calculate migration angle relative to electric field
TotalFrame['relative_angle'] = [shparam_mod.angle3D(-1, 0, 0, x[0], x[1], x[2]) for i,x in TotalFrame[['Trajectory_X','Trajectory_Y','Trajectory_Z']].iterrows()]


#### calculate protrusion and retraction speeds
prsplist = []
for i, cells in TotalFrame.groupby('CellID'):
    cells, runs = utils.get_consecutive_timepoints(cells, 'time', time_interval)
    for r in runs:
        tempcell = cells.iloc[r].copy()
        tempcell.loc[:,'protrusion_speed'] = tempcell.LengthAlongTrajectoryFront.diff()
        tempcell.loc[:,'retraction_speed'] = tempcell.LengthAlongTrajectoryRear.diff()
        prsplist.append(tempcell)
TotalFrame = pd.concat(prsplist).reset_index(drop=True)
###filter the data for only cells that I have 10 or more frames of
TotalFrame_filtered = TotalFrame[TotalFrame['CellID'].map(TotalFrame['CellID'].value_counts()) >= 10]

############### get list of metrics that are significant ttest of CELL AVERAGES ############
ModeFrame = TotalFrame_filtered.groupby(['Treatment','CellID']).mean().reset_index(level='Treatment')

includelist = ['Treatment','Cell_Volume','Volume_Front_Ratio','Cell_SurfaceArea','Cell_Sphericity','Cell_Aspect_Ratio',
               'LengthAlongTrajectory','LengthAlongTrajectoryFront','LengthAlongTrajectoryRear','WidthAlongTrajectory',
               'speed','Turn_Angle','PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10',
               'protrusion_speed','retraction_speed','relative_angle']#,'directional_autocorrelation','angular_velocity']


plist = []
for c in includelist:
    if c != 'Treatment':
        samples = [g[1].dropna() for g in ModeFrame.groupby('Treatment')[c]]
        test_stat, p_val = ss.ttest_ind(samples[0], samples[1])   
        plist.append([c, p_val])

#correct for multiple comparisons
parr = np.array(plist)[:,1].astype('float')
reject, pvcorr = multipletests(parr, method = 'fdr_bh')[:2]
allsiglist = list(np.array(plist)[reject,0])


siglist = ['speed','Turn_Angle','relative_angle']

ylabels = ['Instantaneous Speed (µm/sec)','Turn Angle (°)','Alignment to Electric Field (°)']

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
    #set ylim min to zero
    ax.set_ylim(0,ax.get_ylim()[1])
        
    #tick stuff
    ax.set_ylabel(ylabels[i], fontsize = 16)#, labelpad=-0.5)
    ax.set_xlabel('')
    ax.set_xticklabels(['DMSO','Para-Nitro-\nBlebbistatin'])
    # ax.set_xticks([])
    # Turn off all spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    

    #remove legends
    ax.legend_ = None


plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')

