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
from pathlib import Path


def get_stars(pv):
    if pv < 0.001:
        stars = '***'
    elif pv < 0.01:
        stars = '**'
    elif pv < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'
    return stars


#get directories and open separated datasets
treatments = ['Random','Galvanotaxis']
time_interval = 10 #sec/frame


#get directories and open separated datasets
basedir = Path('E:/Aaron/Combined_37C_Confocal_PCA_planar')
datadir = basedir.joinpath('Data_and_Figs')
FullFrame = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col=0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())
#open the centers of the binned PCs
centers = pd.read_csv(datadir.joinpath('PC_bin_centers.csv'), index_col=0)
#limit data to the galv experiments
TotalFrame = FullFrame[FullFrame.Treatment.isin(treatments)]
TotalFrame.loc[:,'Treatment'] = pd.Categorical(TotalFrame.Treatment.to_list(), categories=treatments, ordered=True)

#flip the rear length along trajectory
TotalFrame.loc[:,'LengthAlongTrajectoryRear'] = TotalFrame['LengthAlongTrajectoryRear'].abs()
#calculate migration angle relative to electric field
TotalFrame.loc[:,'relative_angle'] = [shparam_mod.angle3D(-1, 0, 0, x[0], x[1], x[2]) for i,x in TotalFrame[['Trajectory_X','Trajectory_Y','Trajectory_Z']].iterrows()]




###filter the data for only cells that I have 10 or more frames of
TotalFrame_filtered = TotalFrame[TotalFrame['CellID'].map(TotalFrame['CellID'].value_counts()) >= 10].copy()




############### get list of metrics that are significant ttest of CELL AVERAGES ############
ModeFrame = TotalFrame_filtered.groupby(['Treatment','CellID']).mean().reset_index(level='Treatment')
metriclist = ['Treatment','Cell_Volume','Volume_Front_Ratio','Volume_Right_Ratio','Cell_SurfaceArea','Cell_Sphericity','Cell_Aspect_Ratio',
               'LengthAlongTrajectory','LengthAlongTrajectoryFront','LengthAlongTrajectoryRear','WidthAlongTrajectory',
               'speed','directional_autocorrelation']#,'Turn_Angle','angular_velocity']
pclist = [x for x in TotalFrame.columns.to_list() if 'PC' in x and 'bin' not in x]
includelist = metriclist + pclist


plist = []
for c in includelist:
    if c != 'Treatment':
        samples = [g[1].dropna() for g in ModeFrame.groupby('Treatment')[c]]
        test_stat, p_val = ss.mannwhitneyu(samples[0], samples[1])   
        plist.append({'metric':c, 'pvalue':p_val})
pvdf = pd.DataFrame(plist)
#correct for multiple comparisons
#correct pvalues for metrics
metric_reject, metrics_pvcorr = multipletests(pvdf[pvdf.metric.isin(metriclist)]['pvalue'],method='fdr_bh')[:2]
#correct pvalues for PCs
PC_reject, PC_pvcorr = multipletests(pvdf[pvdf.metric.isin(pclist)]['pvalue'],method='fdr_bh')[:2]
#combine rejected hypotheses
sigframe = pvdf.iloc[np.concatenate((metric_reject, PC_reject))]
#add corrected PCs
sigframe.loc[:,'pvcorr'] = np.concatenate((metrics_pvcorr[metric_reject], PC_pvcorr[PC_reject]))
#all significant comparisons
allsiglist = sigframe.metric.unique()


siglist = ['speed']#,'PC7']#,'Turn_Angle','relative_angle']

ylabels = ['Instantaneous Speed (µm/sec)']#, 'PC7']#,'Turn Angle (°)','Alignment to Electric Field (°)']

############### CELL AVERAGES OF SIGNIFICANT METRICS #################################
colorlist = ['0.65','#8adb93']
sns.set_palette(palette=colorlist)

scale = len(siglist)
linewid= 2

fig, axes = plt.subplots(1,scale,figsize=(scale*4*0.7,4))
for i, sig in enumerate(siglist):
    ax = axes#[i]
    sns.swarmplot(x = 'Treatment', y = sig, data = ModeFrame, size = 2.5, ax = ax, zorder = 1)
    sns.boxplot(x = 'Treatment', y = sig, data = ModeFrame, width = 0.5,
                boxprops={
                    'fill': False,
                    'linewidth': 1.5,
                    'edgecolor': 'black',
                    'zorder':2
                    },
                medianprops={
                    'linewidth': 1.5,
                    'color': 'black'
                    },
                whiskerprops={
                    'linewidth': 0,
                    'color': 'black'
                    },
                capprops={
                    'linewidth': 0,
                    'color': 'black'
                    },
                showfliers=False, ax = ax, zorder=2)
    
    
    #set ylim min to zero if no negative values
    if ax.get_ylim()[0]>0:
        ax.set_ylim(0,ax.get_ylim()[1])
    

    #tick stuff
    ax.set_ylabel(ylabels[i], fontsize = 16)#, labelpad=-0.5)
    ax.set_xlabel('')
    ax.set_xticklabels(['Undirected','Electrotaxis'])
    # ax.set_xticks([])
    # Turn off all spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    #remove legends
    ax.legend_ = None


    #get only the significantly different comparisons
    starframe = sigframe[sigframe.metric == sig].reset_index(drop=True)

    print(f'pval for {ylabels[i]} is {pvdf[pvdf.metric == sig].pvcorr.iloc[0]}')
    pstar = 'n.s.' if starframe.empty else get_stars(starframe['pvcorr'].values[0])
    #use different font sizes for stars vs n.s.
    nsfs = 10 if pstar=='n.s.' else 12

    ymin,ymax = ax.get_ylim()
    ax.text(0.5, ymax-(ymax-ymin)*0.03, pstar, fontsize = nsfs, ha = 'center')
        

plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '_speed.png', dpi = 500, bbox_inches='tight')






siglist = allsiglist

ylabels = [
           'Cell Aspect Ratio',
           ]

############### CELL AVERAGES OF SIGNIFICANT METRICS #################################
sns.set_palette(palette=colorlist)

scale = len(siglist)
linewid= 2

fig, ax = plt.subplots(1,scale,figsize=(scale*4*0.7,4))
for i, sig in enumerate(siglist):
    sns.swarmplot(x = 'Treatment', y = sig, data = ModeFrame, size = 2.5, ax = ax, zorder = 1)
    sns.boxplot(x = 'Treatment', y = sig, data = ModeFrame, width = 0.5,
                boxprops={
                    'fill': False,
                    'linewidth': 1.5,
                    'edgecolor': 'black',
                    'zorder':2
                    },
                medianprops={
                    'linewidth': 1.5,
                    'color': 'black'
                    },
                whiskerprops={
                    'linewidth': 0,
                    'color': 'black'
                    },
                capprops={
                    'linewidth': 0,
                    'color': 'black'
                    },
                showfliers=False, ax = ax, zorder=2)
    
    #set axlim to 1 since that is the lowest value possible
    ax.set_ylim(1,ax.get_ylim()[1])
        
    #tick stuff
    ax.set_ylabel(ylabels[i], fontsize = 16)#, labelpad=-0.5)
    ax.set_xlabel('')
    ax.set_xticklabels(['Undirected','Electrotaxis'])
    # ax.set_xticks([])
    # Turn off all spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    #remove legends
    ax.legend_ = None



    #get only the significantly different comparisons
    starframe = sigframe[sigframe.metric == sig].reset_index(drop=True)

    print(f'pval for {ylabels[i]} is {pvdf[pvdf.metric == sig].pvcorr.iloc[0]}')
    pstar = 'n.s.' if starframe.empty else get_stars(starframe['pvcorr'].values[0])
    #use different font sizes for stars vs n.s.
    nsfs = 10 if pstar=='n.s.' else 12

    ymin,ymax = ax.get_ylim()
    ax.text(0.5, ymax-(ymax-ymin)*0.03, pstar, fontsize = nsfs, ha = 'center')
        

plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')



