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
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'galv/'
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())
#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
#limit data to the galv experiments
TotalFrame = FullFrame[FullFrame.Treatment.isin(treatments)]
TotalFrame.loc[:,'Treatment'] = pd.Categorical(TotalFrame.Treatment.to_list(), categories=treatments, ordered=True)

#flip the rear length along trajectory
TotalFrame.loc[:,'LengthAlongTrajectoryRear'] = TotalFrame['LengthAlongTrajectoryRear'].abs()
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
TotalFrame_filtered = TotalFrame[TotalFrame['CellID'].map(TotalFrame['CellID'].value_counts()) >= 10].copy()




############### get list of metrics that are significant ttest of CELL AVERAGES ############
ModeFrame = TotalFrame_filtered.groupby(['Treatment','CellID']).mean().reset_index(level='Treatment')
metriclist = ['Treatment','Cell_Volume','Volume_Front_Ratio','Cell_SurfaceArea','Cell_Sphericity','Cell_Aspect_Ratio',
               'LengthAlongTrajectory','LengthAlongTrajectoryFront','LengthAlongTrajectoryRear','WidthAlongTrajectory',
               'speed','directional_autocorrelation','protrusion_speed','retraction_speed']#,'Turn_Angle','angular_velocity']
pclist = [x for x in TotalFrame.columns.to_list() if 'PC' in x and 'bin' not in x]
includelist = metriclist + pclist


plist = []
for c in includelist:
    if c != 'Treatment':
        samples = [g[1].dropna() for g in ModeFrame.groupby('Treatment')[c]]
        test_stat, p_val = ss.ttest_ind(samples[0], samples[1])   
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
sigframe['pvcorr'] = np.concatenate((metrics_pvcorr[metric_reject], PC_pvcorr[PC_reject]))
#all significant comparisons
allsiglist = sigframe.metric.unique()


siglist = ['speed']#,'PC7']#,'Turn_Angle','relative_angle']

ylabels = ['Instantaneous Speed (µm/sec)']#, 'PC7']#,'Turn Angle (°)','Alignment to Electric Field (°)']

############### CELL AVERAGES OF SIGNIFICANT METRICS #################################
colorlist = ['#a244c9','#6cc46d']
sns.set_palette(palette=colorlist)

scale = len(siglist)
linewid= 2

fig, axes = plt.subplots(1,scale,figsize=(scale*4*0.7,4))
for i, sig in enumerate(siglist):
    ax = axes#[i]
    sns.swarmplot(x = 'Treatment', y = sig, data = ModeFrame, size = 2.5, alpha = 0.6, ax = ax)
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


    #get the adjusted p value of the statistic in question
    pv = metrics_pvcorr[metriclist.index(sig)]
    if pv < 0.001:
        stars = '***'
    elif pv < 0.01:
        stars = '**'
    elif pv < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'
    
    ymin,ymax = ax.get_ylim()
    ax.text(0.5, ymax-(ymax-ymin)*0.03, stars, fontsize = 10, ha = 'center')
        

plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '_speed.png', dpi = 500, bbox_inches='tight')






siglist = allsiglist

ylabels = ['Cell Sphericity',
           'Cell Aspect Ratio',
           'Length Along\nTrajectory (µm)',
           'Rearward Length Along\nTrajectory (µm)',
           'PC7']

############### CELL AVERAGES OF SIGNIFICANT METRICS #################################
colorlist = ['#a244c9','#6cc46d']
sns.set_palette(palette=colorlist)

scale = len(siglist)
linewid= 2

fig, axes = plt.subplots(1,scale,figsize=(scale*4*0.7,4))
for i, sig in enumerate(siglist):
    ax = axes[i]
    sns.swarmplot(x = 'Treatment', y = sig, data = ModeFrame, size = 2.5, alpha = 0.6, ax = ax)
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
    
    
    # #set ylim min to zero if no negative values
    # if ax.get_ylim()[0]>0:
    #     ax.set_ylim(0,ax.get_ylim()[1])
        
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


    #get the adjusted p value of the statistic in question
    pv = sigframe[sigframe.metric == sig].pvcorr.values[0]
    if pv < 0.001:
        stars = '***'
    elif pv < 0.01:
        stars = '**'
    elif pv < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'
    
    ymin,ymax = ax.get_ylim()
    ax.text(0.5, ymax-(ymax-ymin)*0.03, stars, fontsize = 10, ha = 'center')
        

plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')



