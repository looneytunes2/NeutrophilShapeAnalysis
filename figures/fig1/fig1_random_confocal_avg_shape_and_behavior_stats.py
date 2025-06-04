# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:42:54 2025

@author: Aaron
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



#get directories and open separated datasets


treatments = ['Random']

#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
#restrict dataframe to only random experiments
TotalFrame = FullFrame[FullFrame.Treatment.isin(treatments)]
#limit the dataframe to only tracks with at least 10 frames
TotalFrame_filtered = TotalFrame[TotalFrame['CellID'].map(TotalFrame['CellID'].value_counts()) >= 10]



##### Cell Mean stats of interest COLORED BY Stdev ########
soi = ['Cell_Volume','Cell_SurfaceArea','Cell_Aspect_Ratio','speed','directional_autocorrelation']
ylabels = ['Cell Volume (µm$^3$)','Cell Surface Area (µm$^2$)', 'Cell Aspect Ratio','Instantaneous Speed (µm/s)','Persistence']#, 'Turn Angle (°)']
scale = len(soi)
linewid = 2



fig, axes = plt.subplots(1,scale,figsize=(scale*4*0.45,4))
for i, ax in enumerate(axes.flatten()):
    stat = TotalFrame_filtered[['CellID', soi[i]]].groupby('CellID').mean()  

    sns.swarmplot(y = soi[i], data = stat, color = 'grey', size = 3.5, alpha = 0.5, ax = ax)
    sns.boxplot(y = soi[i], data = stat, color = 'white', 
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
    ax.set_xticks([])
    # Turn off all spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #set y limits
    ax.set_ylim(0, ax.get_ylim()[1])

#how many images unique cells (data points) are there
fig.text(0.5, 1, f'n = {len(TotalFrame_filtered.CellID.unique())} cells',
        va='center', ha='center', fontsize=14)

plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')