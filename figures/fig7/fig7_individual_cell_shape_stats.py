# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 14:18:33 2025

@author: Aaron
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from pathlib import Path


scale = 4
basedir = Path('E:/Aaron/Combined_37C_Confocal_PCA_planar_LLS_Apply')
datadir = basedir.joinpath('Data_and_Figs')


FullFrame = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col = 0)


metrics = ['Cell_Volume','Cell_Aspect_Ratio', 'speed','directional_autocorrelation']
labelz = ['Cell Volume (µm$^3$)','Cell Aspect Ratio','Instantaneous Speed (µm/s)','Persistence']



#define the colors to make the meshes
set1 = plt.cm.Set1
set2 = plt.cm.Set2
set3 = plt.cm.Set3
cmap = matplotlib.colors.ListedColormap(list(set3.colors)+[set2.colors[-2]] + [set1.colors[-1]])


fig, axes = plt.subplots(2,2, figsize=(scale*len(metrics)/2, scale*2))

flierprops = dict(marker='.', markersize=1.5)
linewid = 1
for i, ax in enumerate(axes.flatten()):
    sns.boxplot(data = FullFrame, x = 'CellID', y = metrics[i], palette = cmap.colors,
                showcaps=False,
                boxprops={
                    'zorder': 2
                    },
                whiskerprops={
                    'linewidth': 0,
                    },
                flierprops={
                    'marker': '',
                    },
                ax=ax)
    
    
    
    sns.stripplot(data = FullFrame, x = 'CellID', y = metrics[i], jitter = False, color = 'gray',
                  s = 1.2, alpha = 0.7, zorder = 1, ax = ax)
    
    ### set the y limit to zero
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax)
    
    ax.legend_ = None
    ax.set_xlabel('')
    ax.set_xticks([])
    ax.set_xticklabels('')
    ax.set_ylabel(labelz[i], fontsize = 20)
    
    #plot the median of medians
    avgs = FullFrame[['CellID',metrics[i]]].groupby('CellID').median()
    avgavg = avgs[metrics[i]].median()
    ax.axhline(avgavg, ls = '--', color = 'black', alpha = 0.4)
    
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')

