# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:42:54 2025

@author: Aaron
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines


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
TotalFrame = FullFrame[FullFrame.Treatment.isin(treatments)].copy()
#limit the dataframe to only tracks with at least 25 frames
TotalFrame_filtered = TotalFrame[TotalFrame['CellID'].map(TotalFrame['CellID'].value_counts()) >= 25].copy()



##### Define metrics ########
soi = ['Cell_Volume','Cell_Aspect_Ratio','speed']
ylabels = ['Cell Volume','Cell Aspect Ratio','Instantaneous Speed',]
scale = len(soi)
linewid = 2



fig, axes = plt.subplots(1,scale,figsize=(scale*4*0.45,4))
for i, ax in enumerate(axes.flatten()):
    stat = TotalFrame_filtered[['CellID', soi[i]]].groupby('CellID').std()  
    #individual cells
    sns.swarmplot(y = soi[i], data = stat, color = 'grey', size = 3.5, alpha = 0.6, ax = ax, zorder = 1)
    
    #put the IQR box
    sns.boxplot(y = soi[i], data = stat, width = 0.3,
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
    
    #average of the cell stds
    allindstd = TotalFrame_filtered.groupby('CellID')[soi[i]].std().mean()
    avgstdsline = ax.plot([-0.25,0.25],[allindstd, allindstd], lw = 2, color = '#821065',
                          label = 'Average Track StDev', zorder = 3)
    
    #population std
    allstd = TotalFrame_filtered[soi[i]].std()
    allstdline = ax.plot([-0.25,0.25],[allstd, allstd], lw = 2, color = 'magenta',
                         label = 'Population StDev', zorder = 3)

    #tick stuff
    ax.set_ylabel(ylabels[i], fontsize = 16)#, labelpad=-0.5)
    ax.set_xlabel('')
    ax.set_xticks([])
    # Turn off all spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #set y limits
    ax.set_ylim(0, ax.get_ylim()[1])

#fig title
fig.suptitle('Cell Track Standard Deviations', y = 1.04, fontsize=16)

### legend above plots
handles, labels  = ax.get_legend_handles_labels()
legend = fig.legend(handles,
           labels,
           loc='upper center',
           ncol = 2,
           bbox_to_anchor= (0.5,1.01),
           frameon = False)


#how many images data points are there
imnum = fig.text(0.5, 0.91, f'n = {TotalFrame_filtered.CellID.value_counts().shape[0]} cells',
        va='center', ha='center', fontsize=10)



plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')