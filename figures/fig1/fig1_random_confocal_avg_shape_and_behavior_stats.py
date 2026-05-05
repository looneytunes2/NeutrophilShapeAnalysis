# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:42:54 2025

@author: Aaron
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from neutrophil_shape.config.loader import load_config



treatments = ['Random']


#get directories and open separated datasets
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'
datadir = config.common.savedir / 'shape_data'

### open all data
FullFrame = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col=0)
#restrict dataframe to only random experiments
TotalFrame = FullFrame[FullFrame.Treatment.isin(treatments)]
#limit the dataframe to only tracks with at least 10 frames
TotalFrame_filtered = TotalFrame[TotalFrame['CellID'].map(TotalFrame['CellID'].value_counts()) >= 10]



##### Cell Mean stats of interest COLORED BY Stdev ########
soi = ['Cell_Volume','Cell_Aspect_Ratio','speed']
ylabels = ['Cell Volume (µm$^3$)', 'Cell Aspect Ratio','Instantaneous Speed (µm/s)']#, 'Turn Angle (°)']
scale = len(soi)
linewid = 2



fig, axes = plt.subplots(1,scale,figsize=(scale*4*0.45,4))
for i, ax in enumerate(axes.flatten()):
    stat = TotalFrame_filtered[['CellID', soi[i]]].groupby('CellID').mean()  
    sns.swarmplot(y = soi[i], data = stat, color = 'grey', size = 3.5, alpha = 0.6, ax = ax, zorder = 1)
    
    #put the IQR box
    sns.boxplot(y = soi[i], data = stat, width = 0.5,
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
    
    
    #average of averages
    avgavg = TotalFrame.groupby('CellID')[soi[i]].mean().mean()
    avgavgline = ax.plot([-0.35,0.35],[avgavg,avgavg], lw = linewid, color = '#1581b0', label = 'Average Track Mean', zorder = 3)
    
    #draw the average line
    allavg = TotalFrame[soi[i]].mean()
    allavgline = ax.plot([-0.35,0.35],[allavg,allavg], lw = linewid, color = '#44e3d6', label = 'Population Mean', zorder = 3)



    #tick stuff
    ax.set_ylabel(ylabels[i], fontsize = 16)#, labelpad=-0.5)
    ax.set_xlabel('')
    ax.set_xticks([])
    # Turn off all spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #set y limits
    ax.set_ylim(0, ax.get_ylim()[1])

# #how many images unique cells (data points) are there
# fig.text(0.5, 0.91, f'n = {len(TotalFrame_filtered.CellID.unique())} cells',
#         va='center', ha='center', fontsize=10)
    # ax.get_legend_handles_labels()

#fig title
fig.suptitle('Cell Track Means', y = 1.04, fontsize=16)

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