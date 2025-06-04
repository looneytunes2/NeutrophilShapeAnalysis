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
    sns.violinplot(y=TotalFrame[soi[i]], color = '0.65',
                   linewidth = linewid, inner = None, ax=ax, )
    ax.collections[0].set_edgecolor('black')
    sns.boxplot(y=TotalFrame[soi[i]], width = 0.15, color = 'white', 
                showcaps=False, showfliers=False,
                boxprops={
                    'fill': 'white',
                    'linewidth': linewid,
                    'edgecolor': 'black',
                    'zorder': 2
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
                ax=ax)
    #tick label size
    ax.tick_params('y', labelsize=12)
    #remove legends
    ax.legend_ = None
    #adjust ylabel
    ax.set_ylabel(ylabels[i], fontsize=16)
    #set plot limits
    # ax.set_ylim(0,13)
    #remove parts of box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #remove x tick
    ax.set_xticks([])

#how many images data points are there
fig.text(1.55/5, 1, f'n = {len(TotalFrame)} images',
        va='center', ha='center', fontsize=14)

#how many images data points are there
fig.text(4.1/5, 1, f'n = {len(TotalFrame.speed.dropna())} intervals',
        va='center', ha='center', fontsize=14)



plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')