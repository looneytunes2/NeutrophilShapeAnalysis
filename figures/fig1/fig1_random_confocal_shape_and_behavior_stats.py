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
TotalFrame = FullFrame[FullFrame.Treatment.isin(treatments)]


##### Cell Mean stats of interest COLORED BY Stdev ########
soi = ['Cell_Volume','Cell_Aspect_Ratio','speed',]
ylabels = ['Cell Volume (µm$^3$)','Cell Aspect Ratio','Instantaneous Speed (µm/s)']#, 'Turn Angle (°)']
scale = len(soi)
linewid = 2



fig, axes = plt.subplots(1,scale,figsize=(scale*4*0.45,4))
for i, ax in enumerate(axes.flatten()):
    sns.violinplot(y=TotalFrame[soi[i]], color = '0.65',
                   linewidth = 0, inner = None, ax=ax, )
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
                    'linewidth': 0,
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
    
    #go all the way to zero
    ax.set_ylim(min(0, ax.get_ylim()[0]), ax.get_ylim()[1])
    
#how many images data points are there
imnum = fig.text(1.28/len(soi), 1, f'n = {len(TotalFrame)} images',
        va='center', ha='center', fontsize=14)
imnum_xpos = imnum.get_position()[0]
imline = mlines.Line2D([imnum_xpos-0.18, imnum_xpos+0.18], [0.98, 0.98], color='black', lw=1,)
fig.add_artist(imline)


#how many images data points are there
intnum = fig.text(2.65/len(soi), 1, f'n = {len(TotalFrame.speed.dropna())} intervals',
        va='center', ha='center', fontsize=14)
intnum_xpos = intnum.get_position()[0]
intline = mlines.Line2D([intnum_xpos-0.1, intnum_xpos+0.1], [0.98, 0.98], color='black', lw=1,)

fig.add_artist(intline)


plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')