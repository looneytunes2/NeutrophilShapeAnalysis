# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 11:33:21 2025

@author: Aaron
"""

import pandas as pd
import numpy as np
from CustomFunctions import linear_cycle_utils
import matplotlib.pyplot as plt
import seaborn as sns

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


origin = [7, 7]
whichpcs = [1,7]
binrange = 20
direction = 'clockwise'
zerostart = 'left'

angframe = linear_cycle_utils.linearize_cycle_continuous(
            TotalFrame, 
            centers,
            origin, 
            whichpcs,
            zerostart,
            direction,)

angframe =  linear_cycle_utils.bin_angular_coord(
        angframe,
        whichpcs,
        binrange,
        )


#get colors based on linear CGPS radial graphic
colorlist = ['#4085e3','#d93434']
sns.set_palette(palette=colorlist)

fig, ax = plt.subplots(1, 1, figsize=(5,5))#, sharex=True)

sns.histplot(data = angframe, x=f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Coord',
              bins = np.sort(angframe[f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins'].unique()),
              hue = 'Treatment',
             lw = 2, ax = ax)
# #change bar color based on bin #
# for i, p in enumerate(ax.patches):
#     p.set_facecolor(discrete_colors[i,:])

#axis stuff
# ax.legend_ = None
ax.set_ylabel('Image Count', fontsize = 22)
ax.set_xlabel('Angular Bin (°)', fontsize = 22)
ax.tick_params('x',labelsize = 12)
ax.tick_params('y',labelsize = 12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)
