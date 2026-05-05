# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 11:33:21 2025

@author: Aaron
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from neutrophil_shape.CustomFunctions import linear_cycle_utils
from matplotlib import cm
from pathlib import Path
from neutrophil_shape.config.loader import load_config

#get directories and open separated datasets



treatments = ['Random']
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'
time_interval = config.im_params.time_interval #sec/frame
whichpcs = (1,2)
allorigins = config.db_params.origins
pc_combos = config.common.pc_combos
origin = allorigins[pc_combos.index(whichpcs)]
binrange = 360/18 # degrees / bin
direction = 'clockwise' #direction of flux
zerostart = 'left' #what 2D direction to call zero


#get directories and open separated datasets
basedir = config.common.savedir
datadir = basedir / 'shape_data'
    
FullFrame = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col=0)
##restrict to random treatment
TotalFrame = FullFrame[FullFrame.Treatment=='Random'].reset_index(drop = True)

#open the centers of the binned PCs
centers = pd.read_csv(datadir.joinpath('PC_bin_centers.csv'), index_col=0)



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

histdf = angframe[f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins'].value_counts().sort_index().reset_index()

#get colors based on linear CGPS radial graphic
cmap = cm.get_cmap('twilight', int(360/binrange+1))
discrete_colors = cmap(np.linspace(0,1,int(360/binrange+1))[:-1])

fig, ax = plt.subplots(1, 1, figsize=(5,5))

sns.barplot(data = histdf, x = f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins', y='count',
            edgecolor = 'black', lw = 2, ax = ax)

## change bar color based on bin 
for i, p in enumerate(ax.patches):
    p.set_facecolor(discrete_colors[i,:])
    p.set_width(1.0)
    

#axis stuff
ax.legend_ = None
ax.set_ylabel('Image Count', fontsize = 22)
ax.set_xlabel('Angular Bin (°)', fontsize = 22)
ax.set_xlim(-1,18)
ax.set_xticks(np.arange(0,histdf.shape[0],60/binrange)+0.1)
ax.set_xticklabels(np.arange(0,360,60))
ax.tick_params('x',labelsize = 12)
ax.tick_params('y',labelsize = 12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()



plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)
