# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 10:18:37 2025

@author: Aaron
"""


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from CustomFunctions import linear_cycle_utils, utils
import math



#get directories and open separated datasets


treatments = ['Random']
time_interval = 5 #sec/frame
whichpcs = [1,7]
origin = [7, 6]
binrange = 20
direction = 'clockwise'
zerostart = 'left'


#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_Apply/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'
if not os.path.exists(savedir):
    os.makedirs(savedir)
    
    
FullFrame = pd.read_csv(datadir + 'Shape_Metrics_transitionPCbins.csv', index_col=0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())
#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
TotalFrame = FullFrame[FullFrame.Treatment=='Random']
TotalFrame.rename(columns={'Cell_Elongation': 'Aspect_Ratio'}, inplace = True)

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

### add protrusion and retraction speeds
prsplist = []
for i, cells in angframe.groupby('CellID'):
    cells, runs = utils.get_consecutive_timepoints(cells, 'time', time_interval)
    for r in runs:
        tempcell = cells.iloc[r]
        tempcell['protrusion_speed'] = tempcell.LengthAlongTrajectoryFront.diff()
        tempcell['retraction_speed'] = tempcell.LengthAlongTrajectoryRear.diff()
        prsplist.append(tempcell)
angframe = pd.concat(prsplist).reset_index(drop=True)


metlist = ['Cell_Aspect_Ratio','LengthAlongTrajectory','Volume_Front_Ratio','Cell_Volume',
           'speed', 'directional_autocorrelation','LengthAlongTrajectoryFront','LengthAlongTrajectoryRear']

labelz = ['Aspect Ratio','Length Along Trajectory (µm)','Front-Back Volume Ratio','Cell Volume (µm$^3$)',
          'Speed (µm/sec)','Directional Autocorrelation','Forward Length Along\nTrajectory (µm/sec)','Rearward Length Along\nTrajectory (µm/sec)']

CoRo = math.ceil(math.sqrt(len(metlist)))
row = 0
fig, axes = plt.subplots(CoRo, CoRo, figsize=(4*CoRo,3*CoRo))#, sharex=True)

for i, ax in enumerate(axes.flatten()):
    if i<len(metlist):

        sns.lineplot(data = angframe, x=f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins', y = metlist[i],
                     lw = 2, color = '#59bd80', ax = ax)
        ax.axvline(180,color = 'black', linestyle = '--',linewidth=1, alpha = 0.3)
        ax.set_ylabel(metlist[i])#, fontsize = 1.75*CoRo)
        ax.legend_ = None
        ax.set_ylabel(labelz[i], fontsize = 18)
        ax.set_xlabel('')
    else:
        ax.remove()


plt.tight_layout()
plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)