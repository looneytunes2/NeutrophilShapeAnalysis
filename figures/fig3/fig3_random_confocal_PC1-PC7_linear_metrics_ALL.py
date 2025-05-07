# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:56:20 2025

@author: Aaron
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from CustomFunctions import linear_cycle_utils
import math



#get directories and open separated datasets


treatments = ['Random']
time_interval = 10 #sec/frame
whichpcs = [1,7]
origin = [7, 7]
binrange = 20
direction = 'clockwise'
zerostart = 'left'


#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_smooth/'
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


########### remove useless columns
removeex = []
removeex.extend([x for x in angframe.columns if 'avg' in x])
removeex.extend([x for x in angframe.columns if 'PC1_PC2' in x])
removeex.extend([x for x in angframe.columns if 'intensity' in x])
removeex.extend([x for x in angframe.columns if 'Velocity' in x])
removeex.extend([x for x in angframe.columns if '_Volume_' in x])
removeex.extend([x for x in angframe.columns if 'raw' in x])
removeex.extend([x for x in angframe.columns if '_X' in x])
removeex.extend([x for x in angframe.columns if '_Y' in x])
removeex.extend([x for x in angframe.columns if '_Z' in x])
removeex.extend([x for x in angframe.columns if 'Coord' in x])
removeex.extend([x for x in angframe.columns if 'Centroid' in x])
removeex.extend([x for x in angframe.columns if 'Error' in x])
removeex.extend([x for x in angframe.columns if 'Vec' in x])
removeex.extend([x for x in angframe.columns if 'bins' in x])
removeex.extend(['cell','CellID','x','y','z','structure','frame','Treatment','Experiment',
                 'Date','Width_Rotation_Angle','directional_autocorrelation','activity','time'])
newframe = angframe.drop(columns =removeex)
newframe.columns.to_list()




# metlist = ['Aspect_Ratio','Cell_TotalAngle','Volume_Front_Ratio','Cell_Volume', 'speed', 'persistence']

# labelz = ['Aspect_Ratio','Deviation from\nTrajectory (°)','Front-Back Volume\nRatio (a.u.)',
#           'Cell Volume (µm$^3$)','Speed (µm/sec)','Persistence (a.u.)']
CoRo = math.ceil(math.sqrt(len(newframe.columns)))
row = 0
fig, axes = plt.subplots(CoRo, CoRo, figsize=(3.5*CoRo,3*CoRo))#, sharex=True)
# colorlist = ['#3799de','#fc5858']
for i, ax in enumerate(axes.flatten()):
    if i<len(newframe.columns):
        if newframe.iloc[:,i].name == f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins':
            ax.remove()
            continue
        sns.lineplot(data = TotalFrame, x=f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins', 
                     y = newframe.iloc[:,i].name, color = '#59bd80', #palette = colorlist, 
                     ax = ax)
        ax.set_ylabel(newframe.iloc[:,i].name)#, fontsize = 1.75*CoRo)
        ax.legend_ = None
        ax.set_ylabel(ax.get_ylabel(), fontsize = 22)
       
    else:
        ax.remove()


plt.tight_layout()
plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)