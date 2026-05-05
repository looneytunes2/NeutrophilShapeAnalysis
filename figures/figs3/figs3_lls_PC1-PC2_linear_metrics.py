# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:35:03 2025

@author: Aaron
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from neutrophil_shape.CustomFunctions import linear_cycle_utils, utils
from neutrophil_shape.config.loader import load_config
import math
from scipy import interpolate
from matplotlib import cm



#get directories and open separated datasets

treatments = ['Random']
config = load_config(microscope_type='lls')
config._alignment = 'trajectory'
whichpcs = (1,2)
pc_combos = config.common.pc_combos
origin = config.db_params.origins[pc_combos.index(whichpcs)]
binrange = 20
direction = 'clockwise'
zerostart = 'left'


#get directories and open separated datasets
basedir = config.common.savedir
datadir = basedir.joinpath('shape_data')
savedir = basedir.joinpath('detailed_balance')

    
FullFrame = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col=0)
#open the centers of the binned PCs
centers = pd.read_csv(datadir.joinpath('PC_bin_centers.csv'), index_col=0)
TotalFrame = FullFrame[FullFrame.Treatment.isin(treatments)].copy()


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




#change rear length along trajectory to positive values
angframe.loc[:,'LengthAlongTrajectoryRear'] = abs(angframe['LengthAlongTrajectoryRear'].copy())


###### copy zero bin to 360 so that it wraps
zerobin = angframe[angframe[f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins']==0]
zerobin.loc[:,f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins'] = 360
angframe = pd.concat((angframe, zerobin))



#### narrow down columns
# collist = angframe.columns.tolist()
# snippets = ['shcoeffs','Trajectory_','crop','Date','Experiment',
#             'Treatment','Axis','_X','_Y','_Z','bins',
#             'cell','image','structure','frame','time','CellID','Cell_intensity']
# metlist = [x for x in collist if not any([y in x for y in snippets])]

metlist = ['Cell_MajorAxis_Vec_X','Cell_MajorAxis_Vec_Y','directional_autocorrelation','speed']

labelz = ['Major Axis\nX Component','Major Axis\nY Component','Persistence','Instantaneous\nSpeed (µm/sec)']



##### get interpolated average line to plot continuous colormap along line
avgmets = angframe.groupby(f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins')[metlist].mean()



#points to interpolate between each average point
pperp = 250
totalpoints = len(avgmets)*pperp

#define the colors to make the meshes
cmap = cm.get_cmap('twilight')
discrete_colors = cmap(np.linspace(0,1,totalpoints))


sq = math.ceil(np.sqrt(len(metlist)))

fig, axes = plt.subplots(1, 4, figsize=(16,3), sharex = True)
for i, ax in enumerate(axes.flatten()):
    if i>=len(metlist):
        ax.remove()
        continue

    
    cis = angframe.groupby(f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins')[metlist[i]].agg(utils.bs_ci)
    
    #plot lower 95% CI
    ax.plot(cis.index.to_list(),[x[0] for x in cis], ls = 'dotted', lw=1, color = '0.5')
    #plot upper 95% CI
    ax.plot(cis.index.to_list(),[x[1] for x in cis], ls = 'dotted', lw=1, color = '0.5')

        
    
    #interpolate between the points of the average line to plot continuous
    #colormap
    tck = interpolate.splrep(avgmets.index.values,avgmets[metlist[i]].values, k=1, s=0)
    tr = np.linspace(0,np.arange(0,380,20)[-1], totalpoints)
    x = interpolate.splev(tr,tck)


    #plot interpolated mean line
    ax.scatter(tr, x, s = 5, color = discrete_colors[:,:-1], edgecolor = None, zorder=2)
    
    
    #make the line at the middle of the cycle
    ax.axvline(180,color = 'black', linestyle = '--',linewidth=1, alpha = 0.3, zorder=3)
    ax.set_ylabel(metlist[i])#, fontsize = 1.75*CoRo)
    
    
    
    #change x ticks to every 60 and change tick label size
    ax.set_xticks(np.arange(0,420,60))
    ax.tick_params('y',labelsize=12)
    ax.tick_params('x',labelsize=12)
    
    #change y axis labels and sizes
    ax.set_ylabel(labelz[i], fontsize = 18)
    ax.set_xlabel('')
    
    
    #remove right and top spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend_ = None
    
    
# fig.text(0.5,0.01,'Angular Bins ()', fontsize = 18)
    
plt.tight_layout()


plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)


