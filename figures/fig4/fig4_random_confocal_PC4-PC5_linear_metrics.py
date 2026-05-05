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
from pathlib import Path



#get directories and open separated datasets

treatments = ['Random']
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'
whichpcs = (4,5)
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



###### copy zero bin to 360 so that it wraps
zerobin = angframe[angframe[f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins']==0]
zerobin.loc[:,f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins'] = 360
angframe = pd.concat((angframe, zerobin))


### narrow down columns
collist = angframe.columns.tolist()
metlist = [x for x in collist if 'Axis_Vec_' in x]

modvecs = np.zeros((len(angframe),len(metlist)))
for ce, cell_evecs in enumerate(angframe[metlist].values):
    cell_evecs = cell_evecs.copy()
    #### always point the median vector up and enforce right handed-ness
    if cell_evecs[0]<0:
        cell_evecs[:3] *= -1
    if cell_evecs[3+2]<0:
        cell_evecs[3:3+3] *= -1
        
    righth = np.cross(cell_evecs[:3], cell_evecs[3:3+3])
    if (cell_evecs[-1] * righth[-1]) < 0:
        cell_evecs[-3:] *= -1
        
    modvecs[ce] = cell_evecs
    
    
metlist_sign = [x+'_sign' for x in metlist]
for m, mm in enumerate(metlist_sign):
    angframe[mm] = modvecs[:,m]
    
##### get interpolated average line to plot continuous colormap along line
avgmets = angframe.groupby(f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins')[metlist_sign].mean()



#points to interpolate between each average point
pperp = 250
totalpoints = len(avgmets)*pperp

#define the colors to make the meshes
cmap = cm.get_cmap('twilight')
discrete_colors = cmap(np.linspace(0,1,totalpoints))


sq = math.ceil(np.sqrt(len(metlist_sign)))

fig, axes = plt.subplots(sq, sq, figsize=(18,14), sharex = True, sharey = True)

axis_order = ['Major','Median','Minor']
coord_order = ['X','Y','Z']

for row, axe in enumerate(axes):
    for col, ax in enumerate(axe):
        ### define axis and coordinate
        axis = axis_order[col]
        coord = coord_order[row]
        metname = f'Cell_{axis}Axis_Vec_{coord}_sign'

        
        cis = angframe.groupby(f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins')[metname].agg(utils.bs_ci)
        
        #plot lower 95% CI
        ax.plot(cis.index.to_list(),[x[0] for x in cis], ls = 'dotted', lw=1, color = '0.5')
        #plot upper 95% CI
        ax.plot(cis.index.to_list(),[x[1] for x in cis], ls = 'dotted', lw=1, color = '0.5')

        
        #interpolate between the points of the average line to plot continuous
        #colormap
        tck = interpolate.splrep(avgmets.index.values,avgmets[metname].values, k=1, s=0)
        tr = np.linspace(0,np.arange(0,380,20)[-1], totalpoints)
        x = interpolate.splev(tr,tck)


        #plot interpolated mean line
        ax.scatter(tr, x, s = 5, color = discrete_colors[:,:-1], edgecolor = None, zorder=2)
        
        
        #make the line at the middle of the cycle
        ax.axvline(180,color = 'black', linestyle = '--',linewidth=1, alpha = 0.3, zorder=3)


        ### only label left and top
        if row == 0:
            ax.set_title(axis + ' Axis', fontsize = 20)
        if col == 0:
            ax.set_ylabel(coord + ' Component', fontsize = 20)
        if row == 2 and col == 1:
            ax.set_xlabel('Angular Bins (°)', fontsize = 28)
        
        #change x ticks to every 60 and change tick label size
        ax.set_xticks(np.arange(0,420,60))
        ax.tick_params('y',labelsize=12)
        ax.tick_params('x',labelsize=12)
        
        #change y axis labels and sizes
        # ax.set_ylabel(metlist[i], fontsize = 18)
        # ax.set_xlabel('')
        
        
        #remove right and top spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend_ = None
    
    
# fig.text(0.5,0.001,'Angular Bins (°)', fontsize = 28)
    
plt.tight_layout()

# plt.show()


plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)










################## 2D YZ PROJECTION OF VECTORS THROUGH CYCLE

#points to interpolate between each average point
pperp = 250
totalpoints = len(avgmets)*pperp

#define the colors to make the meshes
cmap = cm.get_cmap('twilight')
discrete_colors = cmap(np.linspace(0,1,totalpoints))


axis_order = ['Major','Median','Minor']


fig, axes = plt.subplots(1, 3, figsize=(10,4), sharey = True)
for i, ax in enumerate(axes):

    ##### Y axis stuff    
    ystr = f'Cell_{axis_order[i]}Axis_Vec_Y_sign'
    ycis = angframe.groupby(f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins')[ystr].agg(utils.bs_ci)
    ylow, yhigh = zip(*ycis.tolist())
    yvals = avgmets[ystr].values
    
    zstr = f'Cell_{axis_order[i]}Axis_Vec_Z_sign'
    zcis = angframe.groupby(f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins')[zstr].agg(utils.bs_ci)
    zlow, zhigh = zip(*zcis.tolist())
    zvals = avgmets[zstr].values    
    
    
    #interpolate between the points of the average line to plot continuous
    #colormap
    tck, u = interpolate.splprep(np.vstack([avgmets.index.values,yvals,zvals]), k=1, s=0)
    tr = np.linspace(0,1, totalpoints)
    t,y,z = interpolate.splev(tr,tck)


    #plot interpolated mean line
    ax.scatter(y, z, s = 5, color = discrete_colors[:,:-1], edgecolor = None, zorder=2)
    
    
    #plot lower 95% CI
    ax.plot(ylow,zlow, ls = 'dotted', lw=1, color = '0.5')
    #plot upper 95% CI
    ax.plot(yhigh,zhigh, ls = 'dotted', lw=1, color = '0.5')


    ax.axvline(0,color = 'black', linestyle = '--',linewidth=1, alpha = 0.3, zorder=3)
    ax.axhline(0,color = 'black', linestyle = '--',linewidth=1, alpha = 0.3, zorder=3)

    ax.set_xlim(-0.7,0.7)
    ax.set_ylim(-0.7,0.7)    
    

    ### labels and titles
    ax.set_title(axis_order[i] + ' Axis', fontsize = 16)
    if i == 0:
        ax.set_ylabel('Z', fontsize = 16)
    if i == 1:
        ax.set_xlabel('Y', fontsize = 20)
    
    
plt.tight_layout()



plt.savefig(__file__.split('.')[0]+'_projection.png', bbox_inches='tight', dpi = 500)



