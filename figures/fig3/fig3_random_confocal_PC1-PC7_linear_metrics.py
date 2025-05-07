# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:35:03 2025

@author: Aaron
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from CustomFunctions import linear_cycle_utils, utils
import math
from scipy import interpolate
from matplotlib import cm
from scipy.stats import t

#get directories and open separated datasets


treatments = ['Random']
time_interval = 10 #sec/frame
whichpcs = [1,7]
origin = [7,7]
binrange = 20
direction = 'clockwise'
zerostart = 'left'


#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'
if not os.path.exists(savedir):
    os.makedirs(savedir)
    
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())
#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
TotalFrame = FullFrame[FullFrame.Treatment=='Random']


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

#change rear length along trajectory to positive values
angframe['LengthAlongTrajectoryRear'] = abs(angframe['LengthAlongTrajectoryRear'])


metlist = ['Cell_Aspect_Ratio','LengthAlongTrajectory','LengthAlongTrajectoryFront','LengthAlongTrajectoryRear',
           'Volume_Front_Ratio','Cell_Volume','speed', 'Turn_Angle']

labelz = ['Aspect Ratio','Length Along\nTrajectory (µm)','Forward Length Along\nTrajectory (µm)','Rearward Length Along\nTrajectory (µm)',
          'Front-Back\nVolume Ratio','Cell Volume (µm$^3$)','Speed (µm/sec)','Turn Angle (°)']

##### get interpolated average line to plot continuous colormap along line
avgmets = angframe.groupby(f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins').mean()[metlist]


# CoRo = math.ceil(math.sqrt(len(metlist)))
# row = 0
# fig, axes = plt.subplots(CoRo, CoRo, figsize=(4*CoRo,3*CoRo))#, sharex=True)

#points to interpolate between each average point
pperp = 250
totalpoints = len(avgmets)*pperp

#define the colors to make the meshes
cmap = cm.get_cmap('twilight')
discrete_colors = cmap(np.linspace(0,1,totalpoints))


fig, axes = plt.subplots(2, int(len(metlist)/2), figsize=(2*len(metlist),3*2))
for i, ax in enumerate(axes.flatten()):
 
    sns.lineplot(data = angframe, x=f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins', y = metlist[i],
                 lw = 2, color = [0.4,0.4,0.4], ci=95, ax = ax, zorder=1)
    
    #interpolate between the points of the average line to plot continuous
    #colormap
    tck = interpolate.splrep(avgmets.index.values,avgmets[metlist[i]].values, k=1, s=0)
    tr = np.linspace(0,np.arange(0,360,20)[-1], totalpoints)
    x = interpolate.splev(tr,tck)

    # #interpolate confidence interval
    # lower, upper = t.interval(0.05, avgmets[metlist[i]].values)
    # lower = lower+avgmets[metlist[i]].values
    # upper = upper+avgmets[metlist[i]].values
    # ltck = interpolate.splrep(avgmets.index.values,lower, k=1, s=0)
    # utck = interpolate.splrep(avgmets.index.values,upper, k=1, s=0)
    # lx = interpolate.splev(tr,ltck)
    # ux = interpolate.splev(tr,utck)
    # for c in range(totalpoints):
    #     ax.plot([tr[c],tr[c]],[lx[c],ux[c]], color = discrete_colors[c,:-1], alpha = 0.05, zorder = 2)
    
    ax.scatter(tr, x, s = 2.5, color = discrete_colors[:,:-1], edgecolor = None, zorder=2)

    ax.axvline(180,color = 'black', linestyle = '--',linewidth=1, alpha = 0.3, zorder=3)
    ax.set_ylabel(metlist[i])#, fontsize = 1.75*CoRo)
    ax.legend_ = None
    ax.set_ylabel(labelz[i], fontsize = 18)
    ax.tick_params('y',labelsize=12)
    ax.set_xlabel('')

plt.tight_layout()




plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)