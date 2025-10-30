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
from scipy import interpolate
from matplotlib import cm
from scipy.stats import t

#get directories and open separated datasets


treatments = ['Random']
whichpcs = [1,7]
origin = [12,11]
binrange = 20
direction = 'clockwise'
zerostart = 'left'


#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_apply/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'
if not os.path.exists(savedir):
    os.makedirs(savedir)
    
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
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



#change rear length along trajectory to positive values
angframe.loc[:,'LengthAlongTrajectoryRear'] = abs(angframe['LengthAlongTrajectoryRear'].copy())

###### copy zero bin to 360 so that it wraps
zerobin = angframe[angframe[f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins']==0]
zerobin.loc[:,f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins'] = 360
angframe = pd.concat((angframe, zerobin))




metlist = ['Cell_Aspect_Ratio','Volume_Front_Ratio','directional_autocorrelation','speed',
           'LengthAlongTrajectory','LengthAlongTrajectoryFront','LengthAlongTrajectoryRear','Cell_Volume']

labelz = ['Aspect Ratio','Front-Rear\nVolume Ratio','Persistence','Instantaneous\nSpeed (µm/sec)',
          'Length Along\nTrajectory (µm)','Forward Length Along\nTrajectory (µm)','Rearward Length Along\nTrajectory (µm)','Cell Volume (µm$^3$)']

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



fig, axes = plt.subplots(2, int(len(metlist)/2), figsize=(2*len(metlist),3*2), sharex = True)
for i, ax in enumerate(axes.flatten()):
 
    # sns.lineplot(data = angframe, x=f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins', y = metlist[i],
    #               lw = 2, color = [0.4,0.4,0.4], ci=95, ax = ax, zorder=1)
    
    
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
    
    
    #plot interpolated mean line
    ax.scatter(tr, x, s = 5, color = discrete_colors[:,:-1], edgecolor = None, zorder=2)
    
    
    #make the line at the middle of the cycle
    ax.axvline(180,color = 'black', linestyle = '--',linewidth=1, alpha = 0.3, zorder=3)
    ax.set_ylabel(metlist[i])#, fontsize = 1.75*CoRo)
    
    
    #set the y limits of the front and rear lengths equally
    if metlist[i] == 'LengthAlongTrajectoryFront' or metlist[i] == 'LengthAlongTrajectoryRear':
        ax.set_ylim(8.292890502364488, 13.866759448213358)
    
    
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
    
plt.tight_layout()



plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)

