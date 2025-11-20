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
origin = [9,9]
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

# ### add protrusion and retraction speeds
# prsplist = []
# for i, cells in angframe.groupby('CellID'):
#     cells, runs = utils.get_consecutive_timepoints(cells, 'time', time_interval)
#     for r in runs:
#         tempcell = cells.iloc[r]
#         tempcell.loc[:,'protrusion_speed'] = tempcell.LengthAlongTrajectoryFront.diff()
#         tempcell.loc[:,'retraction_speed'] = tempcell.LengthAlongTrajectoryRear.diff()
#         prsplist.append(tempcell)
# angframe = pd.concat(prsplist).reset_index(drop=True)

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
        ax.set_ylim(8.601372543038753, 13.85642266037562)
    
    
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




fig, ax = plt.subplots()
ax.text(0.5,0.5, 'Angular Bin (°)', fontsize = 24)
ax.axis('off')
plt.savefig(__file__.split('.')[0] + '_label.png', dpi = 500, bbox_inches='tight')

     

# ###### make a custom legend
# from matplotlib.collections import LineCollection
# from matplotlib.legend_handler import HandlerLineCollection

# class HandlerColorLineCollection(HandlerLineCollection):
#     def create_artists(self, legend, artist ,xdescent, ydescent,
#                         width, height, fontsize,trans):
#         x = np.linspace(0,width,self.get_numpoints(legend)+1)
#         y = np.zeros(self.get_numpoints(legend)+1)+height/2.-ydescent
#         points = np.array([x, y]).T.reshape(-1, 1, 2)
#         segments = np.concatenate([points[:-1], points[1:]], axis=1)
#         lc = LineCollection(segments, cmap=artist.cmap,
#                      transform=trans)
#         lc.set_array(x)
#         lc.set_linewidth(artist.get_linewidth())
#         return [lc]

# fig, ax = plt.subplots()

# # Make a simple multicolored line for the legend
# legend_line = LineCollection(
#     [np.array([[0, 0], [1, 0]])],
#     colors=plt.cm.twilight(np.linspace(0, 1, 5)),
#     linewidth=2
# )
# ax.add_collection(legend_line)
# # Add a dummy legend handle
# ax.legend(
#     [legend_line],
#     ['Multicolored Line'],
#     handler_map={LineCollection: HandlerColorLineCollection(numpoints = 5)},
#     loc='upper right'
# )



# #adjust plot limits
# ax.set_xlim(0.6,1.0)
# ax.set_ylim(0.6,1.0)



# ax.axis('off')

# plt.show()
