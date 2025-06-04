# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 16:02:23 2025

@author: Aaron
"""

import multiprocessing
import os
from CustomFunctions import file_management, utils, DetailedBalance
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection


def running_mean_withna(x, N):
    means = []
    for i, r in enumerate(x):
        if np.isnan(r):
            means.append(np.nan)
        elif i<N:
            #get the window to average
            wind = x[:int(i+1)]
            #remove nan
            wind = wind[~np.isnan(wind)]
            #get average
            means.append(np.mean(wind))
        else:
            #get the indicies around the target value
            first = i - N//2+N%2
            second = first + N
            wind = x[first:second]
            #remove nan
            wind = wind[~np.isnan(wind)]
            #get average
            means.append(np.mean(wind))

    return np.array(means)





basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_Apply/'
infodir = basedir + 'processed_data/'
datadir = basedir + 'Data_and_Figs/'
cellid = '20240527_488_EGFP-CAAX_640_SPY650-DNA_cell2_01'
specificdir = basedir+'singlecells/'+cellid+'/'
savedir = basedir + 'random/'
time_interval = 5
whichpcs = [1,7]
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col = 0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())

#reduce the big dataframe to just the cell of interest
TotalFrame = FullFrame[FullFrame.CellID == cellid].copy()

##### quickly calculate aer and cf on the raw CGPS transitions
if os.path.exists(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_transitions_separated.csv'):
    raw_trans = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_transitions_separated.csv', index_col = 0)
    #add a movie columns to separate dataframe on
    raw_trans['Movie'] = [x.split('_frame')[0] for x in raw_trans.cell.to_list()]
    ############# measure aer and cycling frequencies ###########
    #add specific scaling
    xyscaling = [centers[f'PC{whichpcs[0]}'].diff().mean(),centers[f'PC{whichpcs[1]}'].diff().mean()]
    #set the origin to the actual center
    center = [round(nbins/2)]*2
    results = []
    cells = raw_trans[raw_trans.CellID == cellid].copy()
    movielist = sorted(cells.Movie.unique(),key = lambda x: int(x.split('-')[-3]))
    for m in movielist:
        curmov = cells[cells.Movie == m]
        curmov, runs = utils.get_consecutive_timepoints(curmov, 'frame',1)
        for r in runs:
            cell = curmov.iloc[r].reset_index(drop=True)
            results.append(DetailedBalance.get_area_enclosing_rate(
                cell,
                nbins,
                xyscaling,
                center,
                ))
    allaers = pd.concat(results).reset_index(drop=True)
# allaers['cell'] = [c+f'_frame_{int(f)}' for c, f in allaers[['Movie','frame']].values]
#merge aer and cf info
TotalFrame = pd.merge(TotalFrame, allaers[['aer','angular_velocity','cell']],on = 'cell',how='left')


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=60)
    filelist = utils.filename_match_llscellid(cellid, os.listdir(infodir))
    csvlist = [infodir + i for i in filelist]
    celllist = pool.map(file_management.multicsv, csvlist)
    pool.close()
    pool.join()
cell = pd.concat(celllist).reset_index(drop = True)

#get displacements and then cumulative position
cell['movie'] = [d.split('-Subset')[0] for d in cell.cell.to_list()]
### add cell time in minutes
cell['time_min'] = cell.time/60


######### get the projected speeds
cell = utils.project_raw_smooth(cell, time_interval)


###### merge the aer with the cell data
cell = cell.merge(TotalFrame[['cell','aer']], on = 'cell', how = 'left')


# Create a Normalize object
norm = Normalize(vmin=0, vmax=cell.time_min.max())
# Get the twilight colormap
cmap = cm.get_cmap('jet')




######### plot speed coded by time
# Build segments from x, y
points = cell[['time_min','speed']].values
segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
# Create and plot line collection
lc = LineCollection(segments, colors=cmap(norm(cell.time_min.values))[:,:3], linewidths=2)
fig, ax = plt.subplots()
ax.add_collection(lc)
ax.set_ylim(0, cell.speed.max())
ax.set_xlim(0, cell.time_min.max())

plt.tight_layout()


######### plot Turn angle coded by time
# Build segments from x, y
points = cell[['time_min','Turn_Angle']].values
segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
# Create and plot line collection
lc = LineCollection(segments, colors=cmap(norm(cell.time_min.values))[:,:3], linewidths=2)
fig, ax = plt.subplots()
ax.add_collection(lc)
ax.set_ylim(0, cell.Turn_Angle.max())
ax.set_xlim(0, cell.time_min.max())

plt.tight_layout()



########## plot aer coded by time
# Build segments from x, y
points = np.array([cell.time_min.values, cell.aer.cumsum().values]).T
segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
# Create and plot line collection
lc = LineCollection(segments, colors=cmap(norm(cell.time_min.values))[:,:3], linewidths=2)
fig, ax = plt.subplots()
ax.add_collection(lc)
ax.set_ylim(0, cell.aer.cumsum().max())
ax.set_xlim(0, cell.time_min.max())
ax.set_xlabel('Time min()')
plt.tight_layout()




######### smoothened speed smoothened turn angle and aer together
fig, axes = plt.subplots( 3, 1, figsize=(9,4), sharex=True)

smooth_speed = running_mean_withna(cell.speed.values, 5)
smooth_turn = running_mean_withna(cell.Turn_Angle.values, 5)
aercumsum = cell.aer.cumsum().values
labels = ['Instantaneous Speed\n(μm/sec)','Turn Angle (°)','Area Enclosed\n(PC units)']

for i, vals in enumerate([smooth_speed, smooth_turn, aercumsum]):
    ax = axes[i]
    # Build segments from x, y
    points = np.array([cell.time_min.values, vals]).T
    segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
    # Create and plot line collection
    lc = LineCollection(segments, colors=cmap(norm(cell.time_min.values))[:,:3], linewidths=2)
    ax.add_collection(lc)
    
    
    ax.set_ylim(0, np.nanmax(vals))
    ax.set_xlim(0, cell.time_min.max())
    ax.set_xlabel('Time (min)', fontsize = 10)
    ax.set_ylabel(labels[i], fontsize = 10)

plt.tight_layout()
plt.savefig(specificdir + cellid + '_behavior_plots.png', dpi = 500, bbox_inches='tight')



############ plot the speed of the cell along its smoothened trajectory
############ and the projected speed of the raw trajectory onto the smoothened
fig, ax = plt.subplots(1,1)
ax.plot(cell.time, cell.speed*time_interval, label = 'Smoothened')
ax.plot(cell.time, cell.raw_projected_speed*time_interval, label = 'Raw-projected')
plt.tight_layout()





############# get the actual positions of the cell through time
posdiffs = []
cell, runs = utils.get_consecutive_timepoints(cell, 'time', time_interval)
for r in runs:
    c = cell.iloc[r]
    tempc = c[['x_raw','y_raw','z_raw']].diff().values
    tempc[0,:] = [0,0,0]
    posdiffs.append(tempc)
posdiffs = np.concatenate(posdiffs)
cum_pos = np.cumsum(posdiffs, axis = 0)
segments = np.concatenate([cum_pos[:,np.newaxis,:][:-1], cum_pos[:,np.newaxis,:][1:]], axis=1)



############## Plot the 3d trajectory 
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

line_collection = Line3DCollection(segments, colors=cmap(norm(cell.time_min.values[:-1]))[:,:3], linewidths=2)
ax.add_collection3d(line_collection)


max_range = np.max(np.ptp(cum_pos, axis = 0))
midx = cum_pos[:,0].min()+np.ptp(cum_pos[:,0])/2
midy = cum_pos[:,1].min()+np.ptp(cum_pos[:,1])/2
midz = cum_pos[:,2].min()+np.ptp(cum_pos[:,2])/2
ax.set_xlim(midx-max_range/2, midx+max_range/2)
ax.set_ylim(midy-max_range/2, midy+max_range/2)
ax.set_zlim(midz-max_range/2, midz+max_range/2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Needed for colorbar to work
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
cbar.set_label("Time (min)")

ax.view_init(elev=220, azim = 240)

plt.show()
plt.savefig(specificdir + cellid + '3d_trajectory.png', dpi = 500, bbox_inches='tight')


