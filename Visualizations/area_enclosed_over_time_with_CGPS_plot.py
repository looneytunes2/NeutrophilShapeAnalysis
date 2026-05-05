# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 10:11:37 2025

@author: Aaron
"""

import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from neutrophil_shape.config.loader import load_config
from neutrophil_shape.CustomFunctions import utils
import math
from matplotlib import cm
import seaborn as sns

def format_seconds(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02}:{secs:02}"

def color_interpolation(pointarray):
    timepoints = np.unique(pointarray[:,0])
    num_timepoints = len(timepoints)
    all_interp = []
    for p, tp in enumerate(timepoints):
        points = pointarray[pointarray[:,0] == tp]
        ## don't get points for nan
        if points[0,0] == np.nan:
            continue

        ### get colors in this second
        current_colors = np.linspace(p/num_timepoints, (p + 1)/num_timepoints, len(points))
        all_interp.extend(current_colors)


    return all_interp



whichpcs = (1,2)

## choose a cmap
cmap = cm.Greys_r
tail_length = 15 # seconds

### open config and get directories
config = load_config(microscope_type='lls')
config._alignment = 'trajectory'
savedir = config.common.savedir
datadir = savedir / 'shape_data'
dbdir = savedir / 'detailed_balance'
localdir = config.experiment.lls.localdir
moviedir = localdir / 'singlecells'
time_interval = config.im_params.time_interval
nbins = config.db_params.nbins



#open all of the data
aers = pd.read_csv(dbdir / f'{utils.whichpc_string(whichpcs)}_raw_transition_aer_cf.csv', index_col = 0)
centers = pd.read_csv(datadir / 'PC_bin_centers.csv', index_col = 0)



# cellname = '20240611_488_EGFP-CAAX_640_actin-halotag_cell2_01'
for cellname in aers.CellID.unique():

    TotalFrame = aers[aers.CellID==cellname].sort_values('real_time').reset_index(drop=True)

    ###### BUILD DWELL TIME MAP
    transdf_sep = pd.read_csv(dbdir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_interpolated_transitions_separated.csv'), index_col=0)

    ########### calculate the DWELL TIME in the WHOLE CGPS #############
    hms = np.zeros((len(transdf_sep.CellID.unique()), nbins, nbins))
    countmap = np.zeros((len(transdf_sep.CellID.unique()), nbins, nbins))
    for i, (treat, tdf) in enumerate(transdf_sep.groupby('CellID')):
        for x in range(nbins):
            for y in range(nbins):
                current =  tdf[(tdf['from_x'] == x+1) & (tdf['from_y'] == y+1)]
                if current.empty:
                    hms[i,y,x] = 0
                else:
                    hms[i,y,x] = current.time_elapsed.mean()
                    #add the number of counts in this bin
                    countmap[i,y,x] = len(current)

    avg_hm = np.mean(hms, axis = 0)



    ##### subtract initial real time from the dataframe so that time tracking starts at 0
    TotalFrame.real_time = TotalFrame.real_time - TotalFrame.real_time.values[0]
    cell, runs = utils.get_consecutive_transitions(TotalFrame)
    points_list = []
    for r in runs:
        run = cell.iloc[r].copy()
        bintraj = run[['from_x','from_y']].values
        ### add the ending point to the end of the trajectory
        bintraj = np.insert(bintraj, len(bintraj), run.iloc[-1][['to_x','to_y']].values , axis=0)
        ### add the ending time
        inter_time = np.insert(run.real_time.values, len(run), run.iloc[-1].real_time+run.iloc[-1].time_elapsed, axis=0)
        #interpolate based on path
        tck, b = interpolate.splprep(bintraj.T, u=inter_time,k=1, s=0)
        
        ### get distance interpolator
        #position diffs
        diffs = np.diff(bintraj, axis = 0)
        distances = np.sqrt(diffs[:,0]**2 + diffs[:,1]**2)
        ## add zero start to distances
        distances = np.insert(distances, 0, 0)
        ## cumulative distance
        cumdist = np.cumsum(distances)
        ## interpolate distance
        dtck = interpolate.splrep(inter_time, cumdist, k=1, s=0)

        ### interpolate positions for every second
        #range to interpolate
        timerange = np.arange(inter_time[0], inter_time[-1], 1)
        for t in range(len(timerange)-1):
            start = timerange[t]
            stop = timerange[t+1]
            distance = interpolate.splev(stop, dtck) - interpolate.splev(start, dtck)
            #number of points to interpolate based on distance and time interval
            interpoints = np.linspace(start=start, stop = stop, num = math.ceil(30*distance), endpoint = False)
            interlist = interpolate.splev(interpoints,tck)
            interarray = np.array(interlist).T
            interarray = np.insert(interarray, 0, start, axis=1)
            points_list.append(interarray)
    ### stack points
    points_array = np.vstack(points_list)


    ######## fill missing seconds
    def fill_missing_ids(arr):
        col = arr[:, 0].astype(int)
        full_range = np.arange(col.min(), col.max() + 1)
        missing = np.setdiff1d(full_range, np.unique(col))
        
        if len(missing) == 0:
            return arr
        
        # build missing rows
        missing_rows = np.column_stack([
            missing,
            np.full(len(missing), np.nan),
            np.full(len(missing), np.nan)
        ])
        
        # append and sort by first column
        result = np.vstack([arr, missing_rows])
        result = result[result[:, 0].argsort(kind="stable")]
        
        return result

    points_array_filled = fill_missing_ids(points_array)


    #### expand df time index to add aer nans
    #first add aer cumsum
    TotalFrame['area_enclosed'] = TotalFrame.aer * TotalFrame.time_elapsed
    ## get cumulative sum of area enclosed
    TotalFrame['ae_cumsum'] = TotalFrame.area_enclosed.cumsum().copy()
    ### add time in minutes
    TotalFrame['timemin'] = TotalFrame.real_time.values/60
    ### insert nan rows for missing seconds
    breaks = np.where(TotalFrame.time_elapsed != TotalFrame.real_time.diff())[0]
    insert_rows = pd.DataFrame({
        col: [np.nan] * len(breaks) for col in TotalFrame.columns
        })
    insert_rows["real_time"] = TotalFrame.loc[breaks, "real_time"].values - 1
    TotalFrame_with_breaks = pd.concat([TotalFrame, insert_rows]).sort_values("real_time").reset_index(drop=True)




    ########## Create a figure
    fig = plt.figure(figsize=(14, 7))
    # Create a GridSpec with 2 rows and 2 columns
    gs = gridspec.GridSpec(1, 2,
                        #    width_ratios=[image.shape[-3], image.shape[-1]],
                        #    height_ratios=[image.shape[-2], image.shape[-3]],
                            wspace = 0.1,
                            hspace = 0.05,
                            figure=fig)
    # fig.patch.set_facecolor('black')
    fig.subplots_adjust(left=0.08, right=1, top=1, bottom=0.08, wspace=0, hspace=0)
    # Use first column for both subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])




    # #plot heatmap with seaborn
    # sns.heatmap(
    #     np.mean(hms, axis = 0),
    #     vmin=0,
    #     vmax=np.max(np.mean(hms, axis = 0)), 
    #     cmap='viridis',
    #     square=True,
    #     xticklabels = True,
    #     yticklabels = True,
    #     cbar_kws={"shrink": 0.8,"label": "Dwell Time (s)","pad":0.01},
    #     ax = ax1,
    # )

    # #correct axis orientations
    # ax1.invert_yaxis()

    ##### draw the heatmap of dwell times (or anything else you want)

    meshx = np.arange(1, nbins+1, 1)
    meshy = np.arange(1, nbins+1, 1)
    mesh = ax1.pcolormesh(meshx, meshy, avg_hm, cmap='viridis', shading='auto')

    #add "grid lines" first 
    for h in np.linspace(0.5, nbins+0.5, nbins+1):
        ax1.axhline(h, linestyle='-', color='grey', alpha=0.3) # horizontal lines
        ax1.axvline(h, linestyle='-', color='grey', alpha=0.3) # vertical lines
    #### set up all the axes info        
    ax1.set_aspect("equal")
    ax1.set_xlabel(f'PC{whichpcs[0]}', fontsize = 20)
    ax1.set_ylabel(f'PC{whichpcs[1]}', fontsize = 20)
    ax1.set_xticks(list(range(1,nbins+1)),[round(x,1) for x in centers[f'PC{whichpcs[0]}'].to_list()], fontsize = 9)
    ax1.set_yticks(list(range(1,nbins+1)),[round(x,1) for x in centers[f'PC{whichpcs[1]}'].to_list()], fontsize = 9)
    ax1.set_xlim(0.5,nbins+0.5)
    ax1.set_ylim(0.5,nbins+0.5)
    # create a point in the axes
    point = ax1.scatter([],[], c = [], cmap = cmap, s = 6, zorder = 2)




    # Add the colorbar to the new axis
    cbar = fig.colorbar(mesh, ax=ax1, location = 'top', orientation='horizontal', shrink = 0.89, pad=0.01)
    # cbar = fig.colorbar(axes[-1].collections[0], cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Mean Dwell Time (s)', fontsize=17)
    cbar.ax.xaxis.set_label_position('top')
    # cbar.ax.set_xticklabels(np.linspace(-1,1,len(cbar.ax.get_xticklabels())).astype(str),fontsize=22)


    # ### time label
    # timer = ax2.text(3,70,'00:00', color = 'black', fontdict = {'fontsize': 24})


    #### aer graph
    aerplot = ax2.plot(TotalFrame_with_breaks.timemin, TotalFrame_with_breaks.ae_cumsum, color = 'black', zorder = 2)
    ax2.plot(np.arange(0,63,0.1), np.zeros(len(np.arange(0,63,0.1))), color = 'white', zorder = 1)
    ax2.set_xlabel('Time (min)', color = 'black', fontsize = 20)
    ax2.set_ylabel('Area Enclosed', color = 'black', fontsize = 20)
    # ax2.set_facecolor('black')
    ax2.tick_params(axis='x', colors='black')
    ax2.tick_params(axis='y', colors='black')
    ax2.set_xticks(range(0,65,5))
    ax2.set_xticklabels(np.arange(0,65,5).astype(str), fontsize = 8)
    for spine in ax2.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    #vline for the aer graph
    vline = ax2.axvline(color='0.6', zorder=1)



    # make function for updating point position
    def animate(i,):
        #set the current set of CGPS data
        ## get most recent three seconds
        time_window = np.arange(max(0,i-tail_length), i)
        mask = np.isin(points_array_filled[:, 0], time_window)
        points = points_array_filled[mask]
        colorpoints = color_interpolation(points)
        # colors = cmap(colorpoints)
        point.set_offsets(points[:,1:])
        point.set_array(colorpoints)

        #animate the vline on the aer plot
        vtime = i/60
        vline.set_xdata([vtime])
        
        # #timer animation
        # timer.set_text(format_seconds(i))
        
        return point, vline, #timer, 


    ## get total number of seconds as frame
    total_seconds = int(points_array_filled[:,0].max())

    #add two to the frame count to adjust the range function and to add a blank frame at the beginning
    ani = FuncAnimation(fig, animate, interval=10, blit=True, repeat=True,
                        frames=range(total_seconds),)


    plt.show()


    ani.save(moviedir.joinpath(cellname, cellname + '_animated_CGPS_and_AER.mp4'), fps=30, dpi = 300)#, extra_args=['-vcodec', 'libx264'])


    plt.close(fig)


