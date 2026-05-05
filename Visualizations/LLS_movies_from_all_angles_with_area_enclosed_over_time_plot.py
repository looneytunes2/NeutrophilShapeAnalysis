# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 10:11:37 2025

@author: Aaron
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tifffile
import matplotlib.gridspec as gridspec
from neutrophil_shape.config.loader import load_config
from neutrophil_shape.CustomFunctions import utils

def format_seconds(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02}:{secs:02}"


whichpcs = (1,2)

### open config and get directories
config = load_config(microscope_type='lls')
config._alignment = 'trajectory'
savedir = config.common.savedir
localdir = config.experiment.lls.localdir
moviedir = localdir / 'singlecells'
time_interval = config.im_params.time_interval


dirs = [d for d in moviedir.glob("*/") if d.is_dir()]
# dirs = [d for d in moviedir.glob('*') if '20240611_488_EGFP-CAAX_640_actin-halotag_cell2_01' in d.name]
for d in dirs:
    cellname = d.name
    image = tifffile.imread(d / f'{cellname}_full_movie.ome.tiff')
    framelist = pd.read_csv(d / f'{cellname}_framelist.csv', index_col = 0)
    framelist = framelist[framelist.columns[0]]
    #open all of the data
    posinfo = pd.read_csv(localdir / 'position_info' / f'{cellname}_cellpos.csv', index_col = 0)
    aers = pd.read_csv(savedir / 'detailed_balance' / f'{utils.whichpc_string(whichpcs)}_raw_transition_aer_cf.csv', index_col = 0)
    aers = aers.rename(columns={'real_time':'time'})
    TotalFrame = pd.merge(posinfo, aers, on=['CellID','time'], how = 'left')
    TotalFrame = TotalFrame[TotalFrame.CellID==cellname].sort_values('time').reset_index(drop=True)
    #add a movie column
    TotalFrame['Movie'] = [x.split('_frame')[0] for x in TotalFrame.cell.to_list()]
    TotalFrame['time_min'] = TotalFrame.time.values/60
    TotalFrame['area_enclosed'] = TotalFrame.aer.values * TotalFrame.time_elapsed.values
    #get times for all the frames included in the movie
    #(which may have been dropped from dataframes in analysis)
    times = np.array([])
    for m, mov in TotalFrame.groupby('image'):
        row = mov.iloc[0]
        leng = len([x for x in framelist if row.image in x])
        start = row.time
        # if row.frame != 0:
        #     start = row.time - (row.frame*time_interval)
        #     print(row.time)
        times = np.concatenate((times, np.arange(start, row.time+(leng)*time_interval, time_interval)))


    #make all of the max projections
    xyproj = np.max(image[:,0,:,:,:], axis = 1)
    xzproj = np.max(image[:,0,:,:,:], axis = 2)
    yzproj = np.max(image[:,0,:,:,:], axis = 3)
    #flip the yx projection to be portrait orientation
    yzproj = np.rot90(yzproj, axes=(2,1))
    
    
    #bleaching correction
    xyproj_bc = np.zeros(xyproj.shape)
    xzproj_bc = np.zeros(xzproj.shape)
    yzproj_bc = np.zeros(yzproj.shape)
    linearmaxes = np.linspace(xyproj[0].max(),xyproj[-1].max(), len(xyproj))
    for n in range(len(xyproj)):
        #all images have a min of zero so just divide by max
        xyproj_bc[n] = xyproj[n]/linearmaxes[n]
        xzproj_bc[n] = xzproj[n]/linearmaxes[n]
        yzproj_bc[n] = yzproj[n]/linearmaxes[n]

            
        
    
    # #adjust the b+c of all of the projections (also normalize I suppose)
    # xyproj_bc = intensity_normalization(xyproj, [0,10])
    # xzproj_bc = 
    # yzproj_bc = 
    
    # Create a figure
    fig = plt.figure(figsize=(7, 7))
    # Create a GridSpec with 2 rows and 2 columns
    gs = gridspec.GridSpec(2, 2, width_ratios=[image.shape[-3], image.shape[-1]],
                           height_ratios=[image.shape[-2], image.shape[-3]],
                           wspace = 0.01,
                           hspace = 0.05,
                           figure=fig)
    fig.patch.set_facecolor('black')
    fig.subplots_adjust(left=0.08, right=1, top=1, bottom=0.08, wspace=0, hspace=0)
    # Use first column for both subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 0])
    
    
    scalebar_x_displacement = xyproj.shape[-1]-10
    scalebar_y_displacement = xyproj.shape[-2]-14
    scalebar_length = 10
    resolution = 0.145*4 #um / pixel
    
    #scalebar
    sb = ax2.plot([scalebar_x_displacement-(scalebar_length/resolution), scalebar_x_displacement],
            [scalebar_y_displacement, scalebar_y_displacement],
            lw = 3,
            color = 'white',
            zorder=2)
    
    #scalbar text
    sb_label = ax2.text(scalebar_x_displacement-(scalebar_length/resolution)-6,
                        scalebar_y_displacement + 10,
                        f'{scalebar_length} μm',
                        color = 'white',
                        fontdict = {'fontsize': 10})
    
    
    ### time label
    timer = ax2.text(3,18,'00:00', color = 'white', fontdict = {'fontsize': 24})
    
    #all the images
    xyimsh = ax2.imshow(xyproj_bc[0], cmap = 'gray', zorder = 1)
    xzimsh = ax3.imshow(xzproj_bc[0], cmap = 'gray', zorder = 1)
    yzimsh = ax1.imshow(yzproj_bc[0], cmap = 'gray', zorder = 1)
    
    #aer graph
    aerplot = ax4.plot(TotalFrame.time_min, TotalFrame.area_enclosed.cumsum(), color = 'white', zorder = 2)
    ax4.set_xlabel('Time (min)', color = 'white')
    ax4.set_ylabel('Area Enclosed', color = 'white')
    ax4.set_facecolor('black')
    ax4.tick_params(axis='x', colors='white')
    ax4.tick_params(axis='y', colors='white')
    ax4.set_xticks(range(0,65,5))
    ax4.set_xticklabels(np.arange(0,65,5).astype(str), fontsize = 8)
    for spine in ax4.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(1)
    #vline for the aer graph
    vline = ax4.axvline(color='0.6', zorder=1)
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(1)
    
    
    
    
    # make function for updating point position
    def animate(i,):
        #set the current set of data
        xyimsh.set_data(xyproj_bc[i])
        xyimsh.set_zorder(0)
        xzimsh.set_data(xzproj_bc[i])
        xzimsh.set_zorder(0)
        yzimsh.set_data(yzproj_bc[i])
        yzimsh.set_zorder(0)
        
        #animate the vline on the aer plot
        vtime = times[i]/60
        vline.set_xdata([vtime])
        
        ### scale bar info
        scalebar_x_displacement = xyproj.shape[-1]-10
        scalebar_y_displacement = xyproj.shape[-2]-14
        scalebar_length = 10
        resolution = 0.145*4 #um / pixel
        #scalebar "animation"
        sb[0].set_data([scalebar_x_displacement-(scalebar_length/resolution), scalebar_x_displacement],
                [scalebar_y_displacement, scalebar_y_displacement])
        #scalebar label "animation
        sb_label.set_text(f'{scalebar_length} μm')
        
        #timer animation
        timer.set_text(format_seconds(times[i]))
        
        return xyimsh, xzimsh, yzimsh, sb[0], sb_label, timer, vline#, aerplot
    
    
    
    #add two to the frame count to adjust the range function and to add a blank frame at the beginning
    ani = FuncAnimation(fig, animate, interval=10, blit=True, repeat=True,
                        frames=len(xyproj),)
    
    
    plt.show()
    

    ani.save(d.joinpath(cellname + '_animated_allaxes.mp4'), fps=10, dpi = 300)#, extra_args=['-vcodec', 'libx264'])

    
    plt.close(fig)