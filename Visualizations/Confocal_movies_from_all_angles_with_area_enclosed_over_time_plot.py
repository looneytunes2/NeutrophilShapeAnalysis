# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 10:11:37 2025

@author: Aaron
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tifffile
import matplotlib.gridspec as gridspec
from neutrophil_shape.config.loader import load_config
from neutrophil_shape.config.models import Config
from neutrophil_shape.CustomFunctions import utils
import dataclasses


whichpcs = (1,2)

### open config and get directories
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'
xyres = config.im_params.xyres
zstep = config.im_params.zstep
savedir = config.common.savedir
time_interval = config.im_params.time_interval


    
def get_exp_from_name(
        cellname: str, 
        config: Config = config,
        ):
    
    date = cellname.split('_')[0]
    
    dir_dict = dataclasses.asdict(config.experiment)
    for key in dir_dict.keys():
        datelist = dir_dict[key]['dates']
        if date in datelist:
            exp = key
    return exp
    
def generate_confocal_ae_movie(
        cellname: str,
        ):
    ### if cellname isn't provided, randomly use one with at least 30 frames
    
    exp = get_exp_from_name(cellname)
    localdir = getattr(config.experiment, exp).localdir
    ## get directory to save movie in the local data dir
    moviedir = localdir / 'singlecells' / cellname
    if not moviedir.exists():
        moviedir.mkdir(parents = True)

    #open all of the data
    posinfo = pd.read_csv(localdir / 'position_info' / f'{cellname}_cellpos.csv', index_col = 0)
    posinfo['time'] = posinfo.frame * time_interval
    aers = pd.read_csv(savedir / 'detailed_balance' / f'{utils.whichpc_string(whichpcs)}_raw_transition_aer_cf.csv', index_col = 0)
    aers = aers.rename(columns={'real_time':'time'})
    TotalFrame = pd.merge(posinfo, aers, on=['CellID','time'], how = 'left')
    TotalFrame = TotalFrame[TotalFrame.CellID==cellname].sort_values('time').reset_index(drop=True)
    TotalFrame['time_min'] = TotalFrame.time.values/60
    TotalFrame['area_enclosed'] = TotalFrame.aer.values * TotalFrame.time_elapsed.values
    
    times = TotalFrame.time



    
    #get the min and max positions in the original image for this particular cell
    #and this particular timeframe
    maxarr = np.max(posinfo[['xmaxcrop','ymaxcrop','zmaxcrop']].values,axis = 0)
    minarr = np.min(posinfo[['xmincrop','ymincrop','zmincrop']].values,axis = 0)
    croparr = np.array([minarr[0],maxarr[0],minarr[1],maxarr[1],minarr[2],maxarr[2]])
    
    
    #make the image that is the size of the crop
    cropim = np.zeros((len(posinfo),
                        len(range(croparr[4],croparr[5])),
                        len(range(croparr[2],croparr[3])),
                        len(range(croparr[0],croparr[1])),
                        ))
    
    #iterate through the frames in the minidf
    for i, row in posinfo.iterrows():
        #open the cropped image for this frame
        tempim = tifffile.imread(localdir / 'processed_images' / f'{row.cell}_raw.tiff')
        #get the cropped coordinates of the cropped image
        x = row.xmincrop-croparr[0]
        y = row.ymincrop-croparr[2]
        z = row.zmincrop-croparr[4]
        #insert the cropped image into the new total cropped movie
        cropim[i,
               z:z+tempim.shape[-3],
               y:y+tempim.shape[-2],
               x:x+tempim.shape[-1]] = tempim



    #make all of the max projections
    xyproj = np.max(cropim, axis = 1)
    xzproj = np.max(cropim, axis = 2)
    yzproj = np.max(cropim, axis = 3)
    #flip the yx projection to be portrait orientation
    yzproj = np.rot90(yzproj, axes=(2,1))
    
    
    #bleaching correction
    xyproj_bc = np.zeros(xyproj.shape)
    xzproj_bc = np.zeros(xzproj.shape)
    yzproj_bc = np.zeros(yzproj.shape)
    linearmaxes = np.linspace(xyproj[0].max(),xyproj[-1].max(), len(xyproj))
    for n in range(len(xyproj)):
        #all images have a min of zero so just divide by max
        xyproj_bc[n] = (xyproj[n] - xyproj[n].min())/(linearmaxes[n] - xyproj[n].min())
        xzproj_bc[n] = (xzproj[n] - xzproj[n].min())/(linearmaxes[n] - xzproj[n].min())
        yzproj_bc[n] = (yzproj[n] - yzproj[n].min())/(linearmaxes[n] - yzproj[n].min())

            
        


    ##### Create a figure
    fig = plt.figure(figsize=(7, 7))
    # Create a GridSpec with 2 rows and 2 columns
    gs = gridspec.GridSpec(2, 2, width_ratios=[cropim.shape[-3] * zstep, cropim.shape[-1] * xyres],
                           height_ratios=[cropim.shape[-2] * xyres, cropim.shape[-3] * zstep],
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
    xzimsh = ax3.imshow(xzproj_bc[0], cmap = 'gray', zorder = 1, aspect = zstep/xyres)
    yzimsh = ax1.imshow(yzproj_bc[0], cmap = 'gray', zorder = 1, aspect = xyres/zstep)
    
    #aer graph
    ax4.plot(TotalFrame.time_min, TotalFrame.area_enclosed.cumsum(), color = 'white', zorder = 2)
    ax4.set_xlabel('Time (min)', color = 'white')
    ax4.set_ylabel('Area Enclosed', color = 'white')
    ax4.set_facecolor('black')
    ax4.tick_params(axis='x', colors='white')
    ax4.tick_params(axis='y', colors='white')
    # ax4.set_xticks(range(0,65,5))
    ax4.tick_params('x', size = 8)
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
        vtime = TotalFrame.time_min[i]
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
        timer.set_text(utils.format_seconds(times[i]))
        
        return xyimsh, xzimsh, yzimsh, sb[0], sb_label, timer, vline#, aerplot
    
    
    
    #add two to the frame count to adjust the range function and to add a blank frame at the beginning
    ani = FuncAnimation(fig, animate, interval=10, blit=True, repeat=True,
                        frames=len(xyproj),)
    
    
    # plt.show()
    
    ani.save(moviedir.joinpath(cellname + '_allaxes_aeplot.mp4'), fps=4, dpi = 300)#, extra_args=['-vcodec', 'libx264'])
    plt.close(fig)


cellname = '20231116_488EGFP-CAAX_3mA_37C_1_cell_0'

FullFrame = pd.read_csv(savedir / 'shape_data' / 'All_Data_with_CGPS_bins.csv', index_col = 0)
randomframe = FullFrame[FullFrame.Treatment == 'Random'].copy()
framecounts = randomframe.value_counts('CellID')
celllist = framecounts[framecounts>=30].index.tolist()

for cellname in celllist[:20]:
    generate_confocal_ae_movie(cellname)


    
