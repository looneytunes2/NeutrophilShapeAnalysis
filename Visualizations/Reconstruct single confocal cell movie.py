# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 11:41:23 2025

@author: Aaron
"""


import multiprocessing
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from neutrophil_shape.config.loader import load_config
from neutrophil_shape.CustomFunctions.segment_cells2short import read_cropped_substack_confocal





# if __name__ == '__main__':
config = load_config(microscope_type='confocal')
time_interval = config.im_params.time_interval
xyres = config.im_params.xyres
zstep = config.im_params.zstep
basedir = config.experiment.galv.localdir
posdir = basedir / 'position_info'
cellname = '20231116_488EGFP-CAAX_3mA_37C_2_cell_9'
#raw image slices on the server, saved as one tiff per channel, time and z
imagename = cellname.split('_cell_')[0]
rawdir = config.experiment.galv.serverdir / imagename.split('_')[0] / imagename / 'Default'
stackshape = config.im_params.stackshape #zyx size of a single timepoint



def format_seconds(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02}:{secs:02}"



posdf = pd.read_csv(posdir / f'{cellname.split("_frame")[0]}_cellpos.csv', index_col = 0)
#limit the posdf to only frames that I have cropped raw and segmented images for


#get the dataframe for the frame range
minidf = posdf[posdf.frame<180].reset_index(drop=True)


#get the min and max positions in the original image for this particular cell
#and this particular timeframe
maxarr = np.max(minidf[['xmaxcrop','ymaxcrop','zmaxcrop']].values,axis = 0)
minarr = np.min(minidf[['xmincrop','ymincrop','zmincrop']].values,axis = 0)
croparr = np.array([minarr[0],maxarr[0],minarr[1],maxarr[1],minarr[2],maxarr[2]])


#bounding box of the crop in the original image, clipped to the image
croparr = np.array(croparr).astype(int)
xmin, xmax = croparr[0], min(croparr[1], stackshape[-1])
ymin, ymax = croparr[2], min(croparr[3], stackshape[-2])
zmin, zmax = croparr[4], min(croparr[5], stackshape[-3])

#collect the arguments for reading each frame
mapargs = [(rawdir, int(frame), stackshape, range(zmin, zmax), ymin, ymax, xmin, xmax)
           for frame in minidf.frame]
#read the frames in parallel (the slices come over the network, so reading is io bound)
cropframes = []
# with multiprocessing.Pool(processes=min(60, len(mapargs))) as pool:
for ma in mapargs:
    cropframes.append(read_cropped_substack_confocal(ma))
cropim = np.stack(cropframes)
del cropframes


#get outlines of maxprojections from different perspectives
xyproj = np.max(cropim, axis = 1)
xzproj = np.max(cropim, axis = 2)
yzproj = np.max(cropim, axis = 3)
#flip the projections so that it looks like side of a cube folded out
yzproj = np.rot90(yzproj, axes=(2,1))


#get real time
minidf['time'] = minidf.frame*time_interval
times = minidf.time.sort_values().values


# Create a figure
fig = plt.figure(figsize=(7, 7))
# Create a GridSpec with 2 rows and 2 columns
#the three views fill the upper left triangle of the grid: xy in the upper left, yz to its right
#(the same height), xz below it (the same width)
gs = gridspec.GridSpec(2, 2, width_ratios=[cropim.shape[-1] * xyres, cropim.shape[-3] * zstep],
                       height_ratios=[cropim.shape[-2] * xyres, cropim.shape[-3] * zstep],
                       wspace = 0.01,
                       hspace = 0.005,
                       figure=fig)
fig.patch.set_facecolor('white')
# fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
ax1 = fig.add_subplot(gs[0, 1]) #yz
ax2 = fig.add_subplot(gs[0, 0]) #xy
ax3 = fig.add_subplot(gs[1, 0]) #xz





### time label
timer = ax2.text(3,20,'00:00', color = 'black', fontdict = {'fontsize': 20})


xyimsh = ax2.imshow(xyproj[0], cmap = 'gray_r', zorder = 1)#, aspect = 1)
xzimsh = ax3.imshow(xzproj[0], cmap = 'gray_r', zorder = 1,
                    extent=[0, cropim.shape[-1] * xyres, cropim.shape[-3] * zstep, 0])
yzimsh = ax1.imshow(yzproj[0], cmap = 'gray_r', zorder = 1, 
                    extent=[0, cropim.shape[-3] * zstep, cropim.shape[-2] * xyres, 0])


ax2.set_autoscale_on(False)

scalebar_x_displacement = xyproj.shape[-1]-20
scalebar_y_displacement = xyproj.shape[-2]-22
scalebar_length = 10
resolution = xyres


sb = ax2.plot([scalebar_x_displacement-(scalebar_length/resolution), scalebar_x_displacement],
        [scalebar_y_displacement, scalebar_y_displacement],
        lw = 3,
        color = 'black',
        zorder=2)

#scalbar text
sb_label = ax2.text(scalebar_x_displacement-(scalebar_length/resolution)-6,
                    scalebar_y_displacement + 13,
                    f'{scalebar_length} μm',
                    color = 'black',
                    fontdict = {'fontsize': 12})



for ax in [ax1, ax2, ax3]:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)




# make function for updating point position
def animate(i,):
    #set the current set of data
    xyimsh.set_data(xyproj[i])
    xyimsh.set_zorder(0)
    xzimsh.set_data(xzproj[i])
    xzimsh.set_zorder(0)
    yzimsh.set_data(yzproj[i])
    yzimsh.set_zorder(0)

    ### scale bar info

    #scalebar "animation"
    sb[0].set_data([scalebar_x_displacement-(scalebar_length/resolution), scalebar_x_displacement],
            [scalebar_y_displacement, scalebar_y_displacement])
    #scalebar label "animation
    sb_label.set_text(f'{scalebar_length} μm')

    #timer animation
    timer.set_text(format_seconds(times[i]))

    return xyimsh, xzimsh, yzimsh, sb[0], sb_label, timer

#add two to the frame count to adjust the range function and to add a blank frame at the beginning
ani = FuncAnimation(fig, animate, interval=10, blit=True, repeat=True,
                    frames=len(xyproj),)
plt.show()


specificdir = basedir / f'singlecells/{cellname}/'
# #make the directory to save this combined image
# specificdir = infrsavedir + row.CellID +'/'
# if not os.path.exists(specificdir):
#     os.makedirs(specificdir)
ani.save(specificdir / f'{cellname}_animated_allaxes.mp4', fps=10, dpi = 300)#, extra_args=['-vcodec', 'libx264'])