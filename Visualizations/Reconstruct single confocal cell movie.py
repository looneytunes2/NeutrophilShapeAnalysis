# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 11:41:23 2025

@author: Aaron
"""


import numpy as np
import pandas as pd
from aicsimageio.readers.tiff_reader import TiffReader
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#get some directories
basedir = 'E:/Aaron/Galvanotaxis_Confocal_40x_37C_10s/'
posdir = basedir+'position_info/'
segimdir = basedir+'processed_images/'
cellname = '20231116_488EGFP-CAAX_3mA_37C_2_cell_9'
#server directories
fullimshape = [361,150,1024,1024]
xyres = 0.3394
zstep = 0.7
time_interval = 10


def format_seconds(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02}:{secs:02}"



posdf = pd.read_csv(posdir+cellname.split('_frame')[0]+'_cellpos.csv', index_col = 0)
#limit the posdf to only frames that I have cropped raw and segmented images for




#which frames am I going to grab from the big movie
frame_range = posdf[posdf.frame<180].frame.values


#get the dataframe for the frame range
minidf = posdf[posdf.frame.isin(frame_range)].reset_index(drop=True)

#get the min and max positions in the original image for this particular cell
#and this particular timeframe
maxarr = np.max(minidf[['xmaxcrop','ymaxcrop','zmaxcrop']].values,axis = 0)
minarr = np.min(minidf[['xmincrop','ymincrop','zmincrop']].values,axis = 0)
croparr = np.array([minarr[0],maxarr[0],minarr[1],maxarr[1],minarr[2],maxarr[2]])


#make the image that is the size of the crop
cropim = np.zeros((len(minidf),
                    len(range(croparr[4],min(croparr[5], fullimshape[-3]))),
                    len(range(croparr[2],min(croparr[3], fullimshape[-2]))),
                    len(range(croparr[0],min(croparr[1], fullimshape[-1]))),
                    ))

#iterate through the frames in the minidf
for i, row in minidf.iterrows():
    #open the cropped image for this frame
    tempim = TiffReader(segimdir+row.cell+'_raw.tiff').data
    tempshape = tempim.shape
    #get the cropped coordinates of the cropped image
    x = row.xmincrop-croparr[0]
    y = row.ymincrop-croparr[2]
    z = row.zmincrop-croparr[4]
    #insert the cropped image into the new total cropped movie
    cropim[i,
           z:z+tempim.shape[-3],
           y:y+tempim.shape[-2],
           x:x+tempim.shape[-1]] = tempim


#get outlines of maxprojections from different perspectives
xyproj = np.max(cropim, axis = 1)
xzproj = np.max(cropim, axis = 2)
yzproj = np.max(cropim, axis = 3)
#flip the yz projection
yzproj = np.rot90(yzproj, axes=(2,1))

#get real time
minidf['time'] = minidf.frame*time_interval
times = minidf.time.sort_values().values


# Create a figure
fig = plt.figure(figsize=(7, 7))
# Create a GridSpec with 2 rows and 2 columns
gs = gridspec.GridSpec(2, 2, width_ratios=[cropim.shape[-3], cropim.shape[-1]],
                       height_ratios=[cropim.shape[-2], cropim.shape[-3]],
                       wspace = 0.01,
                       hspace = 0.05,
                       figure=fig)
fig.patch.set_facecolor('black')
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
# Use first column for both subplots
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])



scalebar_x_displacement = xyproj.shape[-1]-8
scalebar_y_displacement = xyproj.shape[-2]-14
scalebar_length = 10
resolution = xyres

#scalbar text
sb_label = ax2.text(scalebar_x_displacement-(scalebar_length/resolution)-6,
                    scalebar_y_displacement + 10,
                    f'{scalebar_length} μm',
                    color = 'white',
                    fontdict = {'fontsize': 12})


### time label
timer = ax2.text(1,14,'00:00', color = 'white', fontdict = {'fontsize': 24})


xyimsh = ax2.imshow(xyproj[0], cmap = 'gray', zorder = 1)
xzimsh = ax3.imshow(xzproj[0], cmap = 'gray', zorder = 1)
yzimsh = ax1.imshow(yzproj[0], cmap = 'gray', zorder = 1)



sb = ax2.plot([scalebar_x_displacement-(scalebar_length/resolution), scalebar_x_displacement],
        [scalebar_y_displacement, scalebar_y_displacement],
        lw = 3,
        color = 'white',
        zorder=2)

for ax in [ax1, ax2, ax3]:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
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
    scalebar_x_displacement = xyproj.shape[-1]-8
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
    
    return xyimsh, xzimsh, yzimsh, sb[0], sb_label, timer

#add two to the frame count to adjust the range function and to add a blank frame at the beginning
ani = FuncAnimation(fig, animate, interval=10, blit=True, repeat=True,
                    frames=len(xyproj),)
plt.show()


specificdir = f'E:/Aaron/random_lls/singlecells/{cellname}/'
# #make the directory to save this combined image
# specificdir = infrsavedir + row.CellID +'/'
# if not os.path.exists(specificdir):
#     os.makedirs(specificdir)
ani.save(specificdir + cellname + '_animated_allaxes.mp4', fps=10, dpi = 300)#, extra_args=['-vcodec', 'libx264'])