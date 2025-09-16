# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 10:13:06 2025

@author: Aaron
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from aicsimageio.readers import CziReader
import matplotlib.gridspec as gridspec
import skimage.measure

def format_seconds(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02}:{secs:02}"


basedir = '//10.158.28.35/TheriotLab_LLS7/Aaron/manual_crop_lls_decon/'
cellname = '20240520_488_EGFP-CAAX_561_mysoin-mApple_37C_cell2-04-Subset-01_deskewed_decon.czi'
im = CziReader(basedir + cellname).data[:,1,:,:,:]


#### crop around the cell evenly
fullimshape = im.shape #shape of the image
imcent = np.array(fullimshape[-3:])/2
infolist = []
for i, frame in enumerate(im):
    im_labeled, n_labels = skimage.measure.label(
                              frame>500, background=0, return_num=True,  )

    im_props = skimage.measure.regionprops(im_labeled)
    tempdf = pd.DataFrame([])
    td = []
    for count, prop in enumerate(im_props):
        z,y,x = np.array(prop.centroid)
        thebox = np.array(prop.bbox)
        area = prop.area
        dist = np.sqrt((z-imcent[0])**2 + (y-imcent[1])**2 + (x-imcent[2])**2)
        td.append({'cell':count, 'frame':i,'z_min':thebox[0], 'y_min':thebox[1], 
                'x_min':thebox[2],'z_max':thebox[3], 'y_max':thebox[4], 'x_max':thebox[5],
               'z':z, 'y':y, 'x': x, 'area':area, 'dist':dist})
    tempdf = pd.DataFrame(td)
    infolist.append(tempdf.sort_values(['area','dist']).iloc[-1])
infodf = pd.DataFrame(infolist)#, ignore_index = True)

zmin = int(max(0,infodf.z_min.min() - 30))
zmax = int(min(fullimshape[-3],infodf.z_max.max() + 30))
ymin = int(max(0,infodf.y_min.min() - 30))
ymax = int(min(fullimshape[-2],infodf.y_max.max() + 30))
xmin = int(max(0,infodf.x_min.min() - 30))
xmax = int(min(fullimshape[-1],infodf.x_max.max() + 30))


#crop the giant image
# cropped = im[:,1,:, 50:225, 450:625,]
cropped = im[:,zmin:zmax, ymin:ymax, xmin:xmax]


#get the maximum projection of the full movie
cropshape = cropped.shape

#make all of the max projections
xyproj = np.max(cropped, axis = 1)
xzproj = np.max(cropped, axis = 2)
yzproj = np.max(cropped, axis = 3)
#flip the yx projection to be portrait orientation
yzproj = np.rot90(yzproj, axes=(2,1))

#adjust brightness and contrast for xy
#set min to zero
xyproj_bc = xyproj-xyproj.min()
xzproj_bc = xzproj-xzproj.min()
yzproj_bc = yzproj-yzproj.min()
#set 1 to a good value
maxval = np.percentile(xyproj[0], 99.8)

xyproj_bc = xyproj_bc/maxval
xyproj_bc[xyproj_bc>1] = 1
xzproj_bc = xzproj_bc/maxval
xzproj_bc[xzproj_bc>1] = 1
yzproj_bc = yzproj_bc/maxval
yzproj_bc[yzproj_bc>1] = 1

#invert the colors
xyinv = abs(xyproj_bc - 1)
xzinv = abs(xzproj_bc - 1)
yzinv = abs(yzproj_bc - 1)
    



# Create a figure
fig = plt.figure(figsize=(7, 7))
# Create a GridSpec with 2 rows and 2 columns
gs = gridspec.GridSpec(2, 2, width_ratios=[cropshape[-1], cropshape[-3]],
                       height_ratios=[cropshape[-2], cropshape[-3]],
                       wspace = 0.04,
                       hspace = -0.14,
                       figure=fig)
fig.patch.set_facecolor('white')
# fig.subplots_adjust(left=0.01, right=0.99, top=1, bottom=0, wspace=0, hspace=0)
# Use first column for both subplots
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
# ax4 = fig.add_subplot(gs[1, 1])

#get positions to maximize image space in the square video
ax1pos = ax1.get_position()
left = ax1pos.min[0]
top = ax1pos.max[1]
ax2pos = ax2.get_position()
right = ax2pos.max[0]
ax3pos = ax3.get_position()
bottom = ax3pos.min[1]

leftrightspace = 1 - (right-left)


#scale bar info
scalebar_x_displacement = xyproj.shape[-1]-12
scalebar_y_displacement = xyproj.shape[-2]-24
scalebar_length = 5
resolution = 0.145 #um / pixel

#scalebar
sb = ax1.plot([scalebar_x_displacement-(scalebar_length/resolution), scalebar_x_displacement],
        [scalebar_y_displacement, scalebar_y_displacement],
        lw = 3,
        color = 'black',
        zorder=2)

#scalbar text
sb_label = ax1.text(scalebar_x_displacement-(scalebar_length/resolution)-0.5,
                    scalebar_y_displacement + 17,
                    f'{scalebar_length} μm',
                    color = 'black',
                    fontdict = {'fontsize': 10})


### time label
time_interval = 5
timelist = np.arange(cropshape[0])*time_interval
timer = ax1.text(4,20,'00:00', color = 'black', fontdict = {'fontsize': 16})


#also add perspective labels
xylabel = ax1.text(4, cropshape[-2]-8, 'xy', fontsize = 16)
xzlabel = ax2.text(4, cropshape[-2]-8, 'xz', fontsize = 16)
yzlabel = ax3.text(4, cropshape[-3]-8, 'yz', fontsize = 16)


#all the images
xyimsh = ax1.imshow(xyinv[0], cmap = 'gray', zorder = 1)
xzimsh = ax3.imshow(xzinv[0], cmap = 'gray', zorder = 1)
yzimsh = ax2.imshow(yzinv[0], cmap = 'gray', zorder = 1)

# #aer graph
# aerplot = ax4.plot(TotalFrame.time/60, TotalFrame.aer.cumsum(), color = 'white', zorder = 2)
# ax4.set_xlabel('Time (min)', color = 'white')
# ax4.set_ylabel('Area Enclosed', color = 'white')
# ax4.set_facecolor('black')
# ax4.tick_params(axis='x', colors='white')
# ax4.tick_params(axis='y', colors='white')
# ax4.set_xticks(range(0,65,5))
# ax4.set_xticklabels(np.arange(0,65,5).astype(str), fontsize = 8)
# for spine in ax4.spines.values():
#     spine.set_edgecolor('white')
#     spine.set_linewidth(1)
# #vline for the aer graph
# vline = ax4.axvline(color='0.6', zorder=1)

for a,ax in enumerate([ax1, ax2, ax3]):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linestyle('-')
        spine.set_linewidth(1)




# make function for updating point position
def animate(i,):
    #set the current set of data
    xyimsh.set_data(xyinv[i])
    xyimsh.set_zorder(0)
    xzimsh.set_data(xzinv[i])
    xzimsh.set_zorder(0)
    yzimsh.set_data(yzinv[i])
    yzimsh.set_zorder(0)
    
    # #animate the vline on the aer plot
    # vtime = timelist[i]/60
    # vline.set_xdata(vtime)
    
    # ### scale bar info
    # scalebar_x_displacement = xyproj.shape[-1]-10
    # scalebar_y_displacement = xyproj.shape[-2]-14
    # scalebar_length = 10
    # resolution = 0.145*4 #um / pixel
    #scalebar "animation"
    sb[0].set_data([scalebar_x_displacement-(scalebar_length/resolution), scalebar_x_displacement],
            [scalebar_y_displacement, scalebar_y_displacement])
    #scalebar label "animation
    sb_label.set_text(f'{scalebar_length} μm')
    
    #timer animation
    timer.set_text(format_seconds(timelist[i]))
    
    return xyimsh, xzimsh, yzimsh, sb[0], sb_label, timer, xylabel, xzlabel, yzlabel#vline#, aerplot



#add two to the frame count to adjust the range function and to add a blank frame at the beginning
ani = FuncAnimation(fig, animate, interval=10, blit=True, repeat=True,
                    frames=len(xyproj),)


plt.show()


#save the animation
ani.save(__file__.split('.')[0]  + '.mp4', fps=10, dpi = 300)#, extra_args=['-vcodec', 'libx264'])


plt.close(fig)

