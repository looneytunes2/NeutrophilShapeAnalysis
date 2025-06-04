# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 10:13:06 2025

@author: Aaron
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pandas as pd
from matplotlib import cm
from scipy.spatial import distance
from scipy import interpolate
from matplotlib.colors import Normalize
from aicssegmentation.core import pre_processing_utils
import matplotlib.gridspec as gridspec
from aicsimageio.readers.tiff_reader import TiffReader

basedir = 'E:/Aaron/random_lls/'
imdir = basedir + 'processed_images/'
cellname = '20240520_488_EGFP-CAAX_561_mysoin-mApple_37C_cell2-04-Subset-01_frame_29'
im = TiffReader(imdir + cellname + '_raw.tiff').data

time_interval = 5
resolution = 0.145 #um/pixel



#get maximum projections
xymaxproj = np.max(im[0], axis=0)
xzmaxproj = np.max(im[0], axis=1)
yzmaxproj = np.max(im[0], axis=2)
#flip the yx projection to be portrait orientation
yzmaxproj = np.rot90(yzmaxproj, 3,axes=(1,0))
yzmaxproj = np.flip(yzmaxproj, axis=0)
#adjust brightness and contrast
#set min to zero
xymaxproj_bc = xymaxproj-xymaxproj.min()
xzmaxproj_bc = xzmaxproj-xzmaxproj.min()
yzmaxproj_bc = yzmaxproj-yzmaxproj.min()
#set 1 to a good value
maxval = 9000
xymaxproj_bc = xymaxproj_bc/maxval
xymaxproj_bc[xymaxproj_bc>1] = 1
xzmaxproj_bc = xzmaxproj_bc/maxval
xzmaxproj_bc[xzmaxproj_bc>1] = 1
yzmaxproj_bc = yzmaxproj_bc/maxval
yzmaxproj_bc[yzmaxproj_bc>1] = 1


# Create a figure
fig = plt.figure()#figsize=(7, 7))
# Create a GridSpec with 2 rows and 2 columns
gs = gridspec.GridSpec(2, 2,
                       height_ratios=[im.shape[-2]*resolution, im.shape[-3]*resolution],
                       width_ratios=[im.shape[-1]*resolution, im.shape[-3]*resolution],
                        hspace = 0.05,
                        wspace = -0.25,
                       figure=fig)
# fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
# Use first column for both subplots
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])#, sharex=ax1)
ax3 = fig.add_subplot(gs[1,0])

#plot images with the right proportions (basically turn the axes to microns)
ax1.imshow(xymaxproj_bc, cmap='gray') #,extent = [0, cropim.shape[-1]*xyres, 0, cropim.shape[-2]*xyres])
ax2.imshow(yzmaxproj_bc, cmap='gray') #, extent = [0, cropim.shape[-3]*zstep, 0, cropim.shape[-2]*xyres])
ax3.imshow(xzmaxproj_bc, cmap='gray') #, extent = [0, cropim.shape[-1]*xyres, 0, cropim.shape[-3]*zstep])


### add scale bars
scalebar_length = 5 # um
scalebar_x_displacement = im.shape[-1]-scalebar_length/resolution-8
scalebar_y_displacement = im.shape[-2]-8
scalebar_z_displacement = im.shape[-3]-scalebar_length/resolution-8
ax1.plot([scalebar_x_displacement+scalebar_length/resolution, scalebar_x_displacement],
        [scalebar_y_displacement, scalebar_y_displacement],
        lw = 3,
        color = 'white',
        zorder=1)
ax2.plot([scalebar_z_displacement+scalebar_length/resolution, scalebar_z_displacement],
        [scalebar_y_displacement, scalebar_y_displacement],
        lw = 3,
        color = 'white',
        zorder=1)
ax3.plot([scalebar_x_displacement+scalebar_length/resolution, scalebar_x_displacement],
        [im.shape[-3]-8, im.shape[-3]-8],
        lw = 3,
        color = 'white',
        zorder=1)
    

ax1.axis('off')
ax2.axis('off')
ax3.axis('off')


plt.tight_layout()




plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')