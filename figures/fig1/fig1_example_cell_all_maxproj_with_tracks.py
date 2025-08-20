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
from CustomFunctions.segment_cells2short import MM_slicetostack_reader
from matplotlib import cm
from scipy.spatial import distance
from scipy import interpolate
from matplotlib.colors import Normalize
from aicssegmentation.core import pre_processing_utils
import matplotlib.gridspec as gridspec


basedir = 'E:/Aaron/Galvanotaxis_Confocal_40x_37C_10s/'
trackdir = basedir + 'Tracking_Images/'
image = os.listdir(trackdir)[1]
trackinfo = pd.read_csv(Path(trackdir,image,image+'_TrackMateLog.csv'))
time_interval = 10

#server directories
raw_dir = '//10.158.28.37/ExpansionHomesA/avlnas/HL60 Galv/'
direct = raw_dir +image.split('_')[0]+'/' +image+'/Default/'
fullimshape = [361,150,1024,1024]
xyres = 0.3394 #um / pixel 
zstep = 0.7 # um


#which frames to project and which to extract for tracks
example_frame = 90
track_length = 24
frame_range = np.array(range(int(example_frame-track_length), example_frame+1))
#get just the tracks that are in this range
df = trackinfo[trackinfo.FRAME.isin(frame_range)]
#narrow it down to only cells that exist in the frame of interest
df = df[df.TRACK_ID.isin(df[df.FRAME==example_frame-1].TRACK_ID.unique())].reset_index(drop=True)
#open the image
im = MM_slicetostack_reader(direct, frame_range[-1], fullimshape[-3:], range(fullimshape[-3]))
#crop image based on this box
cbox  = np.array([65, 113, 495,600,150,50])#zbottom ztop left right bottom top
cropim = im[cbox[0]:cbox[1], cbox[-1]:cbox[-2], cbox[-4]:cbox[-3]].copy()
#get maximum projections
xymaxproj = np.max(cropim, axis=0)
xzmaxproj = np.max(cropim, axis=1)
yzmaxproj = np.max(cropim, axis=2)
#flip the yx projection to be portrait orientation
yzmaxproj = np.rot90(yzmaxproj, 3,axes=(1,0))
yzmaxproj = np.flip(yzmaxproj, axis=0)
#adjust brightness and contrast
# xymaxproj_bc = pre_processing_utils.intensity_normalization(xymaxproj, [0.5,10])
# xzmaxproj_bc = pre_processing_utils.intensity_normalization(xzmaxproj, [0.5,10])
# yzmaxproj_bc = pre_processing_utils.intensity_normalization(yzmaxproj, [0.5,10])
# #renormalize and invert because for some reason min and max aren't exactly zero
# xymaxproj_bc = ((xymaxproj_bc-xymaxproj_bc.min())/(xymaxproj_bc-xymaxproj_bc.min()).max())
# xzmaxproj_bc = ((xzmaxproj_bc-xzmaxproj_bc.min())/(xzmaxproj_bc-xzmaxproj_bc.min()).max())
# yzmaxproj_bc = ((yzmaxproj_bc-xzmaxproj_bc.min())/(yzmaxproj_bc-yzmaxproj_bc.min()).max())
#set min to zero
xymaxproj_bc = xymaxproj-xymaxproj.min()
xzmaxproj_bc = xzmaxproj-xzmaxproj.min()
yzmaxproj_bc = yzmaxproj-yzmaxproj.min()
#set 1 to a good value
maxval = 1000
xymaxproj_bc = xymaxproj_bc/maxval
xymaxproj_bc[xymaxproj_bc>1] = 1
xzmaxproj_bc = xzmaxproj_bc/maxval
xzmaxproj_bc[xzmaxproj_bc>1] = 1
yzmaxproj_bc = yzmaxproj_bc/maxval
yzmaxproj_bc[yzmaxproj_bc>1] = 1

#invert the colors
xymaxproj_bc = abs(xymaxproj_bc - 1)
xzmaxproj_bc = abs(xzmaxproj_bc - 1)
yzmaxproj_bc = abs(yzmaxproj_bc - 1)


# Create a figure
fig = plt.figure()#figsize=(7, 7))
# Create a GridSpec with 2 rows and 2 columns
gs = gridspec.GridSpec(2, 2,
                       height_ratios=[cropim.shape[-2]*xyres, cropim.shape[-3]*zstep],
                       width_ratios=[cropim.shape[-1]*xyres, cropim.shape[-3]*zstep],
                        hspace = 0.05,
                        wspace = -0.34,
                       figure=fig)
# fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
# Use first column for both subplots
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])#, sharex=ax1)
ax3 = fig.add_subplot(gs[1,0])

#plot images with the right proportions (basically turn the axes to microns)
ax1.imshow(xymaxproj_bc, cmap='gray',extent = [0, cropim.shape[-1]*xyres, 0, cropim.shape[-2]*xyres])
ax2.imshow(yzmaxproj_bc, cmap='gray', extent = [0, cropim.shape[-3]*zstep, 0, cropim.shape[-2]*xyres])
ax3.imshow(xzmaxproj_bc, cmap='gray', extent = [0, cropim.shape[-1]*xyres, 0, cropim.shape[-3]*zstep])

#make the color map to interpolate with 
cmap = cm.get_cmap('rainbow')
#get normalizer for the frame range
norm = Normalize(vmin=0, vmax=len(frame_range))

cell = df[df.TRACK_ID == 9].copy()

cell = cell.sort_values('FRAME').reset_index(drop=True)
frames = cell.FRAME.values
traj = cell[['POSITION_X','POSITION_Y','POSITION_Z']].values
#subtract crop starting positions from the trajectory
adjtraj = traj-np.array([cbox[-4],cbox[-1],cbox[0]])
#change to microns
adjtraj = adjtraj*[xyres,xyres,zstep]
#interpolate based on path
tck, b = interpolate.splprep(adjtraj.T, u=range(len(adjtraj)),k=1, s=0)

#measure the trajectory and interpolate evenly by distance (mostly from DetailedBalance)
interlist = []
for t in range(len(adjtraj)-1):
    di = distance.pdist([adjtraj[t,:],adjtraj[t+1,:]])[0]
    intt = round(di/0.1)
    if intt>0:
        interpoints = np.linspace(start=t, stop = t+1, num = intt, endpoint = True)
        x, y, z = interpolate.splev(interpoints,tck)
        fr = [frames[t]]*len(interpoints)
        interlist.append(np.stack([fr,x,y,z,interpoints]).T)
intarr = np.concatenate(interlist)

#plot tracks
ax1.scatter(intarr[:,1], abs(intarr[:,2]-cropim.shape[-2]*xyres), s = 4, color = cmap(norm(intarr[:,-1]))[:,:3], alpha = 0.1)
ax2.scatter(intarr[:,3], abs(intarr[:,2]-cropim.shape[-2]*xyres), s = 4, color = cmap(norm(intarr[:,-1]))[:,:3], alpha = 0.1)
ax3.scatter(intarr[:,1], cropim.shape[-3]*zstep-intarr[:,3], s = 4, color = cmap(norm(intarr[:,-1]))[:,:3], alpha = 0.1)

### add scale bars
scalebar_length = 10
scalebar_x_displacement = cropim.shape[-1]*xyres-scalebar_length-4
scalebar_y_displacement = 5.5
scalebar_z_displacement = cropim.shape[-3]*zstep-scalebar_length-4
ax1.plot([scalebar_x_displacement+scalebar_length, scalebar_x_displacement],
        [scalebar_y_displacement, scalebar_y_displacement],
        lw = 3,
        color = 'black',
        zorder=1)
#scalebar text
ax1.text(scalebar_x_displacement-2,
        scalebar_y_displacement - 4.5,
        f'{scalebar_length} μm',
        color = 'black',
        fontdict = {'fontsize': 16},
        zorder = 1)

# ax2.plot([scalebar_z_displacement+scalebar_length, scalebar_z_displacement],
#         [scalebar_y_displacement, scalebar_y_displacement],
#         lw = 3,
#         color = 'white',
#         zorder=1)
# ax3.plot([scalebar_x_displacement+scalebar_length, scalebar_x_displacement],
#         [scalebar_y_displacement, scalebar_y_displacement],
#         lw = 3,
#         color = 'white',
#         zorder=1)

    

ax1.axis('off')
ax2.axis('off')
ax3.axis('off')


plt.tight_layout()




plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')