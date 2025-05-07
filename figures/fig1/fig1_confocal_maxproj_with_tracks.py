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
image = os.listdir(trackdir)[2]
trackinfo = pd.read_csv(Path(trackdir,image,image+'_TrackMateLog.csv'))
'20231116_488EGFP-CAAX_3mA_37C_2'
time_interval = 10

#server directories
raw_dir = '//10.158.28.37/ExpansionHomesA/avlnas/HL60 Galv/'
direct = raw_dir +image.split('_')[0]+'/' +image+'/Default/'
fullimshape = [361,150,1024,1024]

#which frames to project and which to extract for tracks
example_frame = 92
track_length = 26
frame_range = np.array(range(int(example_frame-track_length), example_frame))
#get just the tracks that are in this range
df = trackinfo[trackinfo.FRAME.isin(frame_range)]
#narrow it down to only cells that exist in the frame of interest
df = df[df.TRACK_ID.isin(df[df.FRAME==example_frame-1].TRACK_ID.unique())].reset_index(drop=True)
#open the image
im = MM_slicetostack_reader(direct, frame_range[-1], fullimshape[-3:], range(fullimshape[-3]))
#get maximum x-y and y-z projection
xymaxproj = np.max(im, axis=0)
xzmaxproj = np.max(im, axis=1)
#adjust brightness and contrast
#get suggestion
print(pre_processing_utils.suggest_normalization_param(xymaxproj))
print(pre_processing_utils.suggest_normalization_param(xzmaxproj))
#apply the b and c
xymaxproj_bc = pre_processing_utils.intensity_normalization(xymaxproj, [0.5,10])
xzmaxproj_bc = pre_processing_utils.intensity_normalization(xzmaxproj, [0.5,10])
#renormalize and invert because for some reason min and max aren't exactly zero
xymaxproj_bc = ((xymaxproj_bc-xymaxproj_bc.min())/(xymaxproj_bc-xymaxproj_bc.min()).max())
xzmaxproj_bc = ((xzmaxproj_bc-xzmaxproj_bc.min())/(xzmaxproj_bc-xzmaxproj_bc.min()).max())


# Create a figure
fig = plt.figure()#figsize=(7, 7))
# Create a GridSpec with 2 rows and 2 columns
gs = gridspec.GridSpec(2, 1,
                       height_ratios=[im.shape[-2], im.shape[-3]],
                       wspace = 0.01,
                       hspace = 0.05,
                       figure=fig)
fig.patch.set_facecolor('black')
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
# Use first column for both subplots
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])


fig, ax = plt.subplots(1,1, figsize=(10,10))
#plot the actual image
ax.imshow(xymaxproj_bc, cmap='gray')

#make the color map to interpolate with 
cmap = cm.get_cmap('rainbow')
#get normalizer for the frame range
norm = Normalize(vmin=0, vmax=len(frame_range))


for i, cell in df.groupby('TRACK_ID'):
    cell = cell.sort_values('FRAME').reset_index(drop=True)
    frames = cell.FRAME.values
    traj = cell[['POSITION_X','POSITION_Y']].values
    if len(traj)>1:
        #interpolate based on path
        tck, b = interpolate.splprep(traj.T, u=range(len(traj)),k=1, s=0)
        
        #measure the trajectory and interpolate evenly by distance (mostly from DetailedBalance)
        interlist = []
        for t in range(len(traj)-1):
            di = distance.pdist([traj[t,:],traj[t+1,:]])[0]
            intt = round(di/0.1)
            if intt>0:
                interpoints = np.linspace(start=t, stop = t+1, num = intt, endpoint = True)
                x, y = interpolate.splev(interpoints,tck)
                fr = [frames[t]]*len(interpoints)
                interlist.append(np.stack([fr,x,y,interpoints]).T)
        intarr = np.concatenate(interlist)
    
        #plot tracks
        ax.scatter(intarr[:,1], intarr[:,2], s = 0.5, color = cmap(norm(intarr[:,-1]))[:,:3], alpha = 0.1)
    
    
# Turn off all spines and ticks
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(
    axis='both',       # Apply to both x and y axis
    which='both',      # Apply to both major and minor ticks
    bottom=False,      # Turn off bottom ticks
    top=False,         # Turn off top ticks
    left=False,        # Turn off left ticks
    right=False,       # Turn off right ticks
    labelbottom=False, # Turn off x tick labels
    labelleft=False    # Turn off y tick labels
)

#### colorbar stuff for the tracks
# cbar_ax = fig.add_axes([0.988, 0.25, 0.03, 0.5]) #vertical
cbar_ax = fig.add_axes([0.25, 0.989, 0.5, 0.03]) #horizontal
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                       cax = cbar_ax,
                        ticks = np.linspace(0, track_length, int((track_length/6)+1)),
                       orientation='horizontal')
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")
cbar.set_label("Time (min)", fontsize = 20, labelpad = 7)  # Label for colorbar
# cbar.set_ticklabels([str(int(float(x.get_text())*time_interval/60)) for x in cbar.ax.yaxis.get_ticklabels()])
cbar.set_ticklabels((np.linspace(0, track_length, int((track_length/6)+1))*time_interval/60).astype(int).astype('str'))
cbar.ax.tick_params(axis="x", labelsize = 15)#pad=-1, , rotation=-45)

plt.tight_layout()
plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')