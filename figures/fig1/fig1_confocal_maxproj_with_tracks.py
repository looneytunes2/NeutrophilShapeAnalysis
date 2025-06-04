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



#scale bar info
scalebar_x_displacement = fullimshape[-1]-30
scalebar_y_displacement = fullimshape[-2]-45
scalebar_length = 20
resolution = 0.3394 #um / pixel


#which frames to project and which to extract for tracks
example_frame = 90
track_length = 24
mesh_image_interval = 3 #frame interval between meshes in the mesh rendering for time bar
frame_range = np.array(range(int(example_frame-track_length), example_frame+1))
#get just the tracks that are in this range
df = trackinfo[trackinfo.FRAME.isin(frame_range)]
#narrow it down to only cells that exist in the frame of interest
df = df[df.TRACK_ID.isin(df[df.FRAME==example_frame-1].TRACK_ID.unique())].reset_index(drop=True)
#open the image
im = MM_slicetostack_reader(direct, frame_range[-1], fullimshape[-3:], range(fullimshape[-3]))
#get maximum x-y and y-z projection
xymaxproj = np.max(im, axis=0)
#adjust brightness and contrast
#set min to zero
xymaxproj_bc = xymaxproj-xymaxproj.min()
#set 1 to a good value
maxval = 1000
xymaxproj_bc = xymaxproj_bc/maxval
xymaxproj_bc[xymaxproj_bc>1] = 1




fig, ax = plt.subplots(1,1, figsize=(10,10))
#plot the actual image
ax.imshow(xymaxproj_bc, cmap='gray', zorder = 0)

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
        ax.scatter(intarr[:,1], intarr[:,2], s = 1.2, color = cmap(norm(intarr[:,-1]))[:,:3], alpha = 0.1, zorder = 1)
 
    
#scalebar
ax.plot([scalebar_x_displacement-(scalebar_length/resolution), scalebar_x_displacement],
        [scalebar_y_displacement, scalebar_y_displacement],
        lw = 5,
        color = 'white',
        zorder=1)
#scalbar text
ax.text(scalebar_x_displacement-(scalebar_length/resolution)-13,
                    scalebar_y_displacement + 30,
                    f'{scalebar_length} μm',
                    color = 'white',
                    fontdict = {'fontsize': 18},
                    zorder = 1)

#box around cell of interest
boxcolor = '#e34444'
cbox  = np.array([495,600,150,50])#left right bottom top
ax.plot([cbox[0],cbox[0]],[cbox[2],cbox[3]], color = boxcolor)
ax.plot([cbox[1],cbox[1]],[cbox[2],cbox[3]], color = boxcolor)
ax.plot([cbox[0],cbox[1]],[cbox[2],cbox[2]], color = boxcolor)
ax.plot([cbox[0],cbox[1]],[cbox[3],cbox[3]], color = boxcolor)   
    
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
cbar_ax = fig.add_axes([-0.01, 0.049, 0.018, 0.905]) #vertical
# cbar_ax = fig.add_axes([0.25, 0.989, 0.5, 0.03]) #horizontal
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                       cax = cbar_ax,
                        ticks = np.linspace(0, track_length, int((track_length/6)+1)),
                       orientation='vertical')
cbar.ax.yaxis.set_ticks_position("left")
cbar.ax.yaxis.set_label_position("left")
cbar.set_label("Time (min)", fontsize = 26, labelpad = 7)  # Label for colorbar
# cbar.set_ticklabels([str(int(float(x.get_text())*time_interval/60)) for x in cbar.ax.yaxis.get_ticklabels()])
cbar.set_ticks(np.arange(0, track_length+1, mesh_image_interval))
cbar.set_ticklabels((np.arange(0, track_length+1, mesh_image_interval)*time_interval/60).astype('str'))
cbar.ax.tick_params(axis="y", labelsize = 15)#pad=-1, , rotation=-45)

plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')