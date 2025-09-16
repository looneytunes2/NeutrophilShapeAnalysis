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
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arrow
import multiprocessing

def format_seconds(seconds):
    sec = abs(seconds)
    minutes = int(sec // 60)
    secs = int(sec % 60)
    return f"{minutes:02}:{secs:02}" if seconds>0 else f"-{minutes:02}:{secs:02}"

#open tracking data
basedir = 'E:/Aaron/Galvanotaxis_Confocal_40x_37C_10s/'
trackdir = basedir + 'Tracking_Images/'
imagename = os.listdir(trackdir)[1]
df = pd.read_csv(Path(trackdir,imagename,imagename+'_TrackMateLog.csv'))
time_interval = 10



#server directories
raw_dir = '//10.158.28.37/ExpansionHomesA/avlnas/HL60 Galv/'
direct = raw_dir +imagename.split('_')[0]+'/' +imagename+'/Default/'
fullimshape = [361,150,1024,1024]

#create args
args = []
#start at the frame 5 minutes before the electric field
startframe = int(181-(5*60/time_interval))
for fr in range(startframe, fullimshape[0]):
    args.append([direct, fr, fullimshape[-3:], range(fullimshape[-3])])
#open the  full image
with multiprocessing.Pool(processes=60) as pool:
    result = pool.starmap(MM_slicetostack_reader, args)
#get the maximum projection of the full movie
maxproj = np.zeros((len(range(startframe, fullimshape[0])), fullimshape[-2], fullimshape[3]))
for i,r in enumerate(result):
    maxproj[i] = np.max(r, axis = 0)



#scale bar info
scalebar_x_displacement = fullimshape[-1]-30
scalebar_y_displacement = fullimshape[-2]-45
scalebar_length = 20
resolution = 0.3394 #um / pixel





mesh_image_interval = 3 #frame interval between meshes in the mesh rendering for time bar

#adjust brightness and contrast
#set min to zero
maxproj_bc = maxproj-maxproj.min()
#set 1 to a good value
maxval = 1000
maxproj_bc = maxproj_bc/maxval
maxproj_bc[maxproj_bc>1] = 1

#invert the colors
invert = abs(maxproj_bc - 1)


#set the range of frames to calculate the track worm
frame_range = 12


#make the color map to interpolate with 
cmap = cm.get_cmap('rainbow')
#get normalizer for the frame range
norm = Normalize(vmin=0, vmax=frame_range)


#calculate track worms
wormlist = []
for pf, f in enumerate(range(startframe, fullimshape[0])):
    frame_window = np.arange(max(f-frame_range, 0), f+1)
    fdf = df.loc[df.FRAME.isin(frame_window)]
    for i, cell in fdf.groupby('TRACK_ID'):
        cell = cell.sort_values('FRAME').reset_index(drop=True)
        frames = cell.FRAME.values
        traj = cell[['POSITION_X','POSITION_Y']].values
        if len(traj)>1:
            #interpolate based on path
            tck, b = interpolate.splprep(traj.T, u=np.arange(len(traj)), k=1, s=0)
            
            #measure the trajectory and interpolate evenly by distance (mostly from DetailedBalance)
            interlist = []
            for t in range(len(traj)-1):
                di = distance.pdist([traj[t,:],traj[t+1,:]])[0]
                intt = round(di/0.1)
                if intt>0:
                    interpoints = np.linspace(start=t, stop = t+1, num = intt, endpoint = True)
                    x, y = interpolate.splev(interpoints,tck)
                    fr = [pf]*len(interpoints)
                    interlist.append(np.stack([fr,x,y,interpoints]).T)
                    wormlist.append({
                        'frame': [f]*len(x),
                        'TRACK_ID': [i]*len(x),
                        'point_frame': fr,
                        'x': x,
                        'y': y,
                        'color_interp': interpoints,
                        })

wormdf = pd.DataFrame(wormlist)
wormdf = wormdf.explode(wormdf.columns.to_list()).reset_index(drop = True)




fig, ax = plt.subplots(1,1, figsize=(10,10))

#plot the actual image
imsh = ax.imshow(invert[0], cmap='gray', zorder = 0)


### time label
#get times for the timestamp
times = wormdf.frame.sort_values().unique()*time_interval-(181*time_interval)
mstimes = [format_seconds(x) for x in times]
timer = ax.text(3,40,mstimes[0], color = 'black', fontdict = {'fontsize': 24})


#create EF labels and colors
eflabels = [['EF OFF','#9e36ba'] if x<181 else ['EF ON','#2ab54f'] for x in range(startframe, fullimshape[0]) ]
EF = ax.text(3, fullimshape[-2]-5, eflabels[0][0], color = eflabels[0][1], fontdict = {'fontsize': 24})
EFarrow = Arrow(135, fullimshape[-2]-70, -120, 0, width = 70, color = '#2ab54f', visible = False)
ax.add_patch(EFarrow)

#plot tracks
worms = None #ax.scatter([], [], s = 1.2, color = cmap(norm([]))[:,:3], alpha = 0.1, zorder = 1)
     
    
#scalebar
sb = ax.plot([scalebar_x_displacement-(scalebar_length/resolution), scalebar_x_displacement],
        [scalebar_y_displacement, scalebar_y_displacement],
        lw = 5,
        color = 'black',
        zorder=1)

#scalbar text
sb_label = ax.text(scalebar_x_displacement-(scalebar_length/resolution)-17,
                    scalebar_y_displacement + 35,
                    f'{scalebar_length} μm',
                    color = 'black',
                    fontdict = {'fontsize': 16},
                    zorder = 1)


#turn off all axis related stuff
ax.axis('off')


#### colorbar stuff for the tracks
# cbar_ax = fig.add_axes([-0.02, 0.25, 0.024, 0.50]) #vertical
cbar_ax = fig.add_axes([0.25, 0.89, 0.5, 0.015]) #horizontal
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                       cax = cbar_ax,
                        ticks = np.arange(0, frame_range+1, mesh_image_interval),
                       orientation='horizontal')
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")
cbar.set_label("Time (min)", fontsize = 18, labelpad = 7)  # Label for colorbar
#invert the time labels
cbar.set_ticklabels((np.arange(-frame_range, mesh_image_interval, mesh_image_interval)*time_interval/60).astype('str'))
cbar.ax.tick_params(axis="x", labelsize = 12)#pad=-1, , rotation=-45)






# make function for updating point position
def animate(i,):
    
    #set the current set of data
    imsh.set_data(invert[i])
    imsh.set_zorder(0)


    #update worms
    global worms
    if worms is not None:
        worms.remove()
    if i>0:
        currentworms = wormdf[wormdf.point_frame==i]
        worms = ax.scatter(currentworms.x, currentworms.y, s = 1.2,
                       color = cmap(norm(currentworms.color_interp.astype(float)))[:,:3],
                       alpha = 0.1, zorder = 1)
    else:
        worms = None


    #timer animation
    timer.set_text(mstimes[i])

    #EF label
    EF.set_text(eflabels[i][0])
    EF.set_color(eflabels[i][1])
    if i==181-startframe:
        EFarrow.set_visible(True)
    
    if worms is not None:
        return imsh, timer, EF, EFarrow, worms
    else:
        return imsh, timer, EF, EFarrow

#add two to the frame count to adjust the range function and to add a blank frame at the beginning
ani = FuncAnimation(fig, animate, interval=10, repeat=True,
                    frames=len(maxproj),)

# ani.save('C:/Users/Aaron/Desktop/' + imagename + '_animated.mp4', fps=10, dpi = 200)#, extra_args=['-vcodec', 'libx264'])


ani.save(__file__.split('.')[0]  + '.mp4', fps=10, dpi = 200)#, extra_args=['-vcodec', 'libx264'])


