# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:18:50 2025

@author: Aaron
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from pathlib import Path
from neutrophil_shape.config.loader import load_config

#define some angular binning stuff
binrange = 20
bin_centers = np.arange(0, 360, binrange)
bin_edges = np.unique([(round(b-binrange/2,4), round(b+binrange/2,4)) for b in bin_centers])


### get info from config
treatments = ['Random']
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'
time_interval = config.im_params.time_interval #sec/frame
whichpcs = (1,2)
nbins = config.db_params.nbins
allorigins = config.db_params.origins
pc_combos = config.common.pc_combos
fluxorigin = allorigins[pc_combos.index(whichpcs)]


#get directories and open separated datasets
basedir = config.common.savedir
datadir = basedir / 'shape_data'
transdir = basedir / 'detailed_balance'




############## get the counts of cells leaving 
trans_rate_df_sep = pd.read_csv(transdir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_binned_transition_rates_separated.csv'), index_col=0)
#limit to treatments
trans_rate_df_sep = trans_rate_df_sep[trans_rate_df_sep.Treatment.isin(treatments)]



    
########### get coordinates of color wedges using interpolated lines between the edges of each bin
actual_origin = np.array(fluxorigin)-0.5
line_arr = np.zeros((int(len(bin_edges)-1),2,2))
tri_points = np.zeros((int(360/binrange),3,2))
for i in range(len(bin_edges)-1):
    cur = bin_edges[i]
    nex = bin_edges[i+1]
    #get the point around the circle and extend the radius
    rad = -np.radians(cur+180)
    radx = np.cos(rad)*(1.5*nbins)+actual_origin[0]
    rady = np.sin(rad)*(1.5*nbins)+actual_origin[1]
    #add this to the line array
    line_arr[i,:] = [[actual_origin[0],actual_origin[1]],[radx,rady]]
    #get the next point around the circle
    nrad = -np.radians(nex+180)
    nradx = np.cos(nrad)*(1.5*nbins)+actual_origin[0]
    nrady = np.sin(nrad)*(1.5*nbins)+actual_origin[1]
    #add to the triangle point array 
    tri_points[i,:,:] = [[actual_origin[0],actual_origin[1]],[radx,rady],[nradx,nrady]]
    
###### solve lines to get the x and y points for all of the labels around the origin
leftticks = []
bottomticks = []
rightticks = []
topticks = []
tickcoords = np.zeros((len(bin_centers),2))
#labels need to be adjusted because 0 should start on the left
bclabels = bin_centers-180
bclabels[bclabels<0] = bclabels[bclabels<0]+360
for i, bc in enumerate(bin_centers):
    rad = -np.radians(bc)
    radx = np.cos(rad)+actual_origin[0]
    rady = np.sin(rad)+actual_origin[1]
    m = (actual_origin[1]-rady)/(actual_origin[0]-radx)
    b = rady - m*radx
    #in the case of 0 and 180
    if m == 0:
        if rad == 0:
            tickcoords[i] = [nbins, actual_origin[1]]
        else:
            tickcoords[i] = [0, actual_origin[1]]
    #intersections up and left
    elif abs(rad)>=np.pi and abs(rad)<np.pi*1.5:
        up = [(nbins-b)/m, nbins]
        left = [0, b]
        if up[0]>=0:
            tickcoords[i] = up
        else:
            ul = np.stack((up,left))
            #minimize y
            miny = np.where(ul[:,1] == np.min(ul[:,1]))
            tickcoords[i] = ul[miny]
    #intersections up and right
    elif abs(rad)>=np.pi*1.5:
        up = [(nbins-b)/m, nbins]
        right = [nbins, nbins*m+b]
        if up[0]<=nbins:
            tickcoords[i] = up
        else:
            ur = np.stack((up,right))
            #minimize y
            miny = np.where(ur[:,1] == np.min(ur[:,1]))
            tickcoords[i] = ur[miny]
    #intersections down and left
    elif abs(rad)<np.pi and abs(rad)>=np.pi/2:
        down = [-b/m,0]
        left = [0,b]
        if down[0]>=0:
            tickcoords[i] = down
        else:
            ul = np.stack((down,left))
            #maximize y
            maxy = np.where(ul[:,1] == np.max(ul[:,1]))
            tickcoords[i] = ul[maxy]
    #intersections down and right
    elif abs(rad)<np.pi/2:
        down = [-b/m,0]
        right = [nbins,m*nbins+b]
        if down[0]<=nbins:
            tickcoords[i] = down
        else:
            ur = np.stack((down,right))
            #minimize y
            maxy = np.where(ur[:,1] == np.max(ur[:,1]))
            tickcoords[i] = ur[maxy]
    
    ####### add tick coordinate to the correct tick list
    if tickcoords[i][0] == nbins:
        rightticks.append([bclabels[i], tickcoords[i][1]])
    elif tickcoords[i][1] == nbins:
        topticks.append([bclabels[i], tickcoords[i][0]])
    elif tickcoords[i][0] == 0:
        leftticks.append([bclabels[i], tickcoords[i][1]])
    elif tickcoords[i][1] == 0:
        bottomticks.append([bclabels[i], tickcoords[i][0]])

####sort all of the tick lists
bottomticks = np.array(bottomticks).T
topticks = np.array(topticks).T
leftticks = np.array(leftticks).T
rightticks = np.array(rightticks).T




######make the plot
fig, ax = plt.subplots(figsize=(10,10))


######### draw all of the flux arrows
# inverse scale for arrows
scale = 0.0012

for x in range(1,nbins+1):
    for y in range(1,nbins+1):
        current = trans_rate_df_sep[(trans_rate_df_sep['x'] == x) & (trans_rate_df_sep['y'] == y)]
        xcurrent = (current.x_plus_rate - current.x_minus_rate)/2
        ycurrent = (current.y_plus_rate - current.y_minus_rate)/2
        anglecolor = (np.arctan2(xcurrent,ycurrent) *180/np.pi)+180
        
        ax.quiver(x-0.5,
                    y-0.5, 
                    xcurrent,
                    ycurrent,
                  angles = 'xy',
                  scale_units = 'xy',
                  scale = scale,
#                   width = 0.012,
#                   minlength = 0.8,
                  color = [0.45,0.45,0.45, 1],
                    zorder = 2)
    


### plot the lines for the bin edges
for l in line_arr:
    l=l.T
    ax.plot(l[0],l[1], color = (0.25,0.25,0.25), lw = 1.5, zorder = 3)


### get the colormap and the number of discrete colors along it
### add one more color than needed so that colors don't start and end in the same place
cmap = cm.get_cmap('twilight', int(360/binrange+1))
discrete_colors = cmap(np.linspace(0,1,int(360/binrange+1))[:-1])
#### plot the triangles
for i,t in enumerate(tri_points):
   #make patch
   triangle = patches.Polygon(t, closed=True, color=discrete_colors[i,:-1], edgecolor='none', linewidth=0)
   # Add triangle to the plot
   ax.add_patch(triangle)
    

# ######### color bar associated with the wedge colors
# # define the bins and normalize
# bounds = np.linspace(0, 360/binrange, int((360/binrange)+1))
# norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
# cbar_ax = fig.add_axes([0.1293, 0.99, 0.85548, 0.03])
# cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
#                        cax = cbar_ax,
#                        ticks = bounds[:-1] + 0.5,
#                        orientation='horizontal')
# cbar.ax.xaxis.set_ticks_position("top")
# cbar.ax.xaxis.set_label_position("top")
# cbar.set_label("Linear Cycle Position", fontsize = 24, labelpad = 7)  # Label for colorbar
# cbar.set_ticklabels(np.arange(0,360,binrange))
# cbar.ax.tick_params(axis="x", pad=-1, labelsize = 18, rotation=-45)
# #scooch the x axis labels by a certain amount
# dx = -5/72.; dy = 0/72. 
# for tick in cbar.ax.xaxis.get_majorticklabels():
#     if len(tick.get_text())==1:
#         dx = -1/72
#     elif len(tick.get_text())==2:
#         dx = -5/72
#     elif len(tick.get_text())==3:
#         dx = -7/72
#     offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
#     tick.set_transform(tick.get_transform() + offset)

#### axis stuff
# ax.set_xlabel('PC1', fontsize = 34)
# ax.set_ylabel('PC7', fontsize = 34)
ax.set_xlim(0,nbins)
ax.set_ylim(0,nbins)



# Set bottom ticks
ax.set_xticks(bottomticks[1,:])
ax.set_xticklabels([str(int(x))+'°' for x in  bottomticks[0,:]], fontsize = 26)

# Set left ticks
ax.set_yticks(leftticks[1,:])
ax.set_yticklabels([str(int(x))+'°' for x in  leftticks[0,:]], fontsize = 26)

# Add top axis with unique ticks
top_ax = ax.secondary_xaxis('top')
top_ax.set_xticks(topticks[1,:])
top_ax.set_xticklabels([str(int(x))+'°' for x in  topticks[0,:]], fontsize = 26)

# Add right axis with unique ticks
right_ax = ax.secondary_yaxis('right')
right_ax.set_yticks(rightticks[1,:])
right_ax.set_yticklabels([str(int(x))+'°' for x in  rightticks[0,:]], fontsize = 26)





plt.savefig(__file__.split('.')[0] + '.png', bbox_inches='tight', dpi =500)
