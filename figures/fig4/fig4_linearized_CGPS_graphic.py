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
from neutrophil_shape.config.loader import load_config

#where the linear cycle starts (0 degrees); it runs clockwise from there
start = 'right' #'top', 'right', 'bottom' or 'left'
#direction of the start, in degrees counterclockwise from the +x axis
start_angle = {'right': 0, 'top': 90, 'left': 180, 'bottom': 270}[start]

#define some angular binning stuff
nbins_angle = 18
binrange = 360/nbins_angle # degrees / bin
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
trans_rate_df_sep = pd.read_csv(transdir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_binned_transition_rates.csv'), index_col=0)
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
    rad = np.radians(start_angle-cur)
    radx = np.cos(rad)*(1.5*nbins)+actual_origin[0]
    rady = np.sin(rad)*(1.5*nbins)+actual_origin[1]
    #add this to the line array
    line_arr[i,:] = [[actual_origin[0],actual_origin[1]],[radx,rady]]
    #get the next point around the circle
    nrad = np.radians(start_angle-nex)
    nradx = np.cos(nrad)*(1.5*nbins)+actual_origin[0]
    nrady = np.sin(nrad)*(1.5*nbins)+actual_origin[1]
    #add to the triangle point array 
    tri_points[i,:,:] = [[actual_origin[0],actual_origin[1]],[radx,rady],[nradx,nrady]]
    
###### solve lines to get the x and y points for all of the labels around the origin
leftticks = []
bottomticks = []
rightticks = []
topticks = []
ox, oy = actual_origin
for bc in bin_centers:
    #direction of this bin center out of the origin
    rad = np.radians(start_angle-bc)
    dx = np.cos(rad)
    dy = np.sin(rad)
    #distance along that line to each side of the plot it is heading towards
    #(in the order the sides win a tie)
    hits = []
    if dx > 1e-12:
        hits.append(((nbins-ox)/dx, 'right'))
    if dy > 1e-12:
        hits.append(((nbins-oy)/dy, 'top'))
    if dx < -1e-12:
        hits.append(((0-ox)/dx, 'left'))
    if dy < -1e-12:
        hits.append(((0-oy)/dy, 'bottom'))
    #the line leaves the plot through the nearest of them
    dist, side = min(hits, key=lambda h: h[0])

    ####### add the label and its coordinate to the correct tick list
    if side == 'right':
        rightticks.append([bc, oy+dist*dy])
    elif side == 'top':
        topticks.append([bc, ox+dist*dx])
    elif side == 'left':
        leftticks.append([bc, oy+dist*dy])
    else:
        bottomticks.append([bc, ox+dist*dx])

####turn all of the tick lists into [label, coordinate] arrays (empty if no line leaves through that side)
bottomticks = np.array(bottomticks).reshape(-1,2).T
topticks = np.array(topticks).reshape(-1,2).T
leftticks = np.array(leftticks).reshape(-1,2).T
rightticks = np.array(rightticks).reshape(-1,2).T




######make the plot
fig, ax = plt.subplots(figsize=(10,10))


######### draw all of the flux arrows
# inverse scale for arrows
scale = 0.0012

for x in range(1,nbins+1):
    for y in range(1,nbins+1):
        current = trans_rate_df_sep[(trans_rate_df_sep['x'] == x) & (trans_rate_df_sep['y'] == y)]
        if current.empty:
            xcurrent = 0
            ycurrent = 0
        else:
            xcurrent = (current.x_plus_rate - current.x_minus_rate)/2
            ycurrent = (current.y_plus_rate - current.y_minus_rate)/2

        
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
cmap = cm.twilight
discrete_colors = cmap(np.linspace(0, 1, nbins_angle, endpoint=False))
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
