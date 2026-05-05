# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:35:03 2025

@author: Aaron
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.transforms
import seaborn as sns
from cmocean import cm
from matplotlib.patches import Ellipse, Rectangle
from neutrophil_shape.config.loader import load_config
from neutrophil_shape.CustomFunctions import linear_cycle_utils, utils


#get directories and open separated datasets
treatments = ['Random']
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'
whichpcs = (1,2)
pc_combos = config.common.pc_combos
origin = config.db_params.origins[pc_combos.index(whichpcs)]
binrange = 20
direction = 'clockwise'
zerostart = 'left'
ntrans = config.db_params.ntrans
time_interval = config.im_params.time_interval #sec/frame


#get directories and open separated datasets
basedir = config.common.savedir
datadir = basedir.joinpath('shape_data')
savedir = basedir.joinpath('detailed_balance')


FullFrame = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col=0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())
#open the centers of the binned PCs
centers = pd.read_csv(datadir.joinpath('PC_bin_centers.csv'), index_col=0)



######## open all of the data
########### interpolate all transitions so that only individual transitions are made ###########
transdf_sep = pd.read_csv(savedir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_interpolated_transitions_separated.csv'), index_col=0)
#limit to treatments
transdf_sep = transdf_sep[transdf_sep.Treatment.isin(treatments)].copy()
############## get the counts of cells leaving 
trans_rate_df_sep = pd.read_csv(savedir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_binned_transition_rates_separated.csv'), index_col=0)
#limit to treatments
trans_rate_df_sep = trans_rate_df_sep[trans_rate_df_sep.Treatment.isin(treatments)].copy()
############# open average bootstrapped currents ###################
bsfield_sep = pd.read_csv(savedir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_bootstrapped_{ntrans}_transitions_average_currents.csv'), index_col=0)
#limit to treatments
bsfield_sep = bsfield_sep[bsfield_sep.Treatment.isin(treatments)].copy()


# inverse scale for arrows
scale = 0.0012

# combine fake error data with real transition data
elldf = bsfield_sep.merge(trans_rate_df_sep,left_on = ['x','y'], right_on = ['x','y'])




########## PC1/PC7 transition with error ellipses oriented to PCs WITHOUT PC MESH SLICES ################

fig, ax = plt.subplots(figsize=(10,10))
cbar_ax = fig.add_axes([0.94, .188, .025, .661])

ttot = transdf_sep.time_elapsed.sum()
#make numpy array with heatmap data
bighm = np.zeros((nbins,nbins))
#get total time observed in the system

for x in range(nbins):
    for y in range(nbins):
        current =  transdf_sep[(transdf_sep['from_x'] == x+1) & (transdf_sep['from_y'] == y+1)]
        if current.empty:
            bighm[y,x] = 0
        else:
            bighm[y,x] = current.time_elapsed.sum()/ttot
#plot heatmap with seaborn
sns.heatmap(
    bighm,
    vmin=0, vmax=bighm.max(), #center=0,
    cmap='rocket',
    square=True,
    xticklabels = True,
    yticklabels = True,
    ax = ax,
#     cbar=i==0,
    cbar_ax = cbar_ax,
#         cbar_kws=cbar_kws
)


    
for x in range(1,nbins+1):
    for y in range(1,nbins+1):
        current = elldf[(elldf['x'] == x) & (elldf['y'] == y)]
        xcurrent = (current.x_plus_rate - current.x_minus_rate)/2
        ycurrent = (current.y_plus_rate - current.y_minus_rate)/2

        #add flux current arrow        
        ax.quiver(x-0.5,
                   y-0.5, 
                   xcurrent,
                   ycurrent,
                  angles = 'xy',
                  scale_units = 'xy',
                  scale = scale,
                  color = 'white',
                    zorder = 3 * 5)


        #determine ellipse width, height and angle
        #always set eval1 to width and adjust angle accordingly
        eh = np.sqrt(abs(current.eval2))*(2/scale)
        ew = np.sqrt(abs(current.eval1))*(2/scale)
        evec = current[['evec1x','evec1y']].values[0]
        eang = np.degrees(np.arctan2(evec[1],evec[0]))
        
        #define the error oval
        ell = Ellipse(xy=(x-0.5+(xcurrent.values*(1/scale)),y-0.5+(ycurrent.values*(1/scale))),
                      width=ew,
                      height=eh,
                      angle=eang,
                      color = 'lightblue',
                      alpha = 0.10,
                      zorder = 2)
        ax.add_artist(ell)
        

    
   
# #plot the origin dot
# ax.scatter(7-0.5, 7-0.5, s = 200, color = '#4481e3', zorder=2)

    

#         print(x, x+(xcurrent.values*scale),y,  y+(ycurrent.values*scale))
ax.set_xlabel(f'PC{whichpcs[0]}', fontsize = 40)
ax.xaxis.set_label_coords(0.46,-0.05)
ax.set_ylabel(f'PC{whichpcs[1]}', fontsize = 40)
ax.yaxis.set_label_coords(-0.05, 0.465)
ax.set_xticks(np.arange(0.5,nbins+0.5))
ax.set_xticklabels([round(x,1) for x in centers[f'PC{whichpcs[0]}'].to_list()], fontsize = 18)
ax.set_yticks(np.arange(0.5,nbins+0.5))
ax.set_yticklabels([round(x,1) for x in centers[f'PC{whichpcs[1]}'].to_list()], fontsize = 18)
ax.set_xlim(0,nbins+1)
ax.set_ylim(0,nbins+1)
# adjust colorbar tick label size
cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(),fontsize=18)
cbar_ax.set_ylabel('Probability', fontsize = 32, rotation = -90, labelpad = 33)

#legend background
lxp = 0.35
lyp = 0.35
legh = 1.15
legw = 3.0
rect = Rectangle((lxp, lyp), legw, legh, linewidth=1, edgecolor='black', facecolor='#80858a')
ax.add_patch(rect)
rect.set_zorder(4 * 5)
scalevalue = 0.0017
#x-axis legend arrow
#position of arrow in middle of box
arrxp = lxp + legw/2 - (scalevalue/scale)/2 
ax.quiver(arrxp,lyp+0.8,scalevalue,0,angles = 'xy',scale_units = 'xy',scale = scale,color = "white",zorder = 4 * 5)
#x-axis legend text
xsc = f'{scalevalue:.1e}'
xsc = xsc.split('e')[0] + 'x10$^{' +  str(int(xsc.split('e')[1])) + '}$'
ax.text(lxp+0.1,lyp+0.15,xsc+' $s^{-1}$', color = 'white', fontsize = 16, fontweight = 'bold',zorder = 4 * 5)


plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '.png', bbox_inches='tight', dpi =500)
