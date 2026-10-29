# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:35:03 2025

@author: Aaron
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse, Rectangle
from neutrophil_shape.config.loader import load_config


# inverse scale for arrows
scale = 0.0012

#get directories and open separated datasets
treatments = ['Random']
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'
whichpcs = (1,2)
pc_combos = config.common.pc_combos
origin = config.db_params.origins[pc_combos.index(whichpcs)]
ntrans = config.db_params.ntrans
nbins = config.db_params.nbins
time_interval = config.im_params.time_interval #sec/frame


#get directories and open separated datasets
savedir = config.common.savedir
datadir = savedir.joinpath('shape_data')
dbdir = savedir.joinpath('detailed_balance')
dbbsdir = dbdir / 'separatedatabs'

#open the centers of the binned PCs
centers = pd.read_csv(datadir.joinpath('PC_bin_centers.csv'), index_col=0)



######## open all of the data
########### interpolate all transitions so that only individual transitions are made ###########
transdf = pd.read_csv(dbdir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_interpolated_transitions.csv'), index_col=0)
#limit to treatments
transdf = transdf[transdf.Treatment.isin(treatments)].copy()
############## get the counts of cells leaving 
trans_rate_df = pd.read_csv(dbdir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_binned_transition_rates.csv'), index_col=0)
#limit to treatments
trans_rate_df = trans_rate_df[trans_rate_df.Treatment.isin(treatments)].copy()
############# open average bootstrapped currents ###################
bsfield_sep = pd.read_csv(dbbsdir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_bootstrapped_{ntrans}_transitions_average_currents.csv'), index_col=0)
#limit to treatments
bsfield_sep = bsfield_sep[bsfield_sep.Treatment.isin(treatments)].copy()


########## PLOT OF PDF WITH AVERAGE FLUX AND ERROR  ################

fig, ax = plt.subplots(figsize=(10,10))
cbar_ax = fig.add_axes([0.94, .188, .025, .661])


########### PDFs AND PROBABILITY FLUX OF THE SEPARATED MIGRATION MODES #############

################ heatmap of probability density #############
#make numpy array with heatmap data
bighm = np.zeros((nbins,nbins))
#get total time observed in the system
ttot = transdf.time_elapsed.sum()
for x in range(nbins):
    for y in range(nbins):
        current = transdf[(transdf['from_x'] == x+1) & (transdf['from_y'] == y+1)]
        if current.empty:
            bighm[y,x] = 0
        else:
            bighm[y,x] = current.time_elapsed.sum()/ttot
            
            
#plot heatmap with seaborn
sns.heatmap(
    bighm,
    vmin=0, vmax=np.percentile(bighm, 99), #center=0,
    cmap='rocket',
    square=True,
    xticklabels = True,
    yticklabels = True,
    ax = ax,
    cbar_ax = cbar_ax,
    zorder = 1
)


######################### vector map of probability flux ################
#combine relevant data
elldf = trans_rate_df.merge(bsfield_sep, on = ['x','y'])
for x in range(1,nbins+1):
    for y in range(1,nbins+1):
        current = elldf[(elldf['x'] == x) & (elldf['y'] == y)]
        if current.empty:
            xcurrent = 0
            ycurrent = 0
            eh = 0
            ew = 0
            evec = [0,0]
        else:
            xcurrent = ((current.x_plus_rate - current.x_minus_rate)/2).iloc[0]
            ycurrent = ((current.y_plus_rate - current.y_minus_rate)/2).iloc[0]
            eh = np.sqrt(abs(current.eval2.iloc[0]))*(2/scale)
            ew = np.sqrt(abs(current.eval1.iloc[0]))*(2/scale)
            evec = current[['evec1x','evec1y']].values[0]
            evec = evec if evec[1]>0 else -evec
        ax.quiver(x-0.5,
                    y-0.5, 
                    xcurrent,
                    ycurrent,
                  angles = 'xy',
                  scale_units = 'xy',
                  scale = scale,
                  color = 'white',
                  zorder = 3)


        #determine ellipse width, height and angle
        #always set eval1 to width and adjust angle accordingly
        ex = xcurrent*(1/scale)
        ey = ycurrent*(1/scale)
        eang = np.degrees(np.arctan2(evec[1],evec[0]))

        ell = Ellipse(xy=(x-0.5+ex,y-0.5+ey),
                      width=ew,
                      height=eh,
                      angle=eang,
                      color = 'lightblue',
                      alpha = 0.12,
                      zorder = 2)
        ax.add_artist(ell)




#restore low-high yaxis orientation
ax.invert_yaxis()

##### draw a little green dot for the origin
ax.scatter(origin[0]-0.5, origin[1]-0.5, s = 90, color = '#11bd20', zorder=2)


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
