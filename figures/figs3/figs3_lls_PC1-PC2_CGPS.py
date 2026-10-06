# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:35:03 2025

@author: Aaron
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse, Rectangle
from pathlib import Path
from neutrophil_shape.config.loader import load_config
from neutrophil_shape.CustomFunctions.DetailedBalance import transition_count_wrapper
from neutrophil_shape.CustomFunctions.utils import whichpc_string

####### load common directories and data
# inverse scale for flux arrows
scale = 0.0012


#which pcs to plot the cgps of
pc_list = [(1,2),(4,5),(2,8)] #PC combinations to plot



#quickly load the config for the confocal to get confocal nbins
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'
nbins = config.db_params.nbins
condatadir = config.common.savedir / 'shape_data'
centers = pd.read_csv(condatadir.joinpath('PC_bin_centers.csv'), index_col=0)


#load the config
config = load_config(microscope_type='lls')
config._alignment = 'trajectory'
## get the directories and some CGPS info
savedir = config.common.savedir
datadir = savedir.joinpath('shape_data')
dbdir = savedir.joinpath('detailed_balance')
dbbsdir = dbdir.joinpath('separatedatabs')
pc_combos = config.common.pc_combos
origins = config.db_params.origins
llsnbins  = config.db_params.nbins

### adjust nbins for plotting
nbin_diff = llsnbins - nbins
bin_trim = nbin_diff/2

for whichpcs in pc_list:
    origin = origins[pc_combos.index(whichpcs)]

    ######## open all of the data
    ########### interpolate all transitions so that only individual transitions are made ###########
    transdf_sep = pd.read_csv(dbdir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_interpolated_transitions.csv'), index_col=0)
    # ############# open average bootstrapped currents ###################
    bsfield_sep = pd.read_csv(dbbsdir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_bootstrapped_{config.db_params.ntrans}_transitions_average_currents.csv'), index_col=0)

    #### calculate the overall average transition rates
    #get total time observed in the system
    ttot = transdf_sep.time_elapsed.sum()
    ratesargs = (transdf_sep, llsnbins)
    trans_rate_df_sep = transition_count_wrapper(ratesargs, False)
    print(trans_rate_df_sep.shape)
    # combine bootstrapped error data with real transition data
    elldf = trans_rate_df_sep.merge(bsfield_sep, on = ['x','y'])
    print(elldf.shape)

    fig, ax = plt.subplots(figsize=(14,14))
    cbar_ax = fig.add_axes([0.96, .188, .025, .661])

    ttot = transdf_sep.time_elapsed.sum()
    #make numpy array with heatmap data
    bighm = np.zeros((nbins,nbins))
    #get total time observed in the system

    for x in range(nbins):
        for y in range(nbins):
            current =  transdf_sep[(transdf_sep['from_x'] == x+bin_trim) & (transdf_sep['from_y'] == y+bin_trim)]
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
            current = elldf[(elldf['x'] == x+bin_trim-1) & (elldf['y'] == y+bin_trim-1)]
            if current.empty:
                xcurrent = 0
                ycurrent = 0
                ex = 0
                ey = 0
                eh = 0
                ew = 0
                eang = 0
            else:
                xcurrent = ((current.x_plus_rate - current.x_minus_rate)/2).iloc[0]
                ycurrent = ((current.y_plus_rate - current.y_minus_rate)/2).iloc[0]
                #determine ellipse width, height and angle
                #always set eval1 to width and adjust angle accordingly
                ex = xcurrent*(1/scale)
                ey = ycurrent*(1/scale)
                eh = np.sqrt(abs(current.eval2.iloc[0]))*(2/scale)
                ew = np.sqrt(abs(current.eval1.iloc[0]))*(2/scale)
                evec = current[['evec1x','evec1y']].values[0]
                evec = evec if evec[1]>0 else -evec
                eang = np.degrees(np.arctan2(evec[1],evec[0]))

            ax.quiver(x-0.5,
                    y-0.5, 
                    xcurrent,
                    ycurrent,
                    angles = 'xy',
                    scale_units = 'xy',
                    scale = scale,
                    color = 'white',
                    zorder = 3)
    
            ell = Ellipse(xy=(x-0.5+ex,y-0.5+ey),
                            width=ew,
                            height=eh,
                            angle=eang,
                            color = 'lightblue',
                            alpha = 0.12,
                            zorder = 2)
            ax.add_artist(ell)


    #### ADD THE FLUX ORIGIN DOT
    ax.scatter(origin[0]-0.5-bin_trim, origin[1]-0.5-bin_trim, s = 160, color = '#11bd20', zorder=2)




    #         print(x, x+(xcurrent.values*scale),y,  y+(ycurrent.values*scale))
    ax.set_xlabel(f'PC{whichpcs[0]}', fontsize = 40)
    ax.xaxis.set_label_coords(0.46,-0.05)
    ax.set_ylabel(f'PC{whichpcs[1]}', fontsize = 40)
    ax.yaxis.set_label_coords(-0.05, 0.465)
    ax.set_xticks(np.arange(0.5,nbins+0.5))
    ax.set_xticklabels([round(x,1) for x in centers.PC1.to_list()], fontsize = 18)
    ax.set_yticks(np.arange(0.5,nbins+0.5))
    ax.set_yticklabels([round(x,1) for x in centers.PC7.to_list()], fontsize = 18)
    ax.set_xlim(0,nbins+1)
    ax.set_ylim(0,nbins+1)
    # adjust colorbar tick label size
    cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(),fontsize=18)
    cbar_ax.set_ylabel('Probability', fontsize = 32, rotation = -90, labelpad = 33)

    ###### flux vector scale 
    #legend background
    lxp = 0.35
    lyp = 0.35
    legh = 1.4
    legw = 3.95
    rect = Rectangle((lxp, lyp), legw, legh, linewidth=1, edgecolor='black', facecolor='#80858a')
    ax.add_patch(rect)
    rect.set_zorder(4 * 5)
    scalevalue = 0.0017
    #x-axis legend arrow
    #position of arrow in middle of box
    arrxp = lxp + legw/2 - (scalevalue/scale)/2 
    ax.quiver(arrxp,lyp+0.95,scalevalue,0,angles = 'xy',scale_units = 'xy',scale = scale,color = "white",zorder = 4 * 5)
    #x-axis legend text
    xsc = f'{scalevalue:.1e}'
    xsc = xsc.split('e')[0] + 'x10$^{' +  str(int(xsc.split('e')[1])) + '}$'
    ax.text(lxp+0.18,lyp+0.15,xsc+' $s^{-1}$', color = 'white', fontsize = 20, fontweight = 'bold',zorder = 4 * 5)


    plt.tight_layout()


    plt.savefig(__file__.split('.')[0] + whichpc_string(whichpcs) + '.png', bbox_inches='tight', dpi =500)
