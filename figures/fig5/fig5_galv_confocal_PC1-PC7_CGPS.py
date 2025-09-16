# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:35:03 2025

@author: Aaron
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.transforms
import seaborn as sns
from cmocean import cm
from matplotlib.patches import Ellipse, Rectangle



#get directories and open separated datasets

ntrans = 1
treatments = ['Galvanotaxis']
time_interval = 10 #sec/frame
whichpcs = [1,7]

#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'galv/'
if not os.path.exists(savedir):
    os.makedirs(savedir)
    


#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
nbins = centers.shape[0]




######## open all of the data
########### interpolate all transitions so that only individual transitions are made ###########
transdf_sep = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_interpolated_transitions_separated.csv', index_col=0)
transdf_sep = transdf_sep[transdf_sep.Treatment == treatments[0]]
############## get the counts of cells leaving 
trans_rate_df_sep = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_binned_transition_rates_separated.csv', index_col=0)
trans_rate_df_sep = trans_rate_df_sep[trans_rate_df_sep.Treatment == treatments[0]]
############# open average bootstrapped currents ###################
bsfield_sep = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_bootstrapped_{ntrans}_transitions_average_currents.csv', index_col=0)
bsfield_sep = bsfield_sep[bsfield_sep.Treatment == treatments[0]]


########## PC1/PC7 transition with error ellipses oriented to PCs WITHOUT PC MESH SLICES ################



# inverse scale for arrows
scale = 0.0008

# combine fake error data with real transition data
elldf = bsfield_sep.merge(trans_rate_df_sep,left_on = ['x','y'], right_on = ['x','y'])



fig, ax = plt.subplots(figsize=(10,10))
#single colorbar axis
cbar_ax = fig.add_axes([.92, .1135, .042, .7545])

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
                      alpha = 0.15,
                      zorder = 2)
        ax.add_artist(ell)
        

    

#         print(x, x+(xcurrent.values*scale),y,  y+(ycurrent.values*scale))
# axis label stuff
ax.set_xlabel('PC1', fontsize = 45)
ax.set_ylabel('PC7', fontsize = 45)
ax.xaxis.set_label_coords(0.46,-0.05)
ax.set_xticks(np.arange(0.5,nbins+0.5))
ax.set_xticklabels([round(x,1) for x in centers.PC1.to_list()], fontsize = 17)
ax.set_yticks(np.arange(0.5,nbins+0.5))
ax.set_yticklabels([round(x,1) for x in centers.PC7.to_list()], fontsize = 17)
ax.set_xlim(0,nbins+1)
ax.set_ylim(0,nbins+1)
ax.set_title('Electrotaxis', fontsize = 50, loc = 'center')#,pad = -100)


# adjust colorbar tick label size
cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(),fontsize=18)
cbar_ax.set_ylabel('Probability', fontsize = 32, rotation = -90, labelpad = 33)


########## add scale for the vectors ##########
#legend background
lxp = 0.215
lyp = 0.215
rect = Rectangle((lxp, lyp), 2.3, 2.3, linewidth=1, edgecolor='black', facecolor='#80858a')
ax.add_patch(rect)
rect.set_zorder(4 * 5)
#x-axis legend arrow
ax.quiver(lxp+0.7,lyp+0.7,1*scale,0,angles = 'xy',scale_units = 'xy',scale = scale,color = "white",zorder = 4 * 5)
#x-axis legend text
xsc = f'{(np.diff(centers.PC1.to_list()).mean()/time_interval)*scale:.1e}'
xsc = xsc.split('e')[0] + 'x10$^{' +  str(int(xsc.split('e')[1])) + '}$'
ax.text(lxp+0.37,lyp+0.05,xsc+' $s^{-1}$', color = 'white', fontsize = 9.5, fontweight = 'bold',zorder = 4 * 5)
#y-axis legend arrow
ax.quiver(lxp+0.7,lyp+0.7,0,1*scale,angles = 'xy',scale_units = 'xy',scale = scale,color = 'white',zorder = 4 * 5)
#y-axis legend text
ysc = f'{(np.diff(centers.PC7.to_list()).mean()/time_interval)*scale:.1e}'
ysc = ysc.split('e')[0] + 'x10$^{' +  str(int(ysc.split('e')[1])) + '}$'
ax.text(lxp+0.11,lyp+0.36,ysc+' $s^{-1}$', rotation = 'vertical', color = 'white', fontsize = 9.5, fontweight = 'bold',zorder = 4 * 5)


plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '.png', bbox_inches='tight', dpi =500)
