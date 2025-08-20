# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 16:13:04 2025

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


treatments = ['Random']
time_interval = 5 #sec/frame
whichpcs = [1,7]
ntrans = 1

#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_Apply/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'
if not os.path.exists(savedir):
    os.makedirs(savedir)
    
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())
#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)




######## open all of the data

############## get the counts of cells leaving 
trans_rate_df_sep = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_binned_transition_rates_separated.csv', index_col=0)



fig, axes = plt.subplots(1,3,figsize=(15,5), sharey = True)

##### loop for plotting error ellipses
for n, ax in enumerate(axes):
    #set axis title
    ax.set_title(f'{n+1} transitions') if n!=0 else ax.set_title(f'{n+1} transition')
    
    ############# open average bootstrapped currents ###################
    bsfield_sep = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_bootstrapped_{n+1}_transitions_average_currents.csv', index_col=0)
    
    ########## PC1/PC7 transition with error ellipses oriented to PCs WITHOUT PC MESH SLICES ################
    
    # inverse scale for arrows
    scale = 0.0008
    
    # combine fake error data with real transition data
    elldf = bsfield_sep.merge(trans_rate_df_sep,left_on = ['x','y'], right_on = ['x','y'])
    
    #add cgps lines
    for h in np.linspace(0, nbins, nbins+1):
        ax.axhline(h, linestyle='-', color='0.9', zorder = 1) # horizontal lines
        ax.axvline(h, linestyle='-', color='0.9', zorder = 1) # vertical lines


        
    for x in range(1,nbins+1):
        for y in range(1,nbins+1):
            current = elldf[(elldf['x'] == x) & (elldf['y'] == y)]
            xcurrent = (current.x_plus_rate - current.x_minus_rate)/2
            ycurrent = (current.y_plus_rate - current.y_minus_rate)/2
    
            ell = Ellipse(xy=(x-0.5+(xcurrent.values*(1/scale)),y-0.5+(ycurrent.values*(1/scale))),
                    width=np.sqrt(abs(current.eval1)*(1/scale)) if current.evec1x.values[0] == 1 else np.sqrt(abs(current.eval2)*(1/scale)),
                      height=np.sqrt(abs(current.eval1)*(1/scale)) if current.evec1y.values[0] == 1 else np.sqrt(abs(current.eval2)*(1/scale)),
                    # angle=np.arctan2(current.evec1y,current.evec1x),
                         # edgecolor = colorlist[i],
                         # facecolor = None,
                         # fill = False,
                         # lw = 2,
                         color = '#f04a4a',
                         alpha = 0.8, 
                         zorder = 2)
            ax.add_artist(ell)
            # ell.set_alpha(0.7)


    #also plot the current arrows
    for x in range(1,nbins+1):
        for y in range(1,nbins+1):
            current = elldf[(elldf['x'] == x) & (elldf['y'] == y)]
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
                      color = '0.7',
                        zorder = 3)
        

    ax.set_aspect('equal')

    if n==0:
        ax.set_ylabel('PC7', fontsize = 25)
    ax.set_xlabel('PC1', fontsize = 25)

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    ax.set_xlim(0,nbins)
    ax.set_ylim(0,nbins)


#legend background
lxp = 0.225
lyp = 0.225
rect = Rectangle((lxp, lyp), 2.2, 2.2, linewidth=1, edgecolor='black', facecolor='#80858a')
axes[0].add_patch(rect)
rect.set_zorder(4 * 5)
#x-axis legend arrow
axes[0].quiver(lxp+0.65,lyp+0.65,1*scale,0,angles = 'xy',scale_units = 'xy',scale = scale,color = "white",zorder = 4 * 5)
#x-axis legend text
xsc = f'{(np.diff(centers.PC1.to_list()).mean()/time_interval)*scale:.1e}'
xsc = xsc.split('e')[0] + 'x10$^{' +  str(int(xsc.split('e')[1])) + '}$'
axes[0].text(lxp+0.30,lyp+0.05,xsc+' $s^{-1}$', color = 'white', fontsize = 5, fontweight = 'bold',zorder = 4 * 5)
#y-axis legend arrow
axes[0].quiver(lxp+0.69,lyp+0.65,0,1*scale,angles = 'xy',scale_units = 'xy',scale = scale,color = 'white',zorder = 4 * 5)
#y-axis legend text
ysc = f'{(np.diff(centers.PC7.to_list()).mean()/time_interval)*scale:.1e}'
ysc = ysc.split('e')[0] + 'x10$^{' +  str(int(ysc.split('e')[1])) + '}$'
axes[0].text(lxp+0.09,lyp+0.35,ysc+' $s^{-1}$', rotation = 'vertical', color = 'white', fontsize = 5, fontweight = 'bold',zorder = 4 * 5)



plt.tight_layout()


          
# custom_legend = [
#     Ellipse(xy=[0,0],width=0.5,height=0.5,color=colorlist[0], label='1'),
#     Ellipse(xy=[0,0],width=0.5,height=0.5,color=colorlist[1], label='2'),
#     Ellipse(xy=[0,0],width=0.5,height=0.5,color=colorlist[2], label='3')
# ]

# # Add custom legend to this subplot
# ax.legend(handles=custom_legend, fontsize = 16, loc='upper left', title='Transition #\nSampled',
#     title_fontsize = 16, )

    





plt.savefig(__file__.split('.')[0] + '.png', bbox_inches='tight', dpi =500)



