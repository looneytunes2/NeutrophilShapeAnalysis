# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:10:43 2025

@author: Aaron
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse, Rectangle


#get directories and open separated datasets

treatments = ['DMSO','CK666']
time_interval = 10 #sec/frame
whichpcs = [1,7]

basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'CK666/'
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())
#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
#limit data to the CK666 experiments
TotalFrame = FullFrame[FullFrame.Experiment == 'Drug']
dates = [20240610,20240617,20240620,20241205,20241209]
TotalFrame = TotalFrame[TotalFrame.Date.isin(dates)]
TotalFrame['Treatment'] = pd.Categorical(TotalFrame.Treatment.to_list(), categories=treatments, ordered=True)



######## open all of the data
########### interpolate all transitions so that only individual transitions are made ###########
transdf_sep = pd.read_csv(savedir+f'interpolated_PC{whichpcs[0]}-PC{whichpcs[1]}_transitions_separated.csv', index_col=0)
#ensure that DMSO is the first in order
transdf_sep['Treatment'] = pd.Categorical(transdf_sep.Treatment, categories=treatments, ordered=True)
transdf_sep = transdf_sep.sort_values(by='Treatment')
############## get the counts of cells leaving 
trans_rate_df_sep = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_binned_transition_rates_separated.csv', index_col=0)
#ensure that DMSO is the first in order
trans_rate_df_sep['Treatment'] = pd.Categorical(trans_rate_df_sep.Treatment, categories=treatments, ordered=True)
trans_rate_df_sep = trans_rate_df_sep.sort_values(by='Treatment')
############# open average bootstrapped currents ###################
bsfield_sep = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_bootstrapped_transitions_average_currents.csv', index_col=0)
#ensure that DMSO is the first in order
bsfield_sep['Treatment'] = pd.Categorical(bsfield_sep.Treatment, categories=treatments, ordered=True)
bsfield_sep = bsfield_sep.sort_values(by='Treatment')



########### PDFs AND PROBABILITY FLUX OF THE SEPARATED Treatments #############

# inverse scale for arrows
scale = 0.0008


# combine error data with real transition data
elldf = bsfield_sep.merge(trans_rate_df_sep,left_on = ['x','y','Treatment'], right_on = ['x','y','Treatment'])
# elldf = trans_rate_df_sep.copy()

fig, graphaxes = plt.subplots(1,len(elldf.Treatment.unique()),figsize=(7.5+(10*(len(elldf.Treatment.unique())-1)),10))
#single colorbar axis
cbar_ax = fig.add_axes([.965, .114, .025, .7215])
################ heatmap of probability density #############
bighm = np.zeros((len(transdf_sep.Treatment.unique()),nbins,nbins))
for i,(t, mdf) in enumerate(transdf_sep.groupby('Treatment')):
    #get total time observed in the system
    ttot = mdf.time_elapsed.sum()
    for x in range(nbins):
        for y in range(nbins):
            current =  mdf[(mdf['from_x'] == x+1) & (mdf['from_y'] == y+1)]
            if current.empty:
                bighm[i,y,x] = 0
            else:
                bighm[i,y,x] = current.time_elapsed.sum()/ttot
for i, ax in enumerate(graphaxes):
    #plot heatmap with seaborn
    sns.heatmap(
        bighm[i],
        vmin=0, vmax=bighm.max(), #center=0,
        cmap='rocket',
        square=True,
        xticklabels = True,
        yticklabels = True,
        ax = ax,
        cbar=i==0,
        cbar_ax = None if i else cbar_ax,
#         cbar_kws=cbar_kws
    )
  
    ######################### vector map of probability flux ################
    mm = elldf.Treatment.unique()[i]
    mdf = elldf[elldf.Treatment==mm]
    
    
    for x in range(1,nbins+1):
        for y in range(1,nbins+1):
            current = mdf[(mdf['x'] == x) & (mdf['y'] == y)]
            xcurrent = (current.x_plus_rate - current.x_minus_rate)/2
            ycurrent = (current.y_plus_rate - current.y_minus_rate)/2

            ell = Ellipse(xy=(x-0.5+(xcurrent.values*(1/scale)),y-0.5+(ycurrent.values*(1/scale))),
                    width=np.sqrt(abs(current.eval1)*(1/scale)) if current.evec1x.values[0] == 1 else np.sqrt(abs(current.eval2)*(1/scale)),
                      height=np.sqrt(abs(current.eval1)*(1/scale)) if current.evec1y.values[0] == 1 else np.sqrt(abs(current.eval2)*(1/scale)),
                    angle=np.arctan2(current.evec1y,current.evec1x),
                     color = 'lightblue')
            ax.add_artist(ell)
            ell.set_alpha(0.2)
    
    
    for x in range(1,nbins+1):
        for y in range(1,nbins+1):
            current = mdf[(mdf['x'] == x) & (mdf['y'] == y)]
            xcurrent = (current.x_plus_rate - current.x_minus_rate)/2
            ycurrent = (current.y_plus_rate - current.y_minus_rate)/2
            ax.quiver(x-0.5,
                       y-0.5, 
                       xcurrent,
                       ycurrent,
                      angles = 'xy',
                      scale_units = 'xy',
                      scale = scale,
#                       width = 0.012,
#                       minlength = 0.8,
                      color = 'white')
            
            
                    


    # axis label stuff
    ax.set_xlabel('PC1', fontsize = 45)
    ax.xaxis.set_label_coords(0.46,-0.05)
    ax.set_xticks(np.arange(0.5,nbins+0.5))
    ax.set_xticklabels([round(x,1) for x in centers.PC1.to_list()], fontsize = 22)
    ax.set_yticks(np.arange(0.5,nbins+0.5))
    ax.set_yticklabels([round(x,1) for x in centers.PC7.to_list()], fontsize = 22)
    ax.set_xlim(0,nbins+1)
    ax.set_ylim(0,nbins+1)
    ax.set_title(mm, fontsize = 45)#, loc = 'center',pad = -100)
    # ax.title.set_position([0.45, -100])

# adjust colorbar tick label size
cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(),fontsize=18)
#set axis title
graphaxes[0].set_ylabel('PC7', fontsize = 45)
graphaxes[0].yaxis.set_label_coords(-0.05, 0.465)


########## add scale for the vectors ##########
#legend background
lxp = 0.125
lyp = 0.125
rect = Rectangle((lxp, lyp), 1.8, 1.8, linewidth=1, edgecolor='black', facecolor='#80858a')
graphaxes[0].add_patch(rect)
rect.set_zorder(4 * 5)
#x-axis legend arrow
graphaxes[0].quiver(lxp+0.5,lyp+0.5,1*scale,0,angles = 'xy',scale_units = 'xy',scale = scale,color = "white",zorder = 4 * 5)
#x-axis legend text
xsc = f'{(np.diff(centers.PC1).mean()/time_interval)*scale:.1e}'
xsc = xsc.split('-')[0] + str(int(xsc.split('e')[1]))
graphaxes[0].text(lxp+0.25,lyp+0.05,xsc+' $s^{-1}$', color = 'white', fontsize = 13, fontweight = 'bold',zorder = 4 * 5)
#y-axis legend arrow
graphaxes[0].quiver(lxp+0.5,lyp+0.5,0,1*scale,angles = 'xy',scale_units = 'xy',scale = scale,color = 'white',zorder = 4 * 5)
#y-axis legend text
ysc = f'{(np.diff(centers.PC7).mean()/time_interval)*scale:.1e}'
ysc = ysc.split('-')[0] + str(int(ysc.split('e')[1]))
graphaxes[0].text(lxp+0.05,lyp+0.3,ysc+' $s^{-1}$', rotation = 'vertical', color = 'white', fontsize = 13, fontweight = 'bold',zorder = 4 * 5)
    


plt.tight_layout()
plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')



