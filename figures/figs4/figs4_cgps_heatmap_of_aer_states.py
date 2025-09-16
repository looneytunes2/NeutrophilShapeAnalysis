# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 16:06:55 2025

@author: Aaron
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from CustomFunctions import utils






derivthresh = 0.0007
scale = 4 # scale of the plot
time_interval = 5
whichpcs = [1,7]
ntrans = 1
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_apply/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'


FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col = 0)
# open aers
allaers = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_raw_transition_aer_cf.csv', index_col = 0)

#merge aer and cf info
TotalFrame = FullFrame.merge(allaers[['aer','angular_velocity','cell']],left_on='cell',right_on='cell')
TotalFrame = TotalFrame.sort_values(['CellID','time'])

#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
nbins = centers.shape[0]

######## open all of the data
############## get the counts of cells leaving 
trans_rate_df_sep = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_binned_transition_rates_separated.csv', index_col=0)

allcells = []
for i, cell in TotalFrame.groupby('CellID'):
    # ####running mean method
    # cell['aer_deriv'] = np.gradient(utils.running_mean_withna(cell.aer.cumsum(), 25), cell.time.values)
    
    cell, tck , w = utils.get_aer_state(cell, time_interval, derivthresh)
    #append that cell
    allcells.append(cell)
    
derivframe = pd.concat(allcells).reset_index(drop=True)
#define just unchanging and increasing
treatments = derivframe.aer_state.unique()[1:]


########### calculate the pdf of the treatments in the CGPS #############
statemap = np.zeros((len(treatments), nbins, nbins))
for i, treat in enumerate(treatments):
    tdf = derivframe[derivframe.aer_state == treat]
    for x in range(nbins):
        for y in range(nbins):
            current =  tdf[(tdf['PC1bins'] == x+1) & (tdf['PC7bins'] == y+1)]
            if current.empty:
                pass
            else:
                #add the number of counts in this bin
                statemap[i,y,x] = len(current)/len(tdf)


########### get the PDF of the whole dataset #############
wholepdf = np.zeros((nbins, nbins))

for x in range(nbins):
    for y in range(nbins):
        current =  TotalFrame[(TotalFrame['PC1bins'] == x+1) & (TotalFrame['PC7bins'] == y+1)]
        if current.empty:
            pass
        else:
            #add the number of counts in this bin
            wholepdf[y,x] = len(current)/len(tdf)


####### Get differences between whole pdf and states ########
differencemaps = statemap[0] - statemap[1]




fig, ax = plt.subplots(1, 1, figsize = (5,5))
cbar_ax = fig.add_axes([.96, .137, .03, .762])
# for i, ax in enumerate(axes):

dm = abs(differencemaps).max()
#plot heatmap with seaborn
sns.heatmap(
    differencemaps,
    vmin=-dm,
    vmax=dm, 
    # center=0,
    cmap=sns.diverging_palette(220, 20, n=200),
    square=True,
    xticklabels = True,
    yticklabels = True,
    ax = ax,
    cbar_ax = cbar_ax,
)
#correct axis orientations
ax.invert_yaxis()
#get rid of ticks and labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticks(np.arange(0.5,nbins+0.5)[[0,nbins//2,-1]])
ax.set_xticklabels([round(centers.PC1.iloc[x],1) for x in [0,nbins//2, int(nbins-1)]],
                   fontsize = 10)
ax.set_yticks(np.arange(0.5,nbins+0.5)[[0,nbins//2,-1]])
ax.set_yticklabels([round(centers.PC7.iloc[x],1) for x in [0,nbins//2, int(nbins-1)]],
                   fontsize = 10)
#set axis titles
ax.set_xlabel(f'PC{whichpcs[0]}', fontsize = 18)
ax.set_ylabel(f'PC{whichpcs[1]}', fontsize = 18)

#set title
ax.set_title('Unchanging - Increasing', fontsize = 24)



# adjust colorbar tick label size
cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(),fontsize=10)
cbar_ax.get_yaxis().labelpad = 18
cbar_ax.set_ylabel('Relative Probability Density', fontsize = 16, rotation=270)


plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')




# scale = 0.0008
# for x in range(1,nbins+1):
#     for y in range(1,nbins+1):
#         current = trans_rate_df_sep[(trans_rate_df_sep['x'] == x) & (trans_rate_df_sep['y'] == y)]
#         xcurrent = (current.x_plus_rate - current.x_minus_rate)/2
#         ycurrent = (current.y_plus_rate - current.y_minus_rate)/2
#         if not xcurrent.empty:
#             anglecolor = (np.arctan2(xcurrent,ycurrent) *180/np.pi)+180
#             ax.quiver(x-0.5,
#                        y-0.5, 
#                        xcurrent,
#                        ycurrent,
#                       angles = 'xy',
#                       scale_units = 'xy',
#                       scale = scale,
#     #                   width = 0.012,
#     #                   minlength = 0.8,
#                       color = '0.5',
#                       alpha = 0.4,
#                         zorder = 3 * 5)
    
    

    
    
# plt.tight_layout()



