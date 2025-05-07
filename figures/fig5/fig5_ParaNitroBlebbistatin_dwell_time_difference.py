# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:20:32 2025

@author: Aaron
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



treatments = ['DMSO','Para-Nitro-Blebbistatin']
time_interval = 10 #sec/frame
whichpcs = [1,7]

#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'Para-Nitro-Blebbistatin/'
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())
#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
#limit data to the Para-Nitro-Blebbistatin experiments
TotalFrame = FullFrame[FullFrame.Experiment == 'Drug']
dates = [20240624,20240626,20240701,20241125,20241126,20241127]
TotalFrame = TotalFrame[TotalFrame.Date.isin(dates)]
TotalFrame['Treatment'] = pd.Categorical(TotalFrame.Treatment.to_list(), categories=treatments, ordered=True)


######## open all of the data
########### interpolate all transitions so that only individual transitions are made ###########
transdf_sep = pd.read_csv(savedir+f'interpolated_PC{whichpcs[0]}-PC{whichpcs[1]}_transitions_separated.csv', index_col=0)
#ensure that DMSO is the first in order
transdf_sep['Treatment'] = pd.Categorical(transdf_sep.Treatment, categories=treatments, ordered=True)
transdf_sep = transdf_sep.sort_values(by='Treatment')






########### plot the DWELL TIME DIFFERENCE of the treatments in the CGPS #############

fig, ax = plt.subplots(1,1,figsize=(5,5))
#single colorbar axis
cbar_ax = fig.add_axes([.99, .165, .04, .805])

#calculate control heatmap to subtract from the treatments
mm = transdf_sep.Treatment.unique()[0]
mdf = transdf_sep[transdf_sep.Treatment==treatments[0]]
ttot = mdf.time_elapsed.sum()
################ heatmap of probability density #############
#make numpy array with heatmap data
ctrlhm = np.zeros((nbins,nbins))
#get total time observed in the system

for x in range(nbins):
    for y in range(nbins):
        current =  mdf[(mdf['from_x'] == x+1) & (mdf['from_y'] == y+1)]
        if current.empty:
            ctrlhm[y,x] = 0
        else:
            ctrlhm[y,x] = current.time_elapsed.mean()



mm = transdf_sep.Treatment.unique()[1]
mdf = transdf_sep[transdf_sep.Treatment==mm]
ttot = mdf.time_elapsed.sum()
################ heatmap of probability density #############
#make numpy array with heatmap data
bighm = np.zeros((nbins,nbins))
#get total time observed in the system

for x in range(nbins):
    for y in range(nbins):
        current =  mdf[(mdf['from_x'] == x+1) & (mdf['from_y'] == y+1)]
        if current.empty:
            bighm[y,x] = 0
        else:
            bighm[y,x] = current.time_elapsed.mean()


difference = bighm-ctrlhm

#plot heatmap with seaborn
sns.heatmap(
    difference,
#         vmin=2.5, vmax=9, 
    center=0,
    cmap=sns.diverging_palette(220, 20, n=200),
    square=True,
    xticklabels = True,
    yticklabels = True,
    ax = ax,
    cbar_ax = cbar_ax,
)
#correct axis orientations
ax.invert_yaxis()
#set tick labels
ax.set_xticks(np.arange(0.5,nbins+0.5)[[0,(round(nbins/2)-1),-1]])
ax.set_xticklabels([round(centers.PC1.iloc[x],1) for x in [0,int(round(nbins/2)-1), int(nbins-1)]], fontsize = 16)
ax.set_yticks(np.arange(0.5,nbins+0.5)[[0,(round(nbins/2)-1),-1]])
ax.set_yticklabels([round(centers.PC7.iloc[x],1) for x in [0,int(round(nbins/2)-1), int(nbins-1)]], fontsize = 16)
#set axis titles
ax.set_xlabel(f'PC{whichpcs[0]}', fontsize = 24)
ax.set_ylabel(f'PC{whichpcs[1]}', fontsize = 24)



#set title
# ax.set_title(mm, fontsize = 32)


# adjust colorbar tick label size
cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(),fontsize=12)
cbar_ax.get_yaxis().labelpad = 18
cbar_ax.set_ylabel('Relative Dwell Time (sec)', fontsize = 16, rotation=270)


plt.tight_layout()

plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')