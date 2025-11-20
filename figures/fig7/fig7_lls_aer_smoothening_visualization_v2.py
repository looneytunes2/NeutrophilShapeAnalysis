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
from CustomFunctions import utils
from scipy import interpolate
from matplotlib.lines import Line2D
import matplotlib.colors as mc
from matplotlib.cm import ScalarMappable
#get directories and open separated datasets


treatments = ['Random']
time_interval = 5 #sec/frame
whichpcs = [1,7]
ntrans = 1
pointspacing = (1/60)/30


#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_apply/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'
if not os.path.exists(savedir):
    os.makedirs(savedir)
    
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col = 0)

# open aers
allaers = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_raw_transition_aer_cf.csv', index_col = 0)
#merge aer and cf info
TotalFrame = pd.merge(FullFrame, allaers[['aer','angular_velocity','cell']],on='cell',how='left')


#open all the bootstrapped realizations
bsaers = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_{ntrans}_transition_Area_Enclosing_Rates.csv', index_col = 0)
bsgaps = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_{ntrans}_transition_Area_Enclosing_Rates_gaps.csv', index_col = 0)
bsaers_gaps = bsaers.merge(bsgaps, on = ['iter','real_time'])

#only use aers that are within the range of observed time of the real cells
minmaxtime = TotalFrame.groupby('CellID').time.max().min()
itertime = bsaers_gaps.groupby('iter').cumulative_time.max()
longiters = itertime[itertime>=minmaxtime].index.to_list()
bsaers_long = bsaers_gaps[bsaers_gaps.iter.isin(longiters)].copy()




# cell = TotalFrame[TotalFrame.CellID == TotalFrame.CellID.unique()[1]]
cellpicks = TotalFrame.CellID.unique()[[0,7]]


#color palette for AER
copal = sns.diverging_palette(20, 220, as_cmap=True)

#colors for diverging cmap
low_color = '#d14c45'   # blue
mid_color = '#a8a8a8'   # white
high_color = '#3e88ad'   # red
#create color map
cmap = mc.LinearSegmentedColormap.from_list(
    "custom_diverging_map",
    [low_color, mid_color, high_color]
)
# Normalize so midpoint corresponds to 0
norm = mc.TwoSlopeNorm(vmin=-0.015, vcenter = 0, vmax=0.015)
mappable = ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array([])  # required for colorbar



#make the figure axis
fig, ax = plt.subplots()

#list to keep track of final areas to find similar bootstraps
nanless = []
#loop through real cell picks and draw smoothened lines with aer state
for i, cell in TotalFrame[TotalFrame.CellID.isin(cellpicks)].groupby('CellID'):

    
    cell, tck, w = utils.get_aer_state(cell, time_interval,)
    #also add area enclosed
    cell['area_enclosed'] = cell.aer.values*time_interval
    #get consecutive runs so that gaps are properly plotted
    cell, runs = utils.get_consecutive_timepoints(cell, 'time', time_interval)
    #get cell without aer NAs
    cellnona = cell[~cell.aer.isna()].copy()
    for r in runs:
        if len(r)>3:
            rc = cell.iloc[r]
            curcell = rc[~rc.aer_smooth.isna()].copy()
            #smoothened aer for scatterplot
            celltime = curcell.time.values
            splevtime = (celltime-cell.time.min())/(cell.time.max()-cell.time.min())
            x, y = interpolate.splev(np.linspace(splevtime.min(),
                                                 splevtime.max(),
                                                 int((splevtime.max()-splevtime.min())/pointspacing)),
                                     tck, der=0)
            #get the interpolated derivative to color AER
            k = 3 if len(curcell)>3 else 1
            dtck, du = interpolate.splprep(np.stack((curcell.time.values,
                                                    curcell.aer_smooth.values)), k=1, s=0)
            dx, dy = interpolate.splev(np.linspace(0,1,len(x)),dtck)
            #plot the smoothened curve snippet
            ax.scatter(x/60, y, color = cmap(norm(dy)),s = 1.75, zorder=2)
            
    #plot the original curve
    ax.plot(cell.time.values/60, cell.area_enclosed.cumsum().values, color = 'black', lw = 1, zorder = 1)
    
    #add the Nan-less aer to a list to measure differences with bootstraps
    nanless.append(cellnona[['time','area_enclosed']])




#rename time in the bootstrapped data
fulliters = bsaers_long.rename(columns = {'real_time':'time'})

#calculate ae cumsums
fulliters['area_enclosed'] = fulliters.aer.values * time_interval
fulliterscumsum = fulliters.groupby('iter').apply(
    lambda x: x.area_enclosed.fillna(0).cumsum()).reset_index().rename(columns = {'area_enclosed':'aecumsum'})
#merge cumsum with time
fulliters = pd.merge(fulliters, fulliterscumsum.drop(columns = ['iter']), left_index = True, right_on = 'level_1')

#get the least differences between the real cell picks and the bs curves
bestfits = []
def myround(x, base=time_interval):
    return base * np.round(x/base)
for n in nanless:
    n['aecumsum'] = n.area_enclosed.cumsum()
    #round times to 5 to be consistent with bs iters
    n.loc[:,'time'] = myround(n.time.values)
    #pivot bs with only the real times
    piviters = fulliters[fulliters.time.isin(n.time)]
    df_wide = piviters.pivot_table(
        index="time",          
        columns="iter",  
        values="aecumsum",     
        aggfunc="first"
        ).reset_index().drop(columns = 'time')
    
    #sum of squared differences    
    diffs = df_wide.apply(lambda x: ((x - n.aecumsum)**2).sum(), axis = 0)
    #sort the diffs and get iter numbers
    bestfits.append(diffs.sort_values().index.to_list()[:5])




bspicks = [o for x in bestfits for o in x]
bspointspacing = 0.5
######### also plot three example bootstrapped curves on the same plot
for i in bspicks:
    #choose a random cell
    cell = bsaers_gaps[bsaers_gaps.iter == i].copy()
    #add a time column
    cell['time'] = cell['real_time']
    
    #get the smooth ae
    cell, tck, w = utils.get_aer_state(cell, time_interval,)
    #get consecutive runs so that gaps are properly plotted
    cell, runs = utils.get_consecutive_timepoints(cell, 'time', time_interval)
    #get cell without aer NAs
    cellnona = cell[~cell.aer.isna()].copy()
    for r in runs:
        if len(r)>3:
            rc = cell.iloc[r]
            curcell = rc[~rc.aer_smooth.isna()].copy()
            #smoothened aer for scatterplot
            celltime = curcell.time.values
            splevtime = (celltime-cell.time.min())/(cell.time.max()-cell.time.min())
            x, y = interpolate.splev(np.linspace(splevtime.min(),
                                                 splevtime.max(),
                                                 int((splevtime.max()-splevtime.min())/pointspacing)),
                                     tck, der=0)

            #plot the smoothened curve snippet
            ax.plot(x/60, y, color = '0.8', lw = 2, ls = 'dotted', zorder=0)
    




    
ax.set_ylabel('Area Enclosed (PC units²)', fontsize = 18)
ax.set_xlabel('Time (min)', fontsize = 18)

#set the ylim to match the plot of all aers
# ax.set_ylim(-0.4331245203743168, 4.876036020105756)#852923124466614)

#remove spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)



# Define custom legend items
type_legend_handles = [
    Line2D([0], [0], color='black', lw=1, label='Raw curve'),
    Line2D([0], [0], color='0.7', lw=3, label='Smoothed curve'),
    Line2D([0], [0], color='0.8', lw=2, ls = 'dotted', label='Bootstrapped'), 
]

# Add custom legend to this subplot
type_legend = ax.legend(handles=type_legend_handles,
                        loc=[0.05,0.7],
                        title='Data type')

ax.add_artist(type_legend)


cbar_ax = fig.add_axes([0.99, 0.215, 0.02, 0.65]) 
# Add the colorbar to the new axis
cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='vertical')
cbar.set_label('Area Enclosing Rate (PC units²/sec)', fontsize=10, rotation = -90, labelpad = 13)
cbar.ax.yaxis.set_label_position('right')
cbar.ax.tick_params(labelsize=8)  
# cbar.ax.set_yticklabels([str(x.get_position()[1]) for x in cbar.ax.get_yticklabels()],fontsize=12)




plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')




    