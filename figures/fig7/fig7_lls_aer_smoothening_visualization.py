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

#get directories and open separated datasets


treatments = ['Random']
time_interval = 5 #sec/frame
derivthresh = 0.0007
whichpcs = [1,7]
ntrans = 1
pointspacing = 0.5


#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_apply/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'
if not os.path.exists(savedir):
    os.makedirs(savedir)
    
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col = 0)

# open aers
allaers = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_raw_transition_aer_cf.csv', index_col = 0)
#open the bootstrapped realizations
bsaers = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_{ntrans}_transition_Area_Enclosing_Rates.csv', index_col = 0)


#merge aer and cf info
TotalFrame = pd.merge(FullFrame, allaers[['aer','angular_velocity','cell']],on='cell',how='left')




# cell = TotalFrame[TotalFrame.CellID == TotalFrame.CellID.unique()[1]]
cellpicks = TotalFrame.CellID.unique()[[0,12,13]]


fig, ax = plt.subplots()

#list to keep track of final areas to find similar bootstraps
finalareas = []
nanless = []
#loop through real cell picks and draw smoothened lines with aer state
for i, cell in TotalFrame[TotalFrame.CellID.isin(cellpicks)].groupby('CellID'):

    #ensure the cell is in time order
    cell = cell.sort_values('time').reset_index(drop=True)
    #get rid of NA in aer which will ruin cumulative sums etc.
    cellnona = cell[~cell.aer.isna()].copy()
    #### weight the points near gaps more
    diffs = cellnona.time.diff().values
    #get the indicies of jumps
    gaps = np.where(diffs>time_interval)[0]
    #add the indices before jumps
    gaps = np.concatenate((gaps,gaps-1))
    w = np.ones(diffs.shape)
    w[gaps] = 3

    #interpolate for smoothening
    tck, u = interpolate.splprep(np.array((cellnona.time.values,
                                           cellnona.aer.cumsum().values)),
                                 k=3, s = 1, w = w)#k=1, s=2, w = w)
    x, y = interpolate.splev(u, tck, der=0)
    #get the derivative of the smoothened curve
    deriv = np.gradient(y, x)
    #threshold with np.select
    threshs = [deriv>=derivthresh, deriv<=-derivthresh]
    choices = ['increasing', 'decreasing']
    statethresh = np.select(threshs, choices, default = 'unchanging')
    #add new values to dataframe
    cell.loc[cellnona.index,'aer_deriv'] = deriv
    cell.loc[cellnona.index,'aer_state'] = statethresh
    
    
    
    #get consecutive runs
    cell, runs = utils.get_consecutive_timepoints(cell, 'time', time_interval)
    #get absolute time max to normalize time windows
    tmax = cell.time.max()
    for r in runs:
        if len(r)>3:
            rc = cell.iloc[r]
            curcell = rc[~rc.aer_state.isna()].copy()
            #smoothened aer for scatterplot
            x, y = interpolate.splev(np.linspace(curcell.time.min()/tmax,
                                                 curcell.time.max()/tmax,
                                                 int((curcell.time.max()-curcell.time.min())/pointspacing)),
                                     tck, der=0)
            
            #interpolate to get colors
            threshs = [curcell.aer_state.values=='decreasing', curcell.aer_state.values=='unchanging', curcell.aer_state.values=='increasing']
            choices = [1, 2, 3]
            numbertransform = np.select(threshs, choices)
            f = interpolate.interp1d(curcell.time.values,numbertransform)
            colornums = f(np.arange(curcell.time.min(),curcell.time.max(), pointspacing))
            threshs = [colornums<1.5, (colornums>=1.5) & (colornums<2.5), colornums>=2.5]
            choices = ['#d14c45','#a8a8a8','#3e88ad'] # ['Decreasing', 'Unchanging', 'Increasing']
            colors = np.select(threshs, choices)
            
            #plot the smoothened curve snippet
            ax.scatter(x/60, y, color = colors,s = 2, zorder=2)
            
    #plot the original curve
    ax.plot(cell.time.values/60, cell.aer.cumsum().values, color = '0.7', lw = 1, zorder = 1)
    
    finalareas.append(cell.aer.cumsum().values[-1])

    #add the Nan-less aer to a list to measure differences with bootstraps
    nanless.append(cellnona[['time','aer']])

#### select three bootstrap iterations that have similar aers to the real cell
#limit to iters that go the full time
maxiter = bsaers.iter.value_counts().max()
iterfull = bsaers.iter.value_counts()[bsaers.iter.value_counts()==maxiter].index

# ## get final areas of the bootstraps
# finalbsarea = bsaers[bsaers.iter.isin(iterfull)].groupby('iter').apply(lambda x: x.aer.cumsum().iloc[-1])

#get all the data from full bs iters
fulliters = bsaers[bsaers.iter.isin(iterfull)].rename(columns = {'real_time':'time'})
#calculate aer cumsums
fulliterscumsum = fulliters.groupby('iter').apply(lambda x: x.aer.fillna(0).cumsum()).reset_index().rename(columns = {'aer':'aercumsum'})
#merge cumsum with time
fulliters = pd.merge(fulliters, fulliterscumsum.drop(columns = ['iter']), left_index = True, right_on = 'level_1')

#get the least differences between the real cell picks and the bs curves
bestfits = []
def myround(x, base=5):
    return base * np.round(x/base)
for n in nanless:
    n['aercumsum'] = n.aer.cumsum()
    #round times to 5 to be consistent with bs iters
    n.loc[:,'time'] = myround(n.time.values)
    #pivot bs with only the real times
    piviters = fulliters[fulliters.time.isin(n.time)]
    df_wide = piviters.pivot_table(
        index="time",          
        columns="iter",  
        values="aercumsum",     
        aggfunc="first"
        ).reset_index().drop(columns = 'time')
    
    #sum of squared differences    
    diffs = df_wide.apply(lambda x: ((x - n.aercumsum)**2).sum(), axis = 0)
    #sort the diffs and get iter numbers
    bestfits.append(diffs.sort_values().index.to_list()[:10])


# #### get bs with closest area to real cells
# firstmatch = (finalbsarea-finalareas[0]).abs().sort_values()[:15] #bs iters near cell 0
# secondmatch = (finalbsarea-finalareas[1]).abs().sort_values()[:15] #bs iters near cell 12
# thirdmatch = (finalbsarea-finalareas[2]).abs().sort_values()[:15] #bs iters near cell 13




bspicks = [2597, 2562, 935]
bspointspacing = 0.5
######### also plot three example bootstrapped curves on the same plot
for i in bspicks:
    #choose a random cell
    cell = bsaers[bsaers.iter == i].copy()
    
    #get the aer state
    #get rid of NA in aer which will ruin cumulative sums etc.
    cellnona = cell[~cell.aer.isna()].copy()
    #### weight the points near gaps more
    diffs = cellnona.cumulative_time.diff().values
    #get the indicies of jumps
    gaps = np.where(diffs>time_interval)[0]
    #add the indices before jumps
    gaps = np.concatenate((gaps,gaps-1))
    w = np.ones(diffs.shape)
    w[gaps] = 3

    # ####running mean method
    # deriv = np.gradient(utils.running_mean_withna(cell.aer.cumsum(), 25), cell.time.values)
    ####interpolation method
    #interpolate for smoothening
    tck, u = interpolate.splprep(np.array((cellnona.cumulative_time.values,
                                           cellnona.aer.cumsum().values)),
                                 k=3, s = 1, w = w)#k=1, s=2, w = w)
    x, y = interpolate.splev(u, tck, der=0)
    #get the derivative of the smoothened curve
    deriv = np.gradient(y, x)
    #threshold with np.select
    threshs = [deriv>=derivthresh, deriv<=-derivthresh]
    choices = ['increasing', 'decreasing']
    statethresh = np.select(threshs, choices, default = 'unchanging')
    #add new values to dataframe
    cell.loc[cellnona.index,'aer_deriv'] = deriv
    cell.loc[cellnona.index,'aer_state'] = statethresh
    
    # ####running mean method
    # deriv = np.gradient(utils.running_mean_withna(cell.aer.cumsum(), 25), cell.time.values)
    ####interpolation method
    tck, u = interpolate.splprep(np.array((cellnona.cumulative_time.values, cellnona.aer.cumsum().values)), k=3, s=1, w = w)
    x, y = interpolate.splev(u, tck, der=0)
    deriv = np.gradient(y, x)
    threshs = [deriv>=derivthresh, deriv<=-derivthresh]
    choices = ['up', 'down']
    statethresh = np.select(threshs, choices, default = 'zero')
    cellnona.loc[:,'aer_deriv'] = deriv
    cellnona.loc[:,'aer_state'] = statethresh
    
    #smoothened aer for scatterplot
    x, y = interpolate.splev(np.linspace(0,1,int((cellnona.cumulative_time.max()-cellnona.cumulative_time.min())/bspointspacing)), tck, der=0)
    
    #interpolate to get colors
    threshs = [cellnona.aer_state.values=='down', cellnona.aer_state.values=='zero', cellnona.aer_state.values=='up']
    choices = [1, 2, 3]
    numbertransform = np.select(threshs, choices)
    f = interpolate.interp1d(cellnona.cumulative_time.values,numbertransform)
    colornums = f(np.arange(cellnona.cumulative_time.min(),cellnona.cumulative_time.max(), bspointspacing))
    threshs = [colornums<1.5, (colornums>=1.5) & (colornums<2.5), colornums>=2.5]
    choices = ['#d14c45','#a8a8a8','#3e88ad'] # ['Decreasing', 'Unchanging', 'Increasing']
    colors = np.select(threshs, choices)
    
    # #plot the original curve
    # ax.plot(cellnona.cumulative_time.values/60, cellnona.aer.cumsum().values, color = 'black', alpha = 0.25, zorder = 1)
    #plot the smoothened curve
    ax.plot(x/60, y, color = '0.8', lw = 2, ls = 'dotted', zorder=0)
    # sns.scatterplot(x = x/60, y = y, hue = colors,linewidth = 0,s = 7, ax =ax)








    
ax.set_ylabel('Area Enclosed (PC units²)', fontsize = 18)
ax.set_xlabel('Time (min)', fontsize = 18)

#set the ylim to match the plot of all aers
ax.set_ylim(-0.4331245203743168, 4.876036020105756)#852923124466614)

#remove spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# Define aer state legend
state_legend_handles = [
    Line2D([0], [0], color='#d14c45', lw=2, label='Decreasing'),
    Line2D([0], [0], color='#a8a8a8', lw=2, label='Unchanging'),
    Line2D([0], [0], color='#3e88ad', lw=2, label='Increasing'),
    # Line2D([0], [0], color='0.7', lw=1, label='Raw curve'),
    # Line2D([0], [0], color='0.8', lw=2, ls = 'dotted', label='Bootstrapped'),
    
]
state_legend = ax.legend(handles=state_legend_handles,
                         loc=[0.02,0.77],
                         title='AER state')


# Define custom legend items
type_legend_handles = [
    Line2D([0], [0], color='0.7', lw=1, label='Raw curve'),
    Line2D([0], [0], color='0.8', lw=2, ls = 'dotted', label='Bootstrapped'), 
]

# Add custom legend to this subplot
type_legend = ax.legend(handles=type_legend_handles,
                        loc=[0.02,0.58],
                        title='Data type')

ax.add_artist(state_legend)


plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')




    