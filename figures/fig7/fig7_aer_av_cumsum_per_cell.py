# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:18:51 2025

@author: Aaron
"""



import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from neutrophil_shape.config.loader import load_config

#which CPGS to look at
whichpcs = (1,2)

config = load_config(microscope_type='lls')
config._alignment = 'trajectory'
time_interval = config.im_params.time_interval

ntrans = config.db_params.ntrans
basedir = config.common.savedir
datadir = basedir.joinpath('shape_data')
dbdir = basedir.joinpath('detailed_balance')
dbbsdir = dbdir.joinpath('separatedatabs')

### open big data, mostly do this to add NaNs for timepoints when aer isn't calculated
FullFrame = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col = 0)
FullFrame['real_time'] = FullFrame.time.copy()
#open aers previously calculated
allaers = pd.read_csv(dbdir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_raw_transition_aer_cf.csv'), index_col = 0)

#merge aer and cf info
TotalFrame = FullFrame.merge(allaers, on=['CellID','real_time','frame'],how='left')

#get cumulative sums
cslist = []
for c, t in TotalFrame.groupby('CellID'):
    t = t.sort_values('time').reset_index(drop = True)
    #### expand the dataframe to insert nan rows between in the gaps
    insert_at = t[t.time_elapsed != t.cumulative_time.diff()].index
    new_index = list(range(len(t) + len(insert_at)))
    # shift original rows around the inserted positions
    t = t.reindex([i for i in new_index if i not in insert_at])
    t = t.reindex(new_index)  # fills missing with NaN
    t = t.reset_index(drop=True)
    t['CellID'] = t.CellID.ffill().copy()
    ## get area enclosed
    t['area_enclosed'] = t.aer * t.time_elapsed
    ## get cumulative sum of area enclosed
    t['ae_cumsum'] = t.area_enclosed.cumsum().copy()

    
    cslist.append(t)
    
csframe = pd.concat(cslist, ignore_index=True)#.reset_index()
#add time in minutes
csframe['timemin'] = csframe.time.values/60



#define the colors to make the meshes
set1 = plt.cm.Set1
set2 = plt.cm.Set2
set3 = plt.cm.Set3
cmap = matplotlib.colors.ListedColormap(list(set3.colors)+[set2.colors[-2]] + [set1.colors[-1]])

sns.set_palette(cmap.colors)

### plot the stuff
fig, ax = plt.subplots()
#plot lines with matplotlib so they show the nan breaks correctly
for ii, (i, cell) in enumerate(csframe.groupby('CellID')):
    # cell = cell.sort_values('timemin')
    ax.plot(cell.timemin, cell.ae_cumsum, c = cmap.colors[ii], lw = 2)
ax.set_xlabel('Time (min)', fontsize =18)
ax.set_ylabel('Area Enclosed (PC units²)', fontsize =18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '_aer.png', dpi = 500, bbox_inches='tight')


