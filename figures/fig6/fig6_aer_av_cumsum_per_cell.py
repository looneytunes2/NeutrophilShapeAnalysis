# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:18:51 2025

@author: Aaron
"""


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


time_interval = 5
whichpcs = [1,7]
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_Apply/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'

### open big data
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col = 0)

#open aers previously calculated
allaers = pd.read_csv(savedir + f'PC{whichpcs[0]}-PC{whichpcs[1]}_raw_transition_aer_cf.csv', index_col = 0)

#merge aer and cf info
TotalFrame = pd.merge(FullFrame, allaers[['aer','angular_velocity','cell']],on='cell',how='left')
#drop all but the important columns
TotalFrame = TotalFrame[['CellID','time','aer']]
TotalFrame = TotalFrame.sort_values(['CellID','time'])

#get cumulative sums
csframe = []
for c, t in TotalFrame.groupby('CellID'):
    t['aer_cumsum'] = t.aer.cumsum().copy()
    
    replacelist = []
    for i in t[t.aer_cumsum.isna()].index:
        if i!=t.iloc[0].name:
            curna = t.loc[i].copy()
            #replace the nan with the previous value
            t.loc[i,'aer_cumsum'] = t.loc[int(i-1)].aer_cumsum
            #add new nan 5sec before previous
            curna.time = curna.time-time_interval
            
            replacelist.append(curna.to_dict())
    t = pd.concat((t, pd.DataFrame(replacelist))).sort_values('time')
    csframe.append(t)
csframe = pd.concat(csframe, ignore_index = True)
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
for ii, (i, c) in enumerate(csframe.groupby('CellID')):
    ax.plot(c.timemin, c.aer_cumsum, c = cmap.colors[ii], lw = 2)
ax.set_xlabel('Time (min)', fontsize =18)
ax.set_ylabel('CGPS Area Enclosed', fontsize =18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '_aer.png', dpi = 500, bbox_inches='tight')


