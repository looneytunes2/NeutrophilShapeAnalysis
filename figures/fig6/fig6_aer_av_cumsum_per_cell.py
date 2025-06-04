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


scale = 4 # scale of the plot
time_interval = 5
whichpcs = [1,7]
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_Apply/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'


FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col = 0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())
#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
# open aers
allaers = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_raw_transition_aer_cf.csv', index_col = 0)

#merge aer and cf info
TotalFrame = pd.merge(FullFrame, allaers[['aer','angular_velocity','cell']],on='cell',how='left')
TotalFrame = TotalFrame.sort_values(['CellID','time'])

#get cumulative sums
csframe = []
for c, t in TotalFrame.groupby('CellID'):
    t = t.sort_values('time')
    t['aer_cumsum'] = t.aer.cumsum().copy()
    t['angular_velocity_cumsum'] = t.angular_velocity.cumsum().copy()
    csframe.append(t)
csframe = pd.concat(csframe).reset_index(drop=True)
csframe['timemin'] = csframe.time.values/60

#define the colors to make the meshes
set1 = plt.cm.Set1
set2 = plt.cm.Set2
set3 = plt.cm.Set3
cmap = matplotlib.colors.ListedColormap(list(set3.colors)+[set2.colors[-2]] + [set1.colors[-1]])

sns.set_palette(cmap.colors)

### plot the stuff
fig, ax = plt.subplots()
sns.lineplot(x='timemin',y='aer_cumsum',data=csframe,hue='CellID', lw=2, ci=None, legend=None)
ax.set_xlabel('Time (min)', fontsize =18)
ax.set_ylabel('CGPS Area Enclosed', fontsize =18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '_aer.png', dpi = 500, bbox_inches='tight')



fig, ax = plt.subplots()
sns.lineplot(x='timemin',y='angular_velocity_cumsum',data=csframe,hue='CellID', lw=2, ci=None, legend=None)
ax.set_xlabel('Time (min)', fontsize =18)
ax.set_ylabel('Degrees Traveled', fontsize =18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '_angularvelocity.png', dpi = 500, bbox_inches='tight')
