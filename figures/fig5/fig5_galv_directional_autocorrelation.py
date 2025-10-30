# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 11:07:51 2025

@author: Aaron
"""

import numpy as np
import pandas as pd
from CustomFunctions import utils
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy import stats


#get directories and open separated datasets
treatments = ['Random','Galvanotaxis']
time_interval = 10 #sec/frame

#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'galv/'
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
#limit data to the galv experiments
TotalFrame = FullFrame[FullFrame.Treatment.isin(treatments)]
TotalFrame.loc[:,'Treatment'] = pd.Categorical(TotalFrame.Treatment.to_list(), categories=treatments, ordered=True)


#define the maximum time lag
maxlag = int(60/time_interval*5) # 5 minutes


plist = []
for c, cell in TotalFrame.groupby(['Treatment','CellID']):
    cell, runs = utils.get_consecutive_timepoints(cell, 'frame', 1)
    fmin = cell.frame.min()
    fmax = cell.frame.max()
    # galvdict[c[0]][c[1]] = {}
    plist.append({'Treatment':c[0],'CellID':c[1],'lag_min':0,'dot_prod':1})
    for lag in range(2,int(maxlag + 2)):
        if len(cell)>lag:
            frames = np.arange(fmin, fmax, step = lag)
            # plist = []
            for f in frames:
                traj = cell[cell.frame.isin([f,int(f+lag-1)])][['Trajectory_X','Trajectory_Z','Trajectory_Z']].values
                if len(traj)==2:
                    # Normalize vectors to get unit direction vectors
                    unitvecs = traj/np.linalg.norm(traj, axis = 1)[:, np.newaxis]
                    # Calculate dot products of consecutive unit vectors with the given lag
                    dot_products = np.sum(unitvecs[:-1] * unitvecs[1:], axis=1)
                    plist.append({'Treatment':c[0],'CellID':c[1],'lag_min':(lag-1)*time_interval/60,'dot_prod':dot_products[0]})
df = pd.DataFrame(plist)
df['Treatment'] = pd.Categorical(df.Treatment.to_list(), categories=treatments, ordered=True)


print(f'n = {df[df.lag_min==time_interval/60].groupby("Treatment").apply(lambda x: x.shape[0])} \
      track segments for time lag {time_interval/60}')
print(f'n = {df[df.lag_min==maxlag*time_interval/60].groupby("Treatment").apply(lambda x: x.shape[0])} \
      track segments for time lag {maxlag*time_interval/60}')




#set color palette
colorlist = ['0.4','#6cb875']
sns.set_palette(palette=colorlist)


##########  plot random versus galv
fig, ax = plt.subplots()
sns.lineplot(data = df, x = 'lag_min',y = 'dot_prod', hue = 'Treatment',
             lw = 3, ax = ax)

ax.set_ylabel('Directional Autocorrelation', fontsize = 18)
ax.set_xlabel('Time lag (min)', fontsize = 18)

# handles, _ = ax.get_legend_handles_labels()
# ax.legend(labels = ['Undirected','Electrotaxis'],
#           handles = handles,
#           title = '',)
#           # loc = [])
### make the legend larger
leg = ax.legend(loc = [0.6, 0.8])
for line in leg.get_lines():
    line.set_linewidth(3)
for t, text in enumerate(leg.get_texts()):
    text.set_text(['Undirected','Electrotaxis'][t])
    text.set_fontsize(14)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()


plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)



