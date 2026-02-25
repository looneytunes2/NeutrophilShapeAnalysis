# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 14:11:14 2025

@author: Aaron
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

#get directories and open separated datasets
treatments = ['Random','Pre-Galvanotaxis','Galvanotaxis']
time_interval = 10 #sec/frame

#get directories and open separated datasets
basedir = Path('E:/Aaron/Combined_37C_Confocal_PCA_planar/')
datadir = basedir.joinpath('Data_and_Figs')
FullFrame = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col=0)
#limit data to the galv experiments
TotalFrame = FullFrame[FullFrame.Treatment.isin(treatments)]
TotalFrame.loc[:,'Treatment'] = pd.Categorical(TotalFrame.Treatment.to_list(), categories=treatments, ordered=True)



#### restrict dataframe to include galv and 5 min of random
df = TotalFrame[TotalFrame.frame>(180-(5*60)/time_interval)].copy()
#add time in minutes relative to the start of the EF
df.loc[:,'relative_time'] = (df.frame.values-180)*time_interval/60

######## get costheta to EF
# Normalize vectors to get unit direction vectors
traj = df[['Trajectory_X','Trajectory_Y','Trajectory_Z']].values
unitvecs = traj/np.linalg.norm(traj, axis = 1)[:, np.newaxis]
# Calculate dot products of consecutive unit vectors with the given lag
df['costheta'] = np.sum(unitvecs * [-1,0,0], axis=1)


##### make the plot
fig, ax = plt.subplots()
#plot first part as green
sns.lineplot(x ='relative_time', y= 'costheta', data = df[df.relative_time<=0], color = '0.4', lw = 2,
             label = 'Undirected', ax = ax, zorder = 2)
#plot the initial EF exposure
sns.lineplot(x ='relative_time', y= 'costheta', data = df[(df.relative_time>=0) & (df.relative_time<=1)], color = '#c4d461', lw = 2,
             label = '0-1 min EF', ax = ax, zorder = 2)
#plot second part as purple
sns.lineplot(x ='relative_time', y= 'costheta', data = df[df.relative_time>=1], color = '#6cb875', lw = 2,
             label = '>1min EF', ax = ax, zorder = 2)


ax.tick_params('x', labelsize = 11)
ax.tick_params('y', labelsize = 11)

#draw line at zero
ax.axvline(0, 0, 1, color = '0.6', lw = 1, ls = '--', zorder = 1)

ax.set_ylabel('Alignment to electric field', fontsize = 20)
ax.set_xlabel('Time (min)', fontsize = 20)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


#legend stuff
leg = ax.legend()
leg.set_bbox_to_anchor([0.85,0.35]) 
for line in leg.get_lines():
    line.set_linewidth(3)
for t, text in enumerate(leg.get_texts()):
    text.set_fontsize(14)


plt.tight_layout()





plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)
