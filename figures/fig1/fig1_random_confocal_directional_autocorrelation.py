# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 11:07:51 2025

@author: Aaron
"""

import numpy as np
import pandas as pd
from neutrophil_shape.CustomFunctions import utils
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from neutrophil_shape.config.loader import load_config

treatments = ['Random']


#get directories and open separated datasets
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'
datadir = config.common.savedir / 'shape_data'
time_interval = config.im_params.time_interval


FullFrame = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col=0)

### limit the data to random and galvanotaxis
treatments = ['Random']
TotalFrame = FullFrame[FullFrame.Treatment.isin(treatments)].copy().reset_index(drop=True)

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

print(f'n = {len(df[df.lag_min==time_interval/60])} track segments for time lag {time_interval/60}')
print(f'n = {len(df[df.lag_min==maxlag*time_interval/60])} track segments for time lag {maxlag*time_interval/60}')

##########  plot random versus galv
fig, ax = plt.subplots()
sns.lineplot(data = df, x = 'lag_min',y = 'dot_prod', color = '0.4',  lw = 3, ax = ax)

ax.set_ylabel('Directional Autocorrelation', fontsize = 18)
ax.set_xlabel('Time lag (min)', fontsize = 18)

ax.tick_params(axis='both', labelsize=14)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)



# Define custom legend items
type_legend_handles = [
    Line2D([0], [0], color='0.4', lw=3, label='Mean'),
    Line2D([0], [0], color=[0.4, 0.4, 0.4, 0.2], lw=9, label='95% CI'),
]

# Add custom legend to this subplot
type_legend = ax.legend(handles=type_legend_handles,
                        loc=[0.6,0.7],
                        fontsize = 14,
                        )#title='Data type')

ax.add_artist(type_legend)



plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')


