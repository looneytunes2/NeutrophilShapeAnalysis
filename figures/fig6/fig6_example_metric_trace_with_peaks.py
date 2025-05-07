# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 16:29:27 2025

@author: Aaron
"""



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import interpolate, signal
import matplotlib
from CustomFunctions.utils import get_consecutive_timepoints

scale = 4
time_interval = 5
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_Apply/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'


TotalFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col = 0)

#define the colors to make the meshes
set1 = plt.cm.Set1
set2 = plt.cm.Set2
set3 = plt.cm.Set3
cmap = matplotlib.colors.ListedColormap(list(set3.colors)+[set2.colors[-2]] + [set1.colors[-1]])


fpdict = {'Cell_Volume':{'distance':8,'prominence':50,'width':5},
          'Cell_Aspect_Ratio':{'distance':8,'prominence':0.4,'width':3},
          'Volume_Front_Ratio':{'distance':8,'prominence':0.08,'width':3},
          'LengthAlongTrajectory':{'distance':8,'prominence':5,'width':3},
          'LengthAlongTrajectoryFront':{'distance':8,'prominence':2.5,'width':3},
          'LengthAlongTrajectoryRear':{'distance':8,'prominence':2.5,'width':3},
          'Cell_TotalAngle':{'distance':6,'prominence':5,'width':3},
          'speed':{'distance':6,'prominence':0.2,'width':3}}


######## using find peaks to look at frequencies ##########
cellnum = 9


dat = TotalFrame[TotalFrame.CellID==TotalFrame.CellID.unique()[cellnum]].sort_values('time').reset_index(drop=True)
tck = interpolate.splrep(dat.time.values,dat.pr_ratio.values, k=1, s=1)
x = interpolate.splev(np.arange(5,dat.time.max(),5),tck,der=0)
peaks, properties = signal.find_peaks(x, distance = 4, prominence=0.05, width=1.5)

fig, ax = plt.subplots(figsize=(14,2))
ax.plot(np.arange(5,dat.time.max(),5)/60, x ,zorder=1) #color = cmap.colors[cellnum]
ax.plot(dat.time.values/60, dat.pr_ratio.values)
ax.scatter(np.arange(5,dat.time.max(),5)[peaks]/60, x[peaks],s=15,marker='o',facecolors='black', edgecolors='black',zorder=2)

ax.set_ylabel('Cell Volume (µm$^3$)')
ax.set_xlabel('Time (min)')

plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')




