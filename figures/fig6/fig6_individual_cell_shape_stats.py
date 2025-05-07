# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 14:18:33 2025

@author: Aaron
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

scale = 4
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_Apply/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'


FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col = 0)


metrics = ['Cell_Volume','Cell_SurfaceArea','Cell_Aspect_Ratio']
labelz = ['Cell Volume (µm$^3$)','Cell Surface Area (µm$^2$)', 'Cell Aspect Ratio']



#define the colors to make the meshes
set1 = plt.cm.Set1
set2 = plt.cm.Set2
set3 = plt.cm.Set3
cmap = matplotlib.colors.ListedColormap(list(set3.colors)+[set2.colors[-2]] + [set1.colors[-1]])


fig, axes = plt.subplots(1,len(metrics), figsize=(scale*len(metrics), scale))

for i, ax in enumerate(axes):
    sns.boxplot(data = FullFrame, x = 'Treatment', y = metrics[i], hue = 'CellID', palette = cmap.colors, ax = ax)
    ax.legend_ = None
    ax.set_xlabel('')
    ax.set_xticks([])
    ax.set_xticklabels('')
    ax.set_ylabel(labelz[i], fontsize = 20)
    
    #plot the average of the averages
    avgs = FullFrame[['CellID',metrics[i]]].groupby('CellID').mean()
    avgavg = avgs[metrics[i]].mean()
    ax.axhline(avgavg, ls = '--', color = 'black', alpha = 0.4)
    
plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')

