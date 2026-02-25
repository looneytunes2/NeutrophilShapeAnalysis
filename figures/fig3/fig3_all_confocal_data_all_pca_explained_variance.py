# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 12:32:55 2026

@author: Aaron
"""


import pickle as pk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from pathlib import Path



####### load common directories and data
basedir = Path('E:/Aaron')
dirlist = ['Combined_37C_Confocal_PCA_shape','Combined_37C_Confocal_PCA_s5','Combined_37C_Confocal_PCA_planar']
labeldict = {
    'Combined_37C_Confocal_PCA_shape': 'Shape Only',
    'Combined_37C_Confocal_PCA_s5': 'Trajectory + Shape',
    'Combined_37C_Confocal_PCA_planar': 'Trajectory Only'
    }
colorlist = cm.Set2.colors[:3][::-1]


var_list = []
for dirr in dirlist:
    tempdir = basedir.joinpath(dirr,'Data_and_Figs','pca.pkl')
    # open confocal pca model
    pca = pk.load(open(tempdir,'rb')) 
    # How much variance is explained?
    cell_variance = np.cumsum(pca.explained_variance_ratio_)
    var_dict = [{'pca': dirr, 'pc': int(v+1), 'var': vv} for v, vv in enumerate(cell_variance)]
    var_list.extend(var_dict)
    
vardf = pd.DataFrame(var_list)


fig, ax = plt.subplots()
sns.lineplot(data = vardf, x = 'pc', y = 'var', hue = 'pca', palette = colorlist,
             linewidth = 3, ax = ax)

##set ax limits
ymin, ymax = ax.get_ylim()
ax.set_ylim(0, ymax)

ax.set_xlabel('Number of PCs', fontsize = 16)
ax.set_ylabel('Proportion of Variance Explained', fontsize = 16)


### legend adjustments
handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(handles, [labeldict[l] for l in labels], title = 'Alignment Method',
                title_fontsize = 12)

for line in leg.get_lines():
    line.set_linewidth(3)

#remove box parts
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()



plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)

