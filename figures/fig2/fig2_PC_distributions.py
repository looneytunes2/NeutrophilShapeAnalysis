# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:00:09 2025

@author: Aaron
"""

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from neutrophil_shape.config.loader import load_config
from pathlib import Path

### load config
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'
npcs = config.common.npcs

#get directories and open separated datasets
savedir = config.common.savedir
datadir = savedir / 'shape_data'
TotalFrame = pd.read_csv(datadir / 'All_Data_with_CGPS_bins.csv', index_col=0)



#get PCs in order
PCs = ['PC'+str(i) for i in range(1,npcs+1)]
#add them together and select them in the dataframe
pcframe = TotalFrame[PCs]

############ no Y-axis
fig, axes = plt.subplots(len(PCs), 1, figsize=(2,len(PCs)), sharey=True, sharex = True)
for i, ax in enumerate(axes):
    sns.kdeplot(data = pcframe, x=f'PC{i+1}', fill = True, color ='grey', ax = ax)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([-2,-1,0,1,2])
    ax.set_yticks([])

axes[-1].set_xlabel('PC Value', fontsize = 16)#, labelpad=-0.5)
fig.suptitle('PC Distributions',fontsize = 13, y = 0.95)


plt.tight_layout()

plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi=300)



