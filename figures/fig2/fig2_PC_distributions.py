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


#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
TotalFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
# make sure all categories are ordered
TotalFrame['Treatment'] = pd.Categorical(TotalFrame.Treatment.to_list(), categories=['Random','Pre-Galvanotaxis','Galvanotaxis','DMSO','CK666','Para-Nitro-Blebbistatin'], ordered=True)
TotalFrame['Experiment'] = pd.Categorical(TotalFrame.Experiment.to_list(), categories=['Galvanotaxis','Drug'], ordered=True)




#get PCs in order
PCs = list(np.unique([re.search('PC\d*',x)[0] for x in TotalFrame.columns.to_list() if re.search('PC\d*',x) is not None]))
PCs.sort(key=lambda x: float(x.split('PC')[1]))
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
plt.tight_layout()

plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi=300)



