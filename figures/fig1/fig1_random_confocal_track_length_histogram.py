# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:42:54 2025

@author: Aaron
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



#get directories and open separated datasets


treatments = ['Random']

#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
#restrict dataframe to only random experiments
TotalFrame = FullFrame[FullFrame.Treatment.isin(treatments)].copy()
#get the counts of each cell id
allcounts = TotalFrame.CellID.value_counts()



fig, ax = plt.subplots(figsize = (3,4))
#histogram
sns.histplot(y = allcounts, binwidth = 5, lw=2, color = '0.6', ax = ax)
#### dashed line cutoff for average cells
ax.plot([0,29.5],[10.625,10.625],ls='--',lw = 3,c='#1581b0')
#### dashed line cutoff for cell stdev
ax.plot([0,29.5],[25.625,25.625],ls='--',lw = 3,c='#821065')

#axis labels
ax.set_xlabel('Count', fontsize = 16)
ax.set_ylabel('# of Images in Track', fontsize = 16)

ax.set_xlim(0,30)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')