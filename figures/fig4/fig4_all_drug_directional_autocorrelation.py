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
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from scipy import stats

time_interval = 10 #sec/frame


#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'

##### open all of the data
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)

### limit the data to random and galvanotaxis
treatments = ['DMSO','Para-Nitro-Blebbistatin','CK666']
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
df['Treatment'] = pd.Categorical(df.Treatment.to_list(), categories=treatments, ordered=True)


print(f'n = {df[df.lag_min==time_interval/60].groupby("Treatment").apply(lambda x: x.shape[0])} \
      track segments for time lag {time_interval/60}')
print(f'n = {df[df.lag_min==maxlag*time_interval/60].groupby("Treatment").apply(lambda x: x.shape[0])} \
      track segments for time lag {maxlag*time_interval/60}')



# # Run repeated measures ANOVA
# aovrm = AnovaRM(data=df.groupby(['Treatment','lag_min']).mean().reset_index(), depvar='dot_prod', subject='Treatment', within=['lag_min'])
# result = aovrm.fit()
# print(result)

### run kruskal wallace at each time point
testlist = []
for i, l in df.groupby('lag_min'):
    if i != 0:
        _,pval = stats.kruskal(*[d.to_list() for _,d in l.groupby('Treatment').dot_prod])
        testlist.append({'lag':i,'pvalue':pval})
kwdf = pd.DataFrame(testlist)
#run BH multiple comparisons correction
reject, pvcorr = multipletests(kwdf['pvalue'],method='fdr_bh')[:2]
sigkw = kwdf[reject]

#set color palette
colorlist = ['#9c836b','#faa7a7','#faf191']
sns.set_palette(palette=colorlist)

##########  plot random versus galv
fig, ax = plt.subplots()
sns.lineplot(data = df, x = 'lag_min',y = 'dot_prod', hue = 'Treatment',  lw = 3, ax = ax)

# ###stars
# ax.plot(kwdf.iloc[[0,10]].lag.values, [0.95,0.95], ls = '-', color = 'black')
# midx = (kwdf.iloc[[0,10]].lag.diff()/2).values[1]+kwdf.iloc[0].lag
# ax.text(midx, 0.96, '*', fontsize = 10, ha = 'center')

# ax.text(sigkw.iloc[-3].lag, 0.32,'*', fontsize = 10, ha = 'center')
# ax.text(sigkw.iloc[-2].lag, 0.22,'*', fontsize = 10, ha = 'center')
# ax.text(sigkw.iloc[-1].lag, 0.22,'*', fontsize = 10, ha = 'center')

ax.set_ylabel('Directional Autocorrelation', fontsize = 18)
ax.set_xlabel('Time lag (min)', fontsize = 18)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()


plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)





