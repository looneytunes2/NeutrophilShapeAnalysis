# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:18:51 2025

@author: Aaron
"""


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression



time_interval = 5
whichpcs = [1,7]
ntrans = 1
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_Apply/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'


FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col = 0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())
#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)

#open aers previously calculated
allaers = pd.read_csv(savedir + f'PC{whichpcs[0]}-PC{whichpcs[1]}_raw_transition_aer_cf.csv', index_col = 0)

#merge aer and cf info
TotalFrame = pd.merge(FullFrame, allaers[['aer','angular_velocity','cell']],on='cell',how='left')
TotalFrame = TotalFrame.sort_values(['CellID','time'])

#open all the bootstrapped realizations
bsaers = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_{ntrans}_transition_Area_Enclosing_Rates.csv', index_col = 0)

#only use aers that are within the range of observed time of the real cells
minmaxtime = TotalFrame.groupby('CellID').time.max().min()
itertime = bsaers.groupby('iter').cumulative_time.max()
longiters = itertime[itertime>=minmaxtime]
bsaers_long = bsaers[bsaers.iter.isin(longiters.index.to_list())].copy()

#calculate aer and fit for real cells
dflist = []
for i, t in TotalFrame.groupby('CellID'):
    t = t[['time','aer']].dropna().sort_values('time').reset_index(drop=True)
    t['time_min'] = t.time/60
    aerreg = LinearRegression(fit_intercept = False).fit(np.insert(t.time_min.values,0,0).reshape(-1, 1),
                                                         np.insert(t.aer.cumsum().values,0,0).reshape(-1, 1))
    aerresid = aerreg.score(np.insert(t.time_min.values,0,0).reshape(-1, 1),
                            np.insert(t.aer.cumsum().values,0,0).reshape(-1, 1))
    dflist.append({'CellID':i,'aerresid':aerresid,'aercoef':aerreg.coef_[0][0]})
avgdf = pd.DataFrame(dflist)


#calculate aer and fit for bootstrapped cells
bslist = []
for i, t in bsaers_long.groupby('iter'):
    t = t[['cumulative_time','aer']].dropna().sort_values('cumulative_time').reset_index(drop=True)
    t['time_min'] = t.cumulative_time/60
    aerreg = LinearRegression(fit_intercept = False).fit(np.insert(t.time_min.values,0,0).reshape(-1, 1),
                                                         np.insert(t.aer.cumsum().values,0,0).reshape(-1, 1))
    aerresid = aerreg.score(np.insert(t.time_min.values,0,0).reshape(-1, 1),
                            np.insert(t.aer.cumsum().values,0,0).reshape(-1, 1))
    bslist.append({'iter':i,'aerresid':aerresid,'aercoef':aerreg.coef_[0][0]})
bsdf = pd.DataFrame(bslist)



#define the colors to make the meshes
set1 = plt.cm.Set1
set2 = plt.cm.Set2
set3 = plt.cm.Set3
cmap = matplotlib.colors.ListedColormap(list(set3.colors)+[set2.colors[-2]] + [set1.colors[-1]])
sns.set_palette(cmap.colors)

#truncate the greys color map for the density plot
greymap = plt.get_cmap('Greys')
new_cmap = matplotlib.colors.ListedColormap(
    greymap(np.linspace(0.15, 0.8, 256)))


### probability density proportions to use as levels for the kde plot
lvls = [0.01,0.2,0.4,0.6,0.8,1]

### plot the stuff
fig, ax = plt.subplots(figsize = (5,4))
cbar_ax = fig.add_axes([.98, .150, .03, .80])

#individual dots
sns.scatterplot(y = 'aerresid', x = 'aercoef', data = avgdf, hue = 'CellID',
                s = 100, edgecolor = '0.4', ax = ax, zorder = 2)

#density plot of the bootstrapped data
sns.kdeplot(data = bsdf, x = 'aercoef', y = 'aerresid', levels = lvls, fill = True,
            cmap = new_cmap, cbar = True, cbar_ax = cbar_ax, ax = ax, zorder = 1)

ax.set_ylabel('AER R$^2$', fontsize = 15)
ax.set_xlabel('Area Enclosing Rate (PC units²/sec)', fontsize = 15)

#change fontsize on axis ticks
ax.tick_params(labelsize = 8)

ax.set_xlim(0.018,0.126)
ax.set_ylim(0.6,1.05)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend_ = None
# ax.set_aspect('equal')

#adjust the colobar stuff
cbar_ax.set_yticklabels(lvls,fontsize=8)
# cbar_ax.get_yaxis().labelpad = 18
cbar_ax.set_ylabel('Bootstrapped Density Proportion', fontsize = 14,
                   rotation=-90, labelpad = 13)


plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')

