# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:43:42 2025

@author: Aaron
"""

import os
import numpy as np
import pandas as pd
from CustomFunctions import utils, DetailedBalance
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib

scale = 4 # scale of the plot
time_interval = 5
whichpcs = [1,7]
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_Apply/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'


FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col = 0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())
#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)

#maybe drop the huge outlier
FullFrame = FullFrame[FullFrame.CellID != '20240527_488_EGFP-CAAX_640_SPY650-DNA_cell3_01'].copy()


##### quickly calculate aer and cf on the raw CGPS transitions
if os.path.exists(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_transitions_separated.csv'):
    raw_trans = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_transitions_separated.csv', index_col = 0)
    #add a movie columns to separate dataframe on
    raw_trans['Movie'] = [x.split('_frame')[0] for x in raw_trans.cell.to_list()]
    ############# measure aer and cycling frequencies ###########
    #add specific scaling
    xyscaling = [centers[f'PC{whichpcs[0]}'].diff().mean(),centers[f'PC{whichpcs[1]}'].diff().mean()]
    #set the origin to the actual center
    center = [round(nbins/2)]*2
    results = []
    for i, cells in raw_trans.groupby('CellID'):
        movielist = sorted(cells.Movie.unique(),key = lambda x: int(x.split('-')[-3]))
        for m in movielist:
            curmov = cells[cells.Movie == m]
            curmov, runs = utils.get_consecutive_timepoints(curmov, 'frame',1)
            for r in runs:
                cell = curmov.iloc[r].reset_index(drop=True)
                results.append(DetailedBalance.get_area_enclosing_rate(
                    cell,
                    nbins,
                    xyscaling,
                    center,
                    ))
    allaers = pd.concat(results).reset_index(drop=True)
# allaers['cell'] = [c+f'_frame_{int(f)}' for c, f in allaers[['Movie','frame']].values]
#merge aer and cf info
TotalFrame = FullFrame.merge(allaers[['aer','angular_velocity','cell']],left_on='cell',right_on='cell')

dflist = []
for i, t in TotalFrame.groupby('CellID'):
    t = t.sort_values('time').reset_index(drop=True)
    avreg = LinearRegression().fit(t.time.values.reshape(-1, 1), t.angular_velocity.cumsum().values.reshape(-1, 1))
    avresid = avreg.score(t.time.values.reshape(-1, 1), t.angular_velocity.cumsum().values.reshape(-1, 1))
    aerreg = LinearRegression().fit(t.time.values.reshape(-1, 1), t.aer.cumsum().values.reshape(-1, 1))
    aerresid = aerreg.score(t.time.values.reshape(-1, 1), t.aer.cumsum().values.reshape(-1, 1))
    dflist.append({'CellID':i,'avresid':avresid, 'avcoef':avreg.coef_[0][0], 'aerresid':aerresid,
                   'aercoef':aerreg.coef_[0][0],'speed':t.speed.mean(), 'speed_sem':t.speed.sem()})
avgdf = pd.DataFrame(dflist)



##### add euclidean distance
euclist = []
for i, cells in TotalFrame.groupby('CellID'):
    eucdist = 0
    cells['Movie'] = [x.split('_frame')[0] for x in cells.cell.to_list()]
    movielist = sorted(cells.Movie.unique(),key = lambda x: int(x.split('-')[-3]))
    for m in movielist:
        curmov = cells[cells.Movie == m]
        curmov, runs = utils.get_consecutive_timepoints(curmov, 'frame',1)
        #also calculate euclidean distance
        first = curmov.iloc[0]
        last = curmov.iloc[-1]
        eucdist = eucdist + np.sqrt((last.x-first.x)**2 +
        (last.y-first.y)**2 +
        (last.z-first.z)**2)
    euclist.append([i,eucdist])
avgdf['euclidean_distance'] = [x[1] for x in euclist] 

stats = ['speed','euclidean_distance']


#define the colors to make the meshes
set1 = plt.cm.Set1
set2 = plt.cm.Set2
set3 = plt.cm.Set3
cmap = matplotlib.colors.ListedColormap(list(set3.colors)+[set2.colors[-2]] + [set1.colors[-1]])


fig, axes = plt.subplots(1,len(stats),figsize=(len(stats)*scale,scale))
for i, ax in enumerate(axes):
    x = avgdf.aercoef
    y = avgdf[stats[i]]
    coef = np.polyfit(x,y,1)
    poly1d_fn = np.poly1d(coef) 
    p_corr, p_val = scipy.stats.pearsonr(x,y)
    sns.scatterplot(x = 'aercoef', y = stats[i], data = avgdf, hue='CellID', palette = cmap.colors,
                    edgecolor = '0.5', ax = ax, legend = None, zorder = 2)
    ax.plot(x, poly1d_fn(x), 'k', zorder = 3)
    if stats[i] != 'euclidean_distance':
        ax.errorbar(x.values, y.values, yerr= avgdf.speed_sem.values,
                    color=[0.3,0.3,0.3], alpha=0.5, capsize=3, ls = 'none', zorder=1)
        
        x_min, x_max = ax.get_xlim()
        ax.text(x_max-(x_max-x_min)/2*1.9,0.258, 'pcorr='+'{0:.3f}'.format(p_corr)+
                '\npval='+'{0:.3f}'.format(p_val))
        
    else:
    
        x_min, x_max = ax.get_xlim()
        ax.text(x_max-(x_max-x_min)/2*1.9,400, 'pcorr='+'{0:.3f}'.format(p_corr)+
                '\npval='+'{0:.3f}'.format(p_val))
    
        
    ax.set_ylabel(stats[i], fontsize = 18)
    ax.set_xlabel('Area Enclosing Rate (area/sec)')

plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')


