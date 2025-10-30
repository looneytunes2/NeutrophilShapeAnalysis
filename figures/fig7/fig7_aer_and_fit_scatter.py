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
from CustomFunctions import utils
from scipy import interpolate

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
minmaxtime = TotalFrame.groupby('CellID').time.count().min()
itertime = bsaers.groupby('iter').cumulative_time.count()
longiters = itertime[itertime>=minmaxtime]
bsaers_long = bsaers[bsaers.iter.isin(longiters.index.to_list())].copy()



#calculate aer and fit for real cells
dflist = []
for i, t in TotalFrame.groupby('CellID'):
    ### smoothen bootstrapped AE curve
    #ensure the cell is in time order
    cellnona = t[['time','aer']].dropna().sort_values('time').reset_index(drop=True)
    #scrunch time all together so there are no gaps
    cellnona['time'] = np.arange(0, len(cellnona)) * time_interval
    #interpolate for smoothening
    tck, u = interpolate.splprep(np.array((cellnona.time.values,
                                           cellnona.aer.cumsum().values)),
                                 k=3, s = 0.5)#k=1, s=2, w = w)
    x, y = interpolate.splev(u, tck, der=0)
    #get the derivative of the smoothened curve
    dx, dy = interpolate.splev(u, tck, der=1)
    deriv = dy/cellnona.time.max()
    #add new values to dataframe
    cellnona['aer_deriv'] = deriv
    
    t = cellnona[['time','aer_deriv']].dropna().sort_values('time').reset_index(drop=True)
    t['time_min'] = t.time/60
    aerreg = LinearRegression().fit(t.time_min.values.reshape(-1, 1),
                                    t.aer_deriv.cumsum().values.reshape(-1, 1))
    aerresid = aerreg.score(t.time_min.values.reshape(-1, 1),
                            t.aer_deriv.cumsum().values.reshape(-1, 1))
    dflist.append({'CellID':i,'aerresid':aerresid,'aercoef':aerreg.coef_[0][0]})
avgdf = pd.DataFrame(dflist)


#calculate aer and fit for bootstrapped cells
bslist = []
for i, t in bsaers_long.groupby('iter'):
    ### smoothen bootstrapped AE curve
    #ensure the cell is in time order
    cellnona = t.sort_values('real_time').reset_index(drop=True)

    #interpolate for smoothening
    tck, u = interpolate.splprep(np.array((cellnona.real_time.values,
                                           cellnona.aer.cumsum().values)),
                                 k=3, s = 0.5)#k=1, s=2, w = w)
    x, y = interpolate.splev(u, tck, der=0)
    #get the derivative of the smoothened curve
    dx, dy = interpolate.splev(u, tck, der=1)
    deriv = dy/cellnona.real_time.max()
    #add new values to dataframe
    cellnona['aer_deriv'] = deriv
    
    t = cellnona[['real_time','aer_deriv']].dropna().sort_values('real_time').reset_index(drop=True)
    t['time_min'] = t.real_time/60
    aerreg = LinearRegression().fit(t.time_min.values.reshape(-1, 1),
                                    t.aer_deriv.cumsum().values.reshape(-1, 1))
    aerresid = aerreg.score(t.time_min.values.reshape(-1, 1),
                            t.aer_deriv.cumsum().values.reshape(-1, 1))
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
lvls = [0.02,0.2,0.4,0.6,0.8,1]

### plot the stuff
fig, ax = plt.subplots()
cbar_ax = fig.add_axes([.98, .24, .03, .6])

#individual dots
sns.scatterplot(y = 'aerresid', x = 'aercoef', data = avgdf, hue = 'CellID',
                s = 100, edgecolor = '0.4', ax = ax, zorder = 2)

#density plot of the bootstrapped data
sns.kdeplot(data = bsdf, x = 'aercoef', y = 'aerresid', levels = lvls, fill = True,
            cmap = new_cmap, cbar = True, cbar_ax = cbar_ax, ax = ax, zorder = 1)

ax.set_ylabel('Area Enclosing Rate R$^2$', fontsize = 18)
ax.set_xlabel('Area Enclosing Rate (PC units²/sec)', fontsize = 18)

#change fontsize on axis ticks
ax.tick_params(labelsize = 8)

ax.set_xlim(0.004,0.024)
ax.set_ylim(0.8,1.03)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend_ = None
# ax.set_aspect('equal')

#adjust the colobar stuff
cbar_ax.set_yticklabels(lvls,fontsize=8)
# cbar_ax.get_yaxis().labelpad = 18
cbar_ax.set_ylabel('Bootstrapped Density Proportion', fontsize = 10,
                   rotation=-90, labelpad = 13)


plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')

