# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 12:52:22 2025

@author: Aaron
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from CustomFunctions import utils
from CustomFunctions.shapePCAtools import filter_extremes_based_on_percentile
from scipy import stats
from pathlib import Path


def get_stars(pv):
    if pv < 0.001:
        stars = '***'
    elif pv < 0.01:
        stars = '**'
    elif pv < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'
    return stars

ntrans = 1
whichpcs = [1,2]
time_interval = 10
treatments = ['Random','Galvanotaxis']
xlabels = ['Undirected', 'Electrotaxis']
basedir = Path('E:/Aaron/Combined_37C_Confocal_PCA_planar/')
datadir = basedir.joinpath('Data_and_Figs')
aerdir = basedir.joinpath('Detailed_Balance')


df = pd.read_csv(aerdir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_bootstrapped_{ntrans}_Area_Enclosing_Rates.csv'), index_col = 0)
df.loc[:,'Treatment'] = pd.Categorical(df.Treatment.to_list(), categories=treatments, ordered=True)


dflist = []
for i, t in df.groupby(['Treatment','iter']):
    t = t.rename(columns = {'cumulative_time':'time'})
    t = t.sort_values('time').reset_index(drop=True)
    #fit aer
    aerresid, aercoef = utils.fit_AER(t,time_interval,'aer')
    dflist.append({'Treatment':i[0],'iter':i[1],'aerresid':aerresid,'aercoef':aercoef})
avgdf = pd.DataFrame(dflist)


avgdf_filtered = filter_extremes_based_on_percentile(
    avgdf,
    ['aercoef','aerresid'],
    1)



############### CELL AVERAGES OF SIGNIFICANT METRICS #################################
colorlist = ['0.65','#8adb93']
sns.set_palette(palette=colorlist)


##### stats for line fits
stat, pval = stats.mannwhitneyu(avgdf[avgdf.Treatment == treatments[0]].aercoef.values,
                            avgdf[avgdf.Treatment == treatments[1]].aercoef.values)
print('Mann Whitney U test for AER coeff p value is ', pval)

fig, ax = plt.subplots(1, 1, figsize=(4,5))#, sharex=True)
linewid = 2
# sns.swarmplot(data = avgcfdf, x='Treatment', y ='average_cf', color = 'grey', size = 3.5, alpha = 0.7, ax = ax)
sns.violinplot(x = 'Treatment', y='aercoef', data = avgdf_filtered,
               linewidth = 0, inner = None, ax=ax, )
ax.collections[0].set_edgecolor('black')
ax.collections[1].set_edgecolor('black')
sns.boxplot(x = 'Treatment', y='aercoef', data = avgdf_filtered, width = 0.15, color = 'white',
            showcaps=False, showfliers=False,
            boxprops={
                'fill': 'white',
                'linewidth': linewid,
                'edgecolor': 'black',
                'zorder': 2
                },
            medianprops={
                'linewidth': linewid,
                'color': 'black'
                },
            whiskerprops={
                'linewidth': 0,
                'color': 'black'
                },
            capprops={
                'linewidth': linewid,
                'color': 'black'
                },
            ax=ax)

ax.text(0.5,0.027,get_stars(pval), fontsize=12, ha='center')
## set y axis min lim to 0
ymin, ymax = ax.get_ylim()
ax.set_ylim(0, ymax)

ax.set_xlabel('', fontsize=20)
ax.tick_params('y', labelsize=10)
#modify the labels to put bleb in two lines
ax.set_xticklabels(xlabels, fontsize = 14)
#remove legends
ax.legend_ = None
ax.set_ylabel('Area Enclosing Rate (PC units²/sec)', fontsize=16)
#remove parts of box
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '_AER.png', dpi = 500, bbox_inches='tight')





##### stats for line fits
stat, pval = stats.mannwhitneyu(avgdf[avgdf.Treatment == treatments[0]].aerresid.dropna().values,
                   avgdf[avgdf.Treatment == treatments[1]].aerresid.dropna().values)
print('Mann Whitney test for R squared p value is ',pval)

fig, ax = plt.subplots(1, 1, figsize=(4,5))#, sharex=True)
linewid = 2
# sns.swarmplot(data = avgcfdf, x='Treatment', y ='average_cf', color = 'grey', size = 3.5, alpha = 0.7, ax = ax)
sns.violinplot(x = 'Treatment', y='aerresid', data = avgdf_filtered,
               linewidth = 0, inner = None, ax=ax, )
ax.collections[0].set_edgecolor('black')
ax.collections[1].set_edgecolor('black')
sns.boxplot(x = 'Treatment', y='aerresid', data = avgdf_filtered, width = 0.15, color = 'white',
            showcaps=False, showfliers=False,
            boxprops={
                'fill': 'white',
                'linewidth': linewid,
                'edgecolor': 'black',
                'zorder': 2
                },
            medianprops={
                'linewidth': linewid,
                'color': 'black' 
                },
            whiskerprops={
                'linewidth': 0,
                'color': 'black'
                },
            capprops={
                'linewidth': linewid,
                'color': 'black'
                },
            ax=ax)

ax.text(0.5,1.005, get_stars(pval), fontsize=12, ha='center')
## set y axis min lim to 0
ymin, ymax = ax.get_ylim()
ax.set_ylim(0, ymax)

ax.set_xlabel('', fontsize=20)
ax.tick_params('y', labelsize=10)
#modify the labels to put bleb in two lines
ax.set_xticklabels(xlabels, fontsize = 14)
#remove legends
ax.legend_ = None
ax.set_ylabel('Area Enclosing Rate R²', fontsize=16)
#remove parts of box
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '_Rsq.png', dpi = 500, bbox_inches='tight')
