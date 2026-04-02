# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 12:52:22 2025

@author: Aaron
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from neutrophil_shape.CustomFunctions.shapePCAtools import filter_extremes_based_on_percentile
from neutrophil_shape.config.loader import load_config
from scipy import stats


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



#define some variables
treatments = ['Random','Galvanotaxis']
xlabels = ['Undirected', 'Electrotaxis']
whichpcs = (4,5)
config = load_config(microscope_type='confocal')
ntrans = config.db_params.ntrans
time_interval = config.im_params.time_interval
config._alignment = 'trajectory'
savedir = config.common.savedir
dbbsdir = savedir.joinpath('detailed_balance','separatedatabs')


df = pd.read_csv(dbbsdir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_bootstrapped_{ntrans}_Area_Enclosed_Linear_Reg.csv'), index_col = 0)
df = df[df.Treatment.isin(treatments)].copy()
df['Treatment'] = pd.Categorical(df.Treatment.to_list(), categories=treatments, ordered=True)


avgdf_filtered = filter_extremes_based_on_percentile(
    df,
    ['aer_coeff','aer_fit'],
    1)


print(f'{treatments[0]} AER mean is {df[df.Treatment == treatments[0]].aer_coeff.mean()}'
          f' and median is {df[df.Treatment == treatments[0]].aer_coeff.median()}')
print(f'{treatments[1]} AER mean is {df[df.Treatment == treatments[1]].aer_coeff.mean()}'
          f' and median is {df[df.Treatment == treatments[1]].aer_coeff.median()}')


############### CELL AVERAGES OF SIGNIFICANT METRICS #################################
colorlist = ['0.65','#8adb93']

##### stats for line fits
stat, pval = stats.mannwhitneyu(df[df.Treatment == treatments[0]].aer_coeff.values,
                            df[df.Treatment == treatments[1]].aer_coeff.values)
print('Mann Whitney U test for AER coeff p value is ', pval)

fig, ax = plt.subplots(1, 1, figsize=(4,5))#, sharex=True)
linewid = 2
# sns.swarmplot(data = avgcfdf, x='Treatment', y ='average_cf', color = 'grey', size = 3.5, alpha = 0.7, ax = ax)
sns.violinplot(x = 'Treatment', y='aer_coeff', data = avgdf_filtered,
               linewidth = 0, inner = None, ax=ax)
for i, ac in enumerate(ax.collections):
    ac.set_facecolor(colorlist[i])
    ac.set_edgecolor('black')
sns.boxplot(x = 'Treatment', y='aer_coeff', data = avgdf_filtered, width = 0.15, color = 'white',
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


ymin, ymax = ax.get_ylim()

ax.text(0.5,ymax*0.95,get_stars(pval), fontsize=12, ha='center')
## set y axi
# ax.set_ylim(ymin, ymax)

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
stat, pval = stats.mannwhitneyu(df[df.Treatment == treatments[0]].aer_fit.dropna().values,
                   df[df.Treatment == treatments[1]].aer_fit.dropna().values)
print('Mann Whitney test for R squared p value is ',pval)

fig, ax = plt.subplots(1, 1, figsize=(4,5))#, sharex=True)
linewid = 2
# sns.swarmplot(data = avgcfdf, x='Treatment', y ='average_cf', color = 'grey', size = 3.5, alpha = 0.7, ax = ax)
sns.violinplot(x = 'Treatment', y='aer_fit', data = avgdf_filtered,
               linewidth = 0, inner = None, ax=ax)
for i, ac in enumerate(ax.collections):
    ac.set_facecolor(colorlist[i])
    ac.set_edgecolor('black')
sns.boxplot(x = 'Treatment', y='aer_fit', data = avgdf_filtered, width = 0.15, color = 'white',
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

ax.text(0.5,1.01, get_stars(pval), fontsize=12, ha='center')
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
