# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 12:52:22 2025

@author: Aaron
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from CustomFunctions.shapePCAtools import filter_extremes_based_on_percentile
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

ntrans = 1
whichpcs = [1,7]
treatments = ['Random','Galvanotaxis']
xlabels = ['Undirected', 'Electrotaxis']
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
savedir = [basedir + 'galv/', basedir + 'random/']

dfs = []
for s in savedir:
    dfs.append(pd.read_csv(s+f'PC{whichpcs[0]}-PC{whichpcs[1]}_{ntrans}_transition_Area_Enclosing_Rates.csv', index_col = 0))
df = pd.concat(dfs, ignore_index = True)
df.loc[:,'Treatment'] = pd.Categorical(df.Treatment.to_list(), categories=treatments, ordered=True)


dflist = []
for i, t in df.groupby(['Treatment','iter']):
    t = t.sort_values('cumulative_time').reset_index(drop=True)
    t['cumulative_time_min'] = t.cumulative_time/60
    aerreg = LinearRegression(fit_intercept = False).fit(np.insert(t.cumulative_time_min.values,0,0).reshape(-1, 1),
                                                         np.insert(t.aer.cumsum().values,0,0).reshape(-1, 1))
    aerresid = aerreg.score(np.insert(t.cumulative_time_min.values,0,0).reshape(-1, 1),
                            np.insert(t.aer.cumsum().values,0,0).reshape(-1, 1))
    dflist.append({'Treatment':i[0],'iter':i[1],'aerresid':aerresid,'aercoef':aerreg.coef_[0][0]})
avgdf = pd.DataFrame(dflist)


avgdf_filtered = filter_extremes_based_on_percentile(
    avgdf,
    ['aercoef','aerresid'],
    1)



############### CELL AVERAGES OF SIGNIFICANT METRICS #################################
# colorlist = [list(sns.color_palette('pastel').as_hex())[i] for i in [0,3,8]]
colorlist = ['#e3a1db','#afe6aa']
sns.set_palette(palette=colorlist)


##### stats for line fits
stat, pval = stats.stats.ttest_ind(avgdf[avgdf.Treatment == treatments[0]].aercoef.values,
                            avgdf[avgdf.Treatment == treatments[1]].aercoef.values)
print('t test for R squared ', pval)

fig, ax = plt.subplots(1, 1, figsize=(4,5))#, sharex=True)
linewid = 2
# sns.swarmplot(data = avgcfdf, x='Treatment', y ='average_cf', color = 'grey', size = 3.5, alpha = 0.7, ax = ax)
sns.violinplot(x = 'Treatment', y='aercoef', data = avgdf_filtered,
               linewidth = 0, inner = None, ax=ax, )
ax.collections[0].set_edgecolor('black')
ax.collections[1].set_edgecolor('black')
sns.boxplot(x = 'Treatment', y='aercoef', data = avgdf, width = 0.15, color = 'white',
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

ax.text(0.5,0.055,get_stars(pval), fontsize=12, ha='center')
# ax.set_ylim(0,0.042)
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
stat, pval = stats.mannwhitneyu(avgdf[avgdf.Treatment == treatments[0]].aerresid.values,
                   avgdf[avgdf.Treatment == treatments[1]].aerresid.values)
print('Mann Whitney test for R squared ',pval)

fig, ax = plt.subplots(1, 1, figsize=(4,5))#, sharex=True)
linewid = 2
# sns.swarmplot(data = avgcfdf, x='Treatment', y ='average_cf', color = 'grey', size = 3.5, alpha = 0.7, ax = ax)
sns.violinplot(x = 'Treatment', y='aerresid', data = avgdf_filtered,
               linewidth = 0, inner = None, ax=ax, )
ax.collections[0].set_edgecolor('black')
ax.collections[1].set_edgecolor('black')
sns.boxplot(x = 'Treatment', y='aerresid', data = avgdf, width = 0.15, color = 'white',
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

ax.text(0.5,1.02, get_stars(pval), fontsize=12, ha='center')
# ax.set_ylim(0,1.05)
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
