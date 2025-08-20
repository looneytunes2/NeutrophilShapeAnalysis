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
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm 
from statsmodels.formula.api import ols 
import scikit_posthocs as sp


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
treatments = ['DMSO','Para-Nitro-Blebbistatin','CK666']
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'drug/'

df = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_{ntrans}_transition_Area_Enclosing_Rates.csv', index_col = 0)
df['Treatment'] = pd.Categorical(df.Treatment.to_list(), categories=treatments, ordered=True)


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
colorlist = ['#d1b59b','#f7bebe','#faf191']
sns.set_palette(palette=colorlist)


#separate dataframes
ctrlframe = avgdf[avgdf.Treatment == treatments[0]]
pnbframe = avgdf[avgdf.Treatment == treatments[1]]
ck666frame = avgdf[avgdf.Treatment == treatments[2]]

tstat, pnbpval = stats.ttest_ind(ctrlframe.aercoef.values, pnbframe.aercoef.values)
tstat, ck666pval = stats.ttest_ind(ctrlframe.aercoef.values, ck666frame.aercoef.values)
reject, pvcorr = multipletests([pnbpval, ck666pval], method = 'bonferroni')[:2]
pnbpval_adj, ck666pval_adj = pvcorr
print(f't test for AER between {treatments[0]} and {treatments[1]} is {pnbpval_adj}')
print(f't test for AER between {treatments[0]} and {treatments[2]} is {ck666pval_adj}')


fig, ax = plt.subplots(1, 1, figsize=(4,5))#, sharex=True)
linewid = 2
# sns.swarmplot(data = avgcfdf, x='Treatment', y ='average_cf', color = 'grey', size = 3.5, alpha = 0.7, ax = ax)
sns.violinplot(x = 'Treatment', y='aercoef', data = avgdf_filtered, alpha = 0.4,
               linewidth = 0, inner = None, ax=ax, )
for ac in ax.collections:
    ac.set_edgecolor('black')
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
#dmso to bleb
ax.text(0.5,0.0409,get_stars(pnbpval_adj), fontsize=12, ha='center')
ax.plot([0.1,0.9],[0.0408,0.0408], color = 'black')
# #bleb to ck666
# ax.text(1.5,0.0315,'***', fontsize=12, ha='center')
# ax.plot([1.1,1.9],[0.0314,0.0314], color = 'black')
#dmso to ck666
ax.text(1,0.0431,get_stars(ck666pval_adj), fontsize=12, ha='center')
ax.plot([0.1,1.9],[0.043,0.043], color = 'black')
# ax.set_ylim(-5,60)
ax.set_xlabel('', fontsize=20)
ax.tick_params('y', labelsize=10)
#modify the labels to put bleb in two lines
ax.set_xticklabels([x[:11]+'\n'+x[11:] if x == 'Para-Nitro-Blebbistatin' else x for x in treatments], fontsize = 10)
#remove legends
ax.legend_ = None
ax.set_ylabel('Area Enclosing Rate (PC units²/sec)', fontsize=16)
#remove parts of box
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '_AER.png', dpi = 500, bbox_inches='tight')




##### stats for line fits
tstat, pnbpval = stats.mannwhitneyu(ctrlframe.aerresid.values, pnbframe.aerresid.values)
tstat, ck666pval = stats.mannwhitneyu(ctrlframe.aerresid.values, ck666frame.aerresid.values)
reject, pvcorr = multipletests([pnbpval, ck666pval], method = 'bonferroni')[:2]
pnbpval_adj, ck666pval_adj = pvcorr
print(f't test for Rsq between {treatments[0]} and {treatments[1]} is {pnbpval_adj}')
print(f't test for Rsq between {treatments[0]} and {treatments[2]} is {ck666pval_adj}')

fig, ax = plt.subplots(1, 1, figsize=(4,5))#, sharex=True)
linewid = 2
# sns.swarmplot(data = avgcfdf, x='Treatment', y ='average_cf', color = 'grey', size = 3.5, alpha = 0.7, ax = ax)
sns.violinplot(x = 'Treatment', y='aerresid', data = avgdf_filtered,
               linewidth = 0, inner = None, ax=ax, )
for ac in ax.collections:
    ac.set_edgecolor('black')
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

#DMSO to bleb
ax.text(0.5,1.095,get_stars(pnbpval_adj), fontsize=12, ha ='center')
ax.plot([0,1],[1.1,1.1], color = 'black')
#DMSO to CK666
ax.text(1,1.172,get_stars(ck666pval_adj), fontsize=12, ha ='center')
ax.plot([0,2],[1.17,1.17], color = 'black')

#ditch x axis label
ax.set_xlabel('', fontsize=20)
#change ticks
ax.tick_params('y', labelsize=10)
#modify the labels to put bleb in two lines
ax.set_xticklabels([x[:11]+'\n'+x[11:] if x == 'Para-Nitro-Blebbistatin' else x for x in treatments], fontsize = 10)
#remove legends
ax.legend_ = None
ax.set_ylabel('Area Enclosing Rate R²', fontsize=16)
#remove parts of box
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '_Rsq.png', dpi = 500, bbox_inches='tight')