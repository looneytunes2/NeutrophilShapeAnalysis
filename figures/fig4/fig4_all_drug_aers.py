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
time_interval = 10
treatments = ['DMSO','Para-Nitro-Blebbistatin','CK666']
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'drug/'

df = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_{ntrans}_transition_Area_Enclosing_Rates.csv', index_col = 0)
df['Treatment'] = pd.Categorical(df.Treatment.to_list(), categories=treatments, ordered=True)


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
colorlist = ['#d1b59b','#f7bebe','#faf191']
sns.set_palette(palette=colorlist)


#separate dataframes
ctrlframe = avgdf[avgdf.Treatment == treatments[0]]
pnbframe = avgdf[avgdf.Treatment == treatments[1]]
ck666frame = avgdf[avgdf.Treatment == treatments[2]]

tstat, pnbpval = stats.ttest_ind(ctrlframe.aercoef.values, pnbframe.aercoef.values)
tstat, ck666pval = stats.ttest_ind(ctrlframe.aercoef.values, ck666frame.aercoef.values)
# reject, pvcorr = multipletests([pnbpval, ck666pval], method = 'bonferroni')[:2]
# pnbpval_adj, ck666pval_adj = pvcorr
print(f't test for AER between {treatments[0]} and {treatments[1]} is {pnbpval}')
print(f't test for AER between {treatments[0]} and {treatments[2]} is {ck666pval}')


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
ax.text(0.5,0.00688,get_stars(pnbpval), fontsize=12, ha='center')
ax.plot([0.1,0.9],[0.0069,0.0069], color = 'black')
#dmso to ck666
ax.text(1,0.00718,get_stars(ck666pval), fontsize=12, ha='center')
ax.plot([0.1,1.9],[0.0072,0.0072], color = 'black')
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
tstat, pnbpval = stats.mannwhitneyu(ctrlframe.aerresid.dropna().values, pnbframe.aerresid.dropna().values)
tstat, ck666pval = stats.mannwhitneyu(ctrlframe.aerresid.dropna().values, ck666frame.aerresid.dropna().values)
# reject, pvcorr = multipletests([pnbpval, ck666pval], method = 'bonferroni')[:2]
# pnbpval_adj, ck666pval_adj = pvcorr
print(f't test for Rsq between {treatments[0]} and {treatments[1]} is {pnbpval}')
print(f't test for Rsq between {treatments[0]} and {treatments[2]} is {ck666pval}')

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
ax.text(0.5,1.057,get_stars(pnbpval), fontsize=12, ha ='center')
ax.plot([0,1],[1.06,1.06], color = 'black')
#DMSO to CK666
ax.text(1,1.107,get_stars(ck666pval), fontsize=12, ha ='center')
ax.plot([0,2],[1.11,1.11], color = 'black')

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