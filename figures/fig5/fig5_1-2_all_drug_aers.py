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
treatments = ['DMSO','Para-Nitro-Blebbistatin','CK666']
basedir = Path('E:/Aaron/Combined_37C_Confocal_PCA_planar')
datadir = basedir.joinpath('Data_and_Figs')
savedir = basedir.joinpath('Detailed_Balance')

df = pd.read_csv(savedir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_bootstrapped_{ntrans}_Area_Enclosing_Rates.csv'), index_col = 0)
df = df[df.Treatment.isin(treatments)].copy()
df['Treatment'] = pd.Categorical(df.Treatment.to_list(), categories=treatments, ordered=True)


dflist = []
for i, t in df.groupby(['Treatment','iter']):
    t = t.rename(columns = {'cumulative_time':'time'})
    t = t.sort_values('time').reset_index(drop=True)
    #fit aer
    rate_fit_dict = utils.fit_rates_linear(t,time_interval,['aer'])
    rate_fit_dict.update({
        'Treatment':i[0],'iter':i[1],
         })
    dflist.append(rate_fit_dict)
avgdf = pd.DataFrame(dflist)


print(f'{treatments[0]} AER mean is {avgdf[avgdf.Treatment == treatments[0]].aercoef.mean()}'
          f' and median is {avgdf[avgdf.Treatment == treatments[0]].aercoef.median()}')
print(f'{treatments[1]} AER mean is {avgdf[avgdf.Treatment == treatments[1]].aercoef.mean()}'
          f' and median is {avgdf[avgdf.Treatment == treatments[1]].aercoef.median()}')
print(f'{treatments[2]} AER mean is {avgdf[avgdf.Treatment == treatments[2]].aercoef.mean()}'
          f' and median is {avgdf[avgdf.Treatment == treatments[2]].aercoef.median()}')


avgdf_filtered = filter_extremes_based_on_percentile(
    avgdf,
    ['aer_coeff','aer_fit'],
    1)



############### CELL AVERAGES OF SIGNIFICANT METRICS #################################
colorlist = ['#d1b59b','#f7bebe','#faf191']
sns.set_palette(palette=colorlist)


#separate dataframes
ctrlframe = avgdf[avgdf.Treatment == treatments[0]]
pnbframe = avgdf[avgdf.Treatment == treatments[1]]
ck666frame = avgdf[avgdf.Treatment == treatments[2]]

tstat, pnbpval = stats.mannwhitneyu(ctrlframe.aercoef.values, pnbframe.aercoef.values)
tstat, ck666pval = stats.mannwhitneyu(ctrlframe.aercoef.values, ck666frame.aercoef.values)
# reject, pvcorr = multipletests([pnbpval, ck666pval], method = 'bonferroni')[:2]
# pnbpval_adj, ck666pval_adj = pvcorr
print(f'Mann Whitney U test for AER between {treatments[0]} and {treatments[1]} is {pnbpval}')
print(f'Mann Whitney U test for AER between {treatments[0]} and {treatments[2]} is {ck666pval}')


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
ax.text(0.5,0.0248,get_stars(pnbpval), fontsize=12, ha='center')
ax.plot([0.1,0.9],[0.0248,0.0248], color = 'black')
#dmso to ck666
ax.text(1,0.0258,get_stars(ck666pval), fontsize=12, ha='center')
ax.plot([0.1,1.9],[0.0258,0.0258], color = 'black')

#set y limit min at zero
ymin, ymax = ax.get_ylim()
ax.set_ylim(0,ymax)
### labels
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




print(f'{treatments[0]} AER R² is {avgdf[avgdf.Treatment == treatments[0]].aerresid.mean()}'
          f' and median is {avgdf[avgdf.Treatment == treatments[0]].aerresid.median()}')
print(f'{treatments[1]} AER R² is {avgdf[avgdf.Treatment == treatments[1]].aerresid.mean()}'
          f' and median is {avgdf[avgdf.Treatment == treatments[1]].aerresid.median()}')
print(f'{treatments[2]} R² mean is {avgdf[avgdf.Treatment == treatments[2]].aerresid.mean()}'
          f' and median is {avgdf[avgdf.Treatment == treatments[2]].aerresid.median()}')


##### stats for line fits
tstat, pnbpval = stats.mannwhitneyu(ctrlframe.aerresid.dropna().values, pnbframe.aerresid.dropna().values)
tstat, ck666pval = stats.mannwhitneyu(ctrlframe.aerresid.dropna().values, ck666frame.aerresid.dropna().values)
# reject, pvcorr = multipletests([pnbpval, ck666pval], method = 'bonferroni')[:2]
# pnbpval_adj, ck666pval_adj = pvcorr
print(f'Mann Whitney U test for Rsq between {treatments[0]} and {treatments[1]} is {pnbpval}')
print(f'Mann Whitney U test for Rsq between {treatments[0]} and {treatments[2]} is {ck666pval}')

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


#set y limit min at zero
ymin, ymax = ax.get_ylim()
# ax.set_ylim(0,ymax)

#bar placement adjustment
barinc = (ymax-ymin)*0.18
starinc = (ymax-ymin)*0.001

#DMSO to bleb
ax.text(0.5,ymax*0.995,get_stars(pnbpval), fontsize=12, ha ='center')
ax.plot([0,1],[ymax,ymax], color = 'black')
#DMSO to CK666
ax.text(1,ymax*0.995+barinc,get_stars(ck666pval), fontsize=12, ha ='center')
ax.plot([0,2],[ymax+barinc,ymax+barinc], color = 'black')

ymin, ymax = ax.get_ylim()
ax.set_ylim(0,ymax)

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