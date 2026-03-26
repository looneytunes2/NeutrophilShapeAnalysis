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

treatments = ['DMSO','Para-Nitro-Blebbistatin','CK666']
whichpcs = [2,8]
config = load_config(microscope_type='confocal')
ntrans = config.db_params.ntrans
time_interval = config.im_params.time_interval
config._alignment = 'trajectory'
savedir = config.common.savedir
dbbsdir = savedir.joinpath('detailed_balance','separatedatabs')

df = pd.read_csv(dbbsdir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_bootstrapped_{ntrans}_Area_Enclosed_Linear_Reg.csv'), index_col = 0)
df = df[df.Treatment.isin(treatments)].copy()
df['Treatment'] = pd.Categorical(df.Treatment.to_list(), categories=treatments, ordered=True)


print(f'{treatments[0]} AER mean is {df[df.Treatment == treatments[0]].aer_coeff.mean()}'
          f' and median is {df[df.Treatment == treatments[0]].aer_coeff.median()}')
print(f'{treatments[1]} AER mean is {df[df.Treatment == treatments[1]].aer_coeff.mean()}'
          f' and median is {df[df.Treatment == treatments[1]].aer_coeff.median()}')
print(f'{treatments[2]} AER mean is {df[df.Treatment == treatments[2]].aer_coeff.mean()}'
          f' and median is {df[df.Treatment == treatments[2]].aer_coeff.median()}')


avgdf_filtered = filter_extremes_based_on_percentile(
    df,
    ['aer_coeff','aer_fit'],
    1)



############### CELL AVERAGES OF SIGNIFICANT METRICS #################################
colorlist = ['#d1b59b','#f7bebe','#faf191']
sns.set_palette(palette=colorlist)



#separate dataframes
ctrlframe = df[df.Treatment == treatments[0]]
pnbframe = df[df.Treatment == treatments[1]]
ck666frame = df[df.Treatment == treatments[2]]

tstat, pnbpval = stats.mannwhitneyu(ctrlframe.aer_coeff.values, pnbframe.aer_coeff.values)
tstat, ck666pval = stats.mannwhitneyu(ctrlframe.aer_coeff.values, ck666frame.aer_coeff.values)
# reject, pvcorr = multipletests([pnbpval, ck666pval], method = 'bonferroni')[:2]
# pnbpval_adj, ck666pval_adj = pvcorr
print(f'Mann Whitney U test for AER between {treatments[0]} and {treatments[1]} is {pnbpval}')
print(f'Mann Whitney U test for AER between {treatments[0]} and {treatments[2]} is {ck666pval}')


fig, ax = plt.subplots(1, 1, figsize=(4,5))#, sharex=True)
linewid = 2
# sns.swarmplot(data = avgcfdf, x='Treatment', y ='average_cf', color = 'grey', size = 3.5, alpha = 0.7, ax = ax)
sns.violinplot(x = 'Treatment', y='aer_coeff', data = avgdf_filtered, alpha = 0.4,
               linewidth = 0, inner = None, ax=ax, )
for ac in ax.collections:
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

#set y limit min at zero
ymin, ymax = ax.get_ylim()
# ax.set_ylim(0,ymax)

#dmso to bleb
ax.text(0.5,ymax-((ymax-ymin)*0.05),get_stars(pnbpval), fontsize=12, ha='center')
ax.plot([0.1,0.9],[ymax-((ymax-ymin)*0.05),ymax-((ymax-ymin)*0.05)], color = 'black')
#dmso to ck666
ax.text(1,ymax,get_stars(ck666pval), fontsize=12, ha='center')
ax.plot([0.1,1.9],[ymax, ymax], color = 'black')


### labels
ax.set_xlabel('', fontsize=20)
ax.tick_params('y', labelsize=10)
#modify the labels to put bleb in two lines
ax.set_xticks(range(len(treatments)))
ax.set_xticklabels([x[:11]+'\n'+x[11:] if x == 'Para-Nitro-Blebbistatin' else x for x in treatments], fontsize = 10)
#remove legends
ax.legend_ = None
ax.set_ylabel('Area Enclosing Rate (PC units²/sec)', fontsize=16)
#remove parts of box
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '_AER.png', dpi = 500, bbox_inches='tight')




print(f'{treatments[0]} AER R² is {df[df.Treatment == treatments[0]].aer_fit.mean()}'
          f' and median is {df[df.Treatment == treatments[0]].aer_fit.median()}')
print(f'{treatments[1]} AER R² is {df[df.Treatment == treatments[1]].aer_fit.mean()}'
          f' and median is {df[df.Treatment == treatments[1]].aer_fit.median()}')
print(f'{treatments[2]} R² mean is {df[df.Treatment == treatments[2]].aer_fit.mean()}'
          f' and median is {df[df.Treatment == treatments[2]].aer_fit.median()}')


##### stats for line fits
tstat, pnbpval = stats.mannwhitneyu(ctrlframe.aer_fit.dropna().values, pnbframe.aer_fit.dropna().values)
tstat, ck666pval = stats.mannwhitneyu(ctrlframe.aer_fit.dropna().values, ck666frame.aer_fit.dropna().values)
# reject, pvcorr = multipletests([pnbpval, ck666pval], method = 'bonferroni')[:2]
# pnbpval_adj, ck666pval_adj = pvcorr
print(f'Mann Whitney U test for Rsq between {treatments[0]} and {treatments[1]} is {pnbpval}')
print(f'Mann Whitney U test for Rsq between {treatments[0]} and {treatments[2]} is {ck666pval}')

fig, ax = plt.subplots(1, 1, figsize=(4,5))#, sharex=True)
linewid = 2
# sns.swarmplot(data = avgcfdf, x='Treatment', y ='average_cf', color = 'grey', size = 3.5, alpha = 0.7, ax = ax)
sns.violinplot(x = 'Treatment', y='aer_fit', data = avgdf_filtered,
               linewidth = 0, inner = None, ax=ax, )
for ac in ax.collections:
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


#set y limit min at zero
ymin, ymax = ax.get_ylim()
ax.set_ylim(0,ymax)



#DMSO to bleb
ax.text(0.5,ymax*0.965,get_stars(pnbpval), fontsize=12, ha ='center')
ax.plot([0,1],[ymax*0.97,ymax*0.97], color = 'black')
#DMSO to CK666
ax.text(1,ymax*0.99,get_stars(ck666pval), fontsize=12, ha ='center')
ax.plot([0,2],[ymax,ymax], color = 'black')


#ditch x axis label
ax.set_xlabel('', fontsize=20)
#change ticks
ax.tick_params('y', labelsize=10)
#modify the labels to put bleb in two lines
ax.set_xticks(range(len(treatments)))
ax.set_xticklabels([x[:11]+'\n'+x[11:] if x == 'Para-Nitro-Blebbistatin' else x for x in treatments], fontsize = 10)
#remove legends
ax.legend_ = None
ax.set_ylabel('Area Enclosing Rate R²', fontsize=16)
#remove parts of box
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '_Rsq.png', dpi = 500, bbox_inches='tight')