# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 21:48:14 2025

@author: Aaron
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

whichpcs = [1,7]
treatments = ['DMSO','Para-Nitro-Blebbistatin']
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_smooth/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'Para-Nitro-Blebbistatin/'

df = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_Area_Enclosing_Rates.csv', index_col = 0)
df['Treatment'] = pd.Categorical(df.Treatment.to_list(), categories=treatments, ordered=True)

avgcfs = df.groupby(['Treatment','iter']).mean().reset_index()
#change units from degrees/sec to minutes/cycle
avgcfs['cycle_freq'] = 1/(avgcfs.angular_velocity*60/360)

#remove the upper and lower 1% of the data
oneper = round(len(avgcfs)/2*0.05)
dropind = []
for treat, trdf in avgcfs.groupby('Treatment'):
    dropind.extend(trdf.cycle_freq.sort_values(ascending = True)[:oneper].index.to_list())
    dropind.extend(trdf.cycle_freq.sort_values(ascending = False)[:oneper].index.to_list())
avgdflessoneper = avgcfs.drop(index = dropind)

############### CELL AVERAGES OF SIGNIFICANT METRICS #################################
# colorlist = [list(sns.color_palette('pastel').as_hex())[i] for i in [0,3,8]]
colorlist = ['#4085e3','#d93434']
sns.set_palette(palette=colorlist)


fig, ax = plt.subplots(1, 1, figsize=(4,5))#, sharex=True)
linewid = 2
# sns.swarmplot(data = avgcfdf, x='Treatment', y ='average_cf', color = 'grey', size = 3.5, alpha = 0.7, ax = ax)
sns.violinplot(x = 'Treatment', y='cycle_freq', data = avgdflessoneper,
               linewidth = linewid, inner = None, ax=ax, )
ax.collections[0].set_edgecolor('black')
ax.collections[1].set_edgecolor('black')
sns.boxplot(x = 'Treatment', y='cycle_freq', data = avgdflessoneper, width = 0.15, color = 'white',
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
                'linewidth': linewid,
                'color': 'black'
                },
            capprops={
                'linewidth': linewid,
                'color': 'black'
                },
            ax=ax)
# ax.set_ylim(0,300)
ax.set_xlabel('', fontsize=20)
ax.tick_params('y', labelsize=10)
#modify the labels to put bleb in two lines
ax.set_xticklabels(ax.get_xticklabels(), fontsize = 10)
#remove legends
ax.legend_ = None
ax.set_ylabel('Average Cycling Period (min)', fontsize=16)

plt.tight_layout()

plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')
