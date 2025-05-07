# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 11:46:29 2025

@author: Aaron
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

whichpcs = [1,7]
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'

df = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_Area_Enclosing_Rates.csv', index_col = 0)


avgdf = df.groupby('iter').mean()
#change units from degrees/sec to minutes/cycle
avgdf['cycle_freq'] = 1/(avgdf.angular_velocity*60/360)

#remove highest and lowest 1% of values
oneper = round(len(avgdf)*0.01)    
upperoneper = avgdf.cycle_freq.sort_values(ascending = True)[:oneper].index.to_list()
loweroneper = avgdf.cycle_freq.sort_values(ascending = False)[:oneper].index.to_list()
avgdflessoneper = avgdf.drop(index = upperoneper+loweroneper)


fig, ax = plt.subplots(1, 1, figsize=(1.8,5))#, sharex=True)
linewid= 1.5

sns.violinplot(y=avgdflessoneper['cycle_freq'], color = 'lightgreen',
               linewidth = linewid, inner = None, ax=ax, )
ax.collections[0].set_edgecolor('black')
sns.boxplot(y=avgdflessoneper['cycle_freq'], width = 0.15, color = 'white', 
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
#tick label size
ax.tick_params('y', labelsize=12)
#remove legends
ax.legend_ = None
#adjust ylabel
ax.set_ylabel('Average Cycle Period (min)', fontsize=16)
#set plot limits
# ax.set_ylim(0,13)
#remove parts of box
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#remove x tick
ax.set_xticks([])

plt.tight_layout()

plt.savefig(__file__.split('.')[0] + '_cf.png', dpi = 500, bbox_inches='tight')


fig, ax = plt.subplots(1, 1, figsize=(1.8,5))#, sharex=True)
linewid= 1.5
sns.violinplot(y=avgdflessoneper['aer'], color = 'lightgreen',
               linewidth = linewid, inner = None, ax=ax, )
ax.collections[0].set_edgecolor('black')
sns.boxplot(y=avgdflessoneper['aer'], width = 0.15, color = 'white', 
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
#tick label size
ax.tick_params('y', labelsize=12)
#remove legends
ax.legend_ = None
#adjust ylabel
ax.set_ylabel('Area Enclosing Rate (PC units²/min)', fontsize=16)
#set plot limits
# ax.set_ylim(0,13)
#remove parts of box
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#remove x tick
ax.set_xticks([])

plt.tight_layout()

plt.savefig(__file__.split('.')[0] + '_aer.png', dpi = 500, bbox_inches='tight')

