# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:33:38 2025

@author: Aaron
"""

import sys
sys.path.append('C:/Users/Aaron/NeutrophilShapeAnalysis')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.font_manager import FontProperties
from neutrophil_shape.config.loader import load_config

config = load_config(microscope_type = 'confocal')


####### load common directories and data
dirlist = ['Combined_37C_Confocal_PCA_shape','Combined_37C_Confocal_PCA_s5','Combined_37C_Confocal_PCA_planar']
alignlist = ['Shape Only','Trajectory + Shape', 'Trajectory Only']
colorlist = cm.Set2.colors[:3][::-1]

dflist = []
for d, ali in enumerate(['shape','trajectory_shape','trajectory']):
    #set alignment
    config._alignment = ali
    time_interval = config.im_params.time_interval
    ntrans = config.db_params.ntrans
    allorigins = config.db_params.origins
    savedir = config.common.savedir
    datadir = savedir / 'shape_data'
    centers = pd.read_csv(datadir.joinpath('PC_bin_centers.csv'), index_col=0)
    binlist = centers.columns.to_list()
    dbdir = savedir / 'detailed_balance'
    dbbsdir = dbdir / 'alldatabs'
    file_list = dbbsdir.glob(f'*bootstrapped_{ntrans}_Area_Enclosed_Linear_Reg.csv')
    for f in file_list:
        tempdf = pd.read_csv(f, index_col = 0)
        firstpc, secondpc = f.name.split('_')[0].split('-')
        tempdf['first_pc'] = firstpc
        tempdf['second_pc'] = secondpc
        tempdf['Alignment Method'] = alignlist[d]
        dflist.append(tempdf)

df = pd.concat(dflist, ignore_index = True)
##drop nans
df = df[~df.aer_fit.isna()]

## add cycle period not just angular velocity
df['cycle_period'] = 360/(df.angular_velocity_coeff.abs()*60) ## (degrees/cycle)/((degrees/sec)*(sec/min)) = min/cycle


### get means and SEMs 
sems = df.groupby(['Alignment Method','first_pc','second_pc'])[['aer_coeff','cycle_period']].sem().reset_index()
means = df.groupby(['Alignment Method','first_pc','second_pc'])[['aer_coeff','cycle_period']].mean().reset_index()
##get abs value of the aers
means['aer_abs'] = means.aer_coeff.abs()

#get the colors for the errorbars
colors = [colorlist[alignlist.index(t)] for t in means['Alignment Method']]

#plot the scattered data
fig, ax = plt.subplots()
scat = sns.scatterplot(data = means, x='aer_abs', y = 'cycle_period', hue = 'Alignment Method',
                palette = colorlist, ax = ax, zorder = 10)
scat.set_yscale("log")
# ax.errorbar(means.aer_abs.values, means.cycle_period.values,
#             xerr=sems.aer_coeff.values, yerr=sems.cycle_period.values, c=colors, ls = 'none')

for xi, yi, xe, ye, c in zip(means.aer_abs.values, means.cycle_period.values,
                              sems.aer_coeff.values, sems.cycle_period.values, colors):
    ax.errorbar(
        xi, yi,
        xerr=xe, yerr=ye,
        fmt='none',
        ecolor=c,
        elinewidth=1,
        # alpha = 0.6,
        # capsize=4,
        zorder=2
    )

#Change legend title and size
leg = ax.legend_
leg.set_title(title = 'Alignment Method', prop = FontProperties(size=12))

ax.set_xlabel("|Mean Area Enclosing Rate| (PC units²/sec)", fontsize = 16)         
ax.set_ylabel("Mean Cycle Period (min/cycle)", fontsize = 16)

# remove upper and right box lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)




plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)








########## WHAT HAPPENS IF YOU JUST DROP THE SINGLE HIGHEST VALUE FROM ALL CGPSs
noex = df.groupby(['Alignment Method','first_pc','second_pc']).apply(lambda x: x.sort_values('cycle_period').iloc[:-1]).reset_index(drop = True)

noex_sems = noex.groupby(['Alignment Method','first_pc','second_pc'])[['aer_coeff','cycle_period']].sem().reset_index()
noex_means = noex.groupby(['Alignment Method','first_pc','second_pc'])[['aer_coeff','cycle_period']].mean().reset_index()
##get abs value of the aers
noex_means['aer_abs'] = noex_means.aer_coeff.abs()



#get the colors for the errorbars
colors = [colorlist[alignlist.index(t)] for t in noex_means['Alignment Method']]

#plot the scattered data
fig, ax = plt.subplots()
scat = sns.scatterplot(data = noex_means, x='aer_abs', y = 'cycle_period', hue = 'Alignment Method',
                palette = colorlist, ax = ax, zorder = 10)
scat.set_yscale("log")
# ax.errorbar(means.aer_abs.values, means.cycle_period.values,
#             xerr=sems.aer_coeff.values, yerr=sems.cycle_period.values, c=colors, ls = 'none')

for xi, yi, xe, ye, c in zip(noex_means.aer_abs.values, noex_means.cycle_period.values,
                              noex_sems.aer_coeff.values, noex_sems.cycle_period.values, colors):
    ax.errorbar(
        xi, yi,
        xerr=xe, yerr=ye,
        fmt='none',
        ecolor=c,
        elinewidth=1,
        # alpha = 0.6,
        # capsize=4,
        zorder=2
    )

#Change legend title and size
leg = ax.legend_
leg.set_title(title = 'Alignment Method', prop = FontProperties(size=12))

ax.set_xlabel("|Mean Area Enclosing Rate| (PC units²/sec)", fontsize = 16)         
ax.set_ylabel("Mean Cycle Period (min/cycle)", fontsize = 16)

# remove upper and right box lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)




df[(df['Alignment Method']=='Trajectory Only') &
   (df.first_pc == 'PC3') &
   (df.second_pc == 'PC4')]


