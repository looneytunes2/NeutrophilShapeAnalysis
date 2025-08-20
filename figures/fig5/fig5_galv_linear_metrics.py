# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 21:00:29 2025

@author: Aaron
"""



import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm 
from statsmodels.formula.api import ols 
from CustomFunctions import linear_cycle_utils, utils, DetailedBalance
import matplotlib.pyplot as plt
import seaborn as sns
import os
import multiprocessing


#get directories and open separated datasets

treatments = ['Random','Galvanotaxis']
time_interval = 10 #sec/frame


#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'galv/'
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())
#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
#limit data to the galv experiments
TotalFrame = FullFrame[FullFrame.Treatment.isin(treatments)].copy()
TotalFrame.loc[:,'Treatment'] = pd.Categorical(TotalFrame.Treatment.to_list(), categories=treatments, ordered=True)

origin = [7, 7]
whichpcs = [1,7]
binrange = 20
direction = 'clockwise'
zerostart = 'left'



angframe = linear_cycle_utils.linearize_cycle_continuous(
            TotalFrame, 
            centers,
            origin, 
            whichpcs,
            zerostart,
            direction,)

angframe =  linear_cycle_utils.bin_angular_coord(
        angframe,
        whichpcs,
        binrange,
        )


#### calculate protrusion and retraction speeds
prsplist = []
for i, cells in angframe.groupby('CellID'):
    cells, runs = utils.get_consecutive_timepoints(cells, 'time', time_interval)
    for r in runs:
        tempcell = cells.iloc[r]
        tempcell.loc[:,'protrusion_speed'] = tempcell.LengthAlongTrajectoryFront.diff()
        tempcell.loc[:,'retraction_speed'] = abs(tempcell.LengthAlongTrajectoryRear).diff()
        prsplist.append(tempcell)
angframe = pd.concat(prsplist).reset_index(drop=True)



##### open aer and cf on the raw CGPS transitions
allaers = pd.read_csv(savedir + f'PC{whichpcs[0]}-PC{whichpcs[1]}_raw_transition_aer_cf.csv', index_col = 0)
#merge aer and cf info
angframe = angframe.merge(allaers[['aer','angular_velocity','cell']],left_on='cell',right_on='cell')



########### only include columns of interest
includelist = ['Treatment','Cell_Volume','Volume_Front_Ratio','Cell_SurfaceArea','Cell_Sphericity','Cell_Aspect_Ratio',
               'LengthAlongTrajectory','LengthAlongTrajectoryFront','LengthAlongTrajectoryRear','WidthAlongTrajectory',
               'speed','directional_autocorrelation','PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8',
               'PC1_PC7_Continuous_Angular_Bins','protrusion_speed','retraction_speed','Turn_Angle']#,,'angular_velocity']


#iterate through remaining columns and do two-way anova with treatment and cycle
reslist = []
for col in includelist:
    if col not in  ['PC1_PC7_Continuous_Angular_Bins','Treatment']:
        tempframe = angframe[['PC1_PC7_Continuous_Angular_Bins','Treatment', col]].dropna().reset_index(drop=True)
        model = ols(f'{col} ~ C(Treatment) + C(PC1_PC7_Continuous_Angular_Bins) + C(Treatment):C(PC1_PC7_Continuous_Angular_Bins)', 
                    data=tempframe).fit() 
        result = sm.stats.anova_lm(model, type=2)
        result['Factor'] = [col]*len(result)
        reslist.append(result)
pvdf = pd.concat(reslist)
pvdf = pvdf[pvdf.index=='C(Treatment):C(PC1_PC7_Continuous_Angular_Bins)'].reset_index(drop=True)
reject, pvcorr = multipletests(pvdf['PR(>F)'],method='fdr_bh')[:2]
sigframe = pvdf.iloc[reject]
siglist = sigframe.Factor.to_list()



fig, ax = plt.subplots(figsize=(3.75,3))#CoRo, CoRo, figsize=(3.5*CoRo,3*CoRo))#, sharex=True)
#set color palette
colorlist = ['#a244c9','#6cc46d']
sns.set_palette(palette=colorlist)

sns.lineplot(data = angframe, x=f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins', 
             y = siglist[0], hue ='Treatment', #palette = colorlist, 
             ax = ax)
ax.set_ylabel('Cell Volume (µm³)', fontsize = 18)
ax.set_xlabel('Angular Bins (°)')
ax.legend_ = None

#remove the upper and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#add significance stars with adjusted p values
pv = pvcorr[pvdf[pvdf.Factor==siglist[0]].index]
if pv < 0.001:
    stars = '***'
elif pv < 0.01:
    stars = '**'
elif pv < 0.05:
    stars = '*'
else:
    stars = 'n.s.'
ymin,ymax = ax.get_ylim()
ax.text(180, ymax-(ymax-ymin)*0.03, stars, fontsize = 10, ha = 'center')


plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')





fig, ax = plt.subplots(figsize=(3.75,3))#CoRo, CoRo, figsize=(3.5*CoRo,3*CoRo))#, sharex=True)

sns.lineplot(data = angframe, x=f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins', 
             y = siglist[1], hue ='Treatment', #palette = colorlist, 
             ax = ax)
ax.set_ylabel(siglist[1], fontsize = 18)
ax.set_xlabel('Angular Bins (°)')
ax.legend_ = None

#remove the upper and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#add significance stars with adjusted p values
pv = pvcorr[pvdf[pvdf.Factor==siglist[0]].index]
if pv < 0.001:
    stars = '***'
elif pv < 0.01:
    stars = '**'
elif pv < 0.05:
    stars = '*'
else:
    stars = 'n.s.'
ymin,ymax = ax.get_ylim()
ax.text(180, ymax-(ymax-ymin)*0.03, stars, fontsize = 10, ha = 'center')


plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '_extras.png', dpi = 500, bbox_inches='tight')