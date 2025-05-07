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
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
import multiprocessing


#get directories and open separated datasets

treatments = ['DMSO','Para-Nitro-Blebbistatin']
time_interval = 10 #sec/frame


#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'Para-Nitro-Blebbistatin/'
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())
#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
#limit data to the Para-Nitro-Blebbistatin experiments
TotalFrame = FullFrame[FullFrame.Experiment == 'Drug']
dates = [20240624,20240626,20240701,20241125,20241126,20241127]
TotalFrame = TotalFrame[TotalFrame.Date.isin(dates)]
TotalFrame['Treatment'] = pd.Categorical(TotalFrame.Treatment.to_list(), categories=treatments, ordered=True)

origin = [7, 6]
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


##### quickly calculate aer and cf on the raw CGPS transitions
if os.path.exists(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_transitions_separated.csv'):
    raw_trans = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_transitions_separated.csv', index_col = 0)
    ############# measure aer and cycling frequencies ###########
    #add specific scaling
    xyscaling = [centers[f'PC{whichpcs[0]}'].diff().mean(),centers[f'PC{whichpcs[1]}'].diff().mean()]
    #set the origin to the actual center
    center = [round(nbins/2)]*2
    results = []
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=60)
        for i, cells in raw_trans.groupby('CellID'):
            cells, runs = utils.get_consecutive_timepoints(cells, 'frame',1)
            for r in runs:
                cell = cells.iloc[r].reset_index(drop=True)
                result = pool.apply_async(DetailedBalance.get_area_enclosing_rate, args = (
                    cell,
                    nbins,
                    xyscaling,
                    center,
                    ))
                results.append(result)
        pool.close()
        pool.join()
    results = [r.get() for r in results]
    allaers = pd.concat(results).reset_index(drop=True)
allaers['cell'] = [c+f'_frame_{int(f)}' for c, f in allaers[['CellID','frame']].values]
#merge aer and cf info
angframe = angframe.merge(allaers[['aer','angular_velocity','cell']],left_on='cell',right_on='cell')


########### only include columns of interest
includelist = ['Treatment','Cell_Volume','Volume_Front_Ratio','Cell_SurfaceArea','Cell_Sphericity','Cell_Aspect_Ratio',
               'LengthAlongTrajectory','LengthAlongTrajectoryFront','LengthAlongTrajectoryRear','WidthAlongTrajectory',
               'speed','directional_autocorrelation','Turn_Angle','PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10',
               'PC1_PC7_Continuous_Angular_Bins','protrusion_speed','retraction_speed']#,'angular_velocity']


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
reject, pvcorr = multipletests(pvdf['PR(>F)'],method='bonferroni')[:2]
sigframe = pvdf.iloc[reject]
siglist = sigframe.Factor.to_list()


CoRo = math.ceil(math.sqrt(len(sigframe)))
row = 0
fig, axes = plt.subplots(CoRo, CoRo, figsize=(3.5*CoRo,3*CoRo))#, sharex=True)
#set color palette
colorlist = ['#4085e3','#d93434']
sns.set_palette(palette=colorlist)
for i, ax in enumerate(axes.flatten()):
    if i<len(sigframe):
        if siglist[i] in [f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins','Treatment']:
            ax.remove()
            continue
        sns.lineplot(data = angframe, x=f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins', 
                     y = siglist[i], hue ='Treatment', #palette = colorlist, 
                     ax = ax)
        ax.set_ylabel(siglist[i], fontsize = 22)
        ax.legend_ = None
    elif i==len(sigframe):
        #add a legend to one of the empty subplots
        ax.axis("off")
        handles, labels = axes[0,0].get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper center')
        
    else:
        ax.remove()

plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')

