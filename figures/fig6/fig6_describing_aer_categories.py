# -*- coding: utf-8 -*-
"""
Created on Mon May 12 11:29:15 2025

@author: Aaron
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate, stats
import math
from statsmodels.stats.multitest import multipletests

treatments = ['Random']
time_interval = 5
whichpcs = [1,7]
derivthresh = 0.0007


basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_apply/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'

FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col = 0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())
#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
# open aers
allaers = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_raw_transition_aer_cf.csv', index_col = 0)

#merge aer and cf info
TotalFrame = FullFrame.merge(allaers[['aer','angular_velocity','cell']],left_on='cell',right_on='cell')
TotalFrame = TotalFrame.sort_values(['CellID','time'])



allcells = []
for i, cell in TotalFrame.groupby('CellID'):
    # ####running mean method
    # cell['aer_deriv'] = np.gradient(utils.running_mean_withna(cell.aer.cumsum(), 25), cell.time.values)
    cell = cell.reset_index(drop=True)
    ####interpolation method
    #### weight the points near gaps more
    diffs = np.diff(np.insert(cell.time.values,0,0))
    #get the indicies of jumps
    gaps = np.where(diffs>time_interval)[0]
    #add the indices before jumps
    gaps = np.concatenate((gaps,gaps-1))
    w = np.ones(len(cell)+1)
    w[gaps] = 3
    #interpolate
    tck, u = interpolate.splprep(np.array((np.insert(cell.time.values,0,0), np.insert(cell.aer.cumsum().values,0,0))), k=1, s=2, w=w)
    x, y = interpolate.splev(u, tck, der=0)
    #get derivative
    deriv = np.gradient(y, x)
    #threshold derivative based on the derivthresh above
    threshs = [deriv>=derivthresh, deriv<=-derivthresh]
    choices = ['up', 'down']
    statethresh = np.select(threshs, choices, default = 'zero')
    #add the derivative and the category to the dataframe
    cell.loc[:,'aer_deriv'] = deriv[1:]
    cell.loc[:,'aer_state'] = statethresh[1:]
    #append that cell
    allcells.append(cell)
    
derivframe = pd.concat(allcells).reset_index(drop=True)



########### only include columns of interest
includelist = ['Cell_Volume','Volume_Front_Ratio','Cell_SurfaceArea','Cell_Sphericity','Cell_Aspect_Ratio',
               'LengthAlongTrajectory','LengthAlongTrajectoryFront','LengthAlongTrajectoryRear','WidthAlongTrajectory','Cell_TotalAngle',
               'speed','Turn_Angle','PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10',
               'aer_state']#,'angular_velocity']


grouped = derivframe.groupby('aer_state')
results = []
for col in includelist:
    if col != 'aer_state':
        stat, pval  = stats.f_oneway(*grouped[col].apply(list).to_list())
        results.append({'stat':col,'pval':pval})
pdf = pd.DataFrame(results)
reject, pvcorr = multipletests(pdf['pval'],method='fdr_bh')[:2]
sigframe = pdf.iloc[reject].reset_index(drop=True)
#all significant comparisons
allsiglist = sigframe.stat.to_list()


derivframe.aer_state = pd.Categorical(derivframe.aer_state, categories=['down', 'zero', 'up'], ordered = True)
includelist = ['speed','Turn_Angle','Cell_Sphericity','Cell_TotalAngle','Cell_Volume']

ylabs = ['Instantaneous Speed (µm/sec)','Turn Angle (°)','Cell Sphericity', 'Long-Axis Total Angle (°)', "volume"]

# CoRo = math.ceil(math.sqrt(len(includelist)))
# row = 0
fig, axes = plt.subplots(1,5, figsize = (15,3))#, figsize=(3.5*CoRo,3*CoRo))#, sharex=True)
#set color palette
# colorlist = ['#4085e3','#edcd00']
# sns.set_palette(palette=colorlist)
linewid = 2

for i, ax in enumerate(axes.flatten()):

    # sns.swarmplot(data = avgcfdf, x='Treatment', y ='average_cf', color = 'grey', size = 3.5, alpha = 0.7, ax = ax)
    sns.violinplot(x = 'aer_state', y=includelist[i], data = derivframe,
                   linewidth = linewid, inner = None, ax=ax, )
    for co in ax.collections:
        co.set_edgecolor('black')
    sns.boxplot(x = 'aer_state', y=includelist[i], data = derivframe, width = 0.15, color = 'white',
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
    
    # ax.text(0.925,0.0405,'***', fontsize=12)
    # ax.set_ylim(0,0.042)
    ax.set_xlabel('', fontsize=12)
    # ax.tick_params('y', labelsize=10)
    #modify the labels to put bleb in two lines
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize = 10)
    #remove legends
    ax.legend_ = None
    ax.set_ylabel(ylabs[i], fontsize=16)
    #remove parts of box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')



