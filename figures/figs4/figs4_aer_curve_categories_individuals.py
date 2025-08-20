# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 14:21:11 2025

@author: Aaron
"""



import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate, stats
import math
from CustomFunctions import utils
from statsmodels.stats.multitest import multipletests

derivthresh = 0.0007
treatments = ['Random','Galvanotaxis']
scale = 4 # scale of the plot
time_interval = 5
whichpcs = [1,7]
ntrans = 1
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_apply/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'


FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col = 0)
# open aers
allaers = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_raw_transition_aer_cf.csv', index_col = 0)

#merge aer and cf info
TotalFrame = FullFrame.merge(allaers[['aer','angular_velocity','cell']],left_on='cell',right_on='cell')
TotalFrame = TotalFrame.sort_values(['CellID','time'])



allcells = []
for i, cell in TotalFrame.groupby('CellID'):
    # ####running mean method
    # cell['aer_deriv'] = np.gradient(utils.running_mean_withna(cell.aer.cumsum(), 25), cell.time.values)
    
    cell, tck , w = utils.get_aer_state(cell, time_interval)
    #append that cell
    allcells.append(cell)
    
derivframe = pd.concat(allcells).reset_index(drop=True)




########### only include columns of interest
includelist = ['Cell_Volume','Volume_Front_Ratio','Cell_SurfaceArea','Cell_Sphericity','Cell_Aspect_Ratio',
                'LengthAlongTrajectory','LengthAlongTrajectoryFront','LengthAlongTrajectoryRear','WidthAlongTrajectory',
                'speed','directional_autocorrelation','Turn_Angle','PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8',
                'aer_state']#,'angular_velocity']

grouped = derivframe.groupby('aer_state')
results = []
for col in includelist:
    if col != 'aer_state':
        stat, pval  = stats.f_oneway(*np.array(grouped[col].apply(list).to_list()))
        results.append(pd.DataFrame({'stat':col,'pval':pval}, index=[0]))
pdf = pd.concat(results)
reject, pvcorr = multipletests(pdf['pval'],method='fdr_bh')[:2]
sigframe = pdf.iloc[reject]





includelist = ['speed','directional_autocorrelation']
ylabels = ['Instantaneous Speed (µm/sec)','Persistence']
xlabels = ['Decreasing','Unchanging','Increasing']


############### PLOT INDIVIDUAL BOXPLOTS
#define the colors to make the meshes
set1 = plt.cm.Set1
set2 = plt.cm.Set2
set3 = plt.cm.Set3
cmap = matplotlib.colors.ListedColormap(list(set3.colors)+[set2.colors[-2]] + [set1.colors[-1]])

fig, axes = plt.subplots(1,len(includelist),figsize = (5*len(includelist),3))
linewid = 2

flierprops = dict(marker='.', markersize=0.75)
for i, ax in enumerate(axes.flatten()):
    sns.boxplot(x = 'aer_state', y=includelist[i], data = derivframe, hue = 'CellID', 
                   palette = cmap.colors, order = ['decreasing','unchanging','increasing'],
                showcaps=False,
                boxprops={
                    'zorder': 2
                    },
                whiskerprops={
                    'linewidth': 0,
                    },
                flierprops={
                    'marker': '',
                    },
                ax=ax)


    sns.stripplot(data = derivframe, x = 'aer_state', y = includelist[i], hue = 'CellID',
                  dodge = True, jitter = False, s = 0.8, color = 'gray', alpha = 0.7,
                  zorder = 1, ax = ax)

    # #plot the average of the averages
    # avgs = derivframe.groupby(['aer_state']).mean()
    # for n, d in enumerate(['decreasing','unchanging','increasing']):
    #     avgavg = avgs[includelist[i]]
    #     ax.plot([n-0.43,n+0.43],[avgavg[d], avgavg[d]], ls = '--', color = ['#d14c45','#a8a8a8','#3e88ad'][n], alpha = 0.4)
    
    # ax.text(0.925,0.33,'***', fontsize=12)
    # ax.set_ylim(0,60)
    ax.set_xticklabels(xlabels, fontsize=10)

    #remove legends
    ax.legend_ = None
    ax.set_ylabel(ylabels[i], fontsize=12)
    ax.set_xlabel('')

    #remove parts of box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')




##### individual cells separately
for n in range(len(includelist)):
    
    fig, axes = plt.subplots(2,int(len(cmap.colors)/2),figsize = (7, 2.5*len(includelist)),
                             sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        #get single cell
        curcell = derivframe.CellID.unique()[i]
        cellframe = derivframe[derivframe.CellID==curcell].copy()
        
        
        
        sns.boxplot(x = 'aer_state', y=includelist[n], data = cellframe, 
                       color = cmap.colors[i], order = ['decreasing','unchanging','increasing'],
                    showcaps=False,
                    boxprops={
                        'zorder': 2
                        },
                    whiskerprops={
                        'linewidth': 0,
                        },
                    flierprops={
                        'marker': '',
                        },
                    ax=ax)
    
    
        sns.stripplot(data = cellframe, x = 'aer_state', y = includelist[n], #hue = 'CellID',
                      dodge = True, jitter = False, s = 0.8, color = 'gray', alpha = 0.7,
                      zorder = 1, ax = ax)
    
        
        if i >= int(len(cmap.colors)/2):
            ax.set_xticklabels(xlabels, fontsize=10)
        else:
            ax.set_xticklabels(['']*len(ax.get_xticklabels()))
            
        if i == 0 or i == int(len(cmap.colors)/2):
            ax.set_ylabel(ylabels[n], fontsize=12)
        else:
            ax.set_ylabel('')
            
        #remove legends
        ax.legend_ = None
        ax.set_xlabel('')
    
        #remove parts of box
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


