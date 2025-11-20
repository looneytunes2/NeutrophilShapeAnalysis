# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:26:03 2025

@author: Aaron
"""



import pandas as pd
import numpy as np
import scipy.stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm 
from statsmodels.formula.api import ols 
import matplotlib.pyplot as plt
import seaborn as sns
from CustomFunctions import utils


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

#get directories and open separated datasets
treatments = ['DMSO','Para-Nitro-Blebbistatin','CK666']
time_interval = 10 #sec/frame


#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'

FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())
#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
#limit data to the Para-Nitro-Blebbistatin experiments
TotalFrame = FullFrame[FullFrame.Treatment.isin(treatments)].copy()
TotalFrame['Treatment'] = pd.Categorical(TotalFrame.Treatment.to_list(), categories=treatments, ordered=True)



#### calculate protrusion and retraction speeds
prsplist = []
for i, cells in TotalFrame.groupby('CellID'):
    cells, runs = utils.get_consecutive_timepoints(cells, 'time', time_interval)
    for r in runs:
        tempcell = cells.iloc[r].copy()
        tempcell['protrusion_speed'] = tempcell.LengthAlongTrajectoryFront.diff()
        tempcell['retraction_speed'] = tempcell.LengthAlongTrajectoryRear.diff()
        prsplist.append(tempcell)
TotalFrame = pd.concat(prsplist).reset_index(drop=True)
###filter the data for only cells that I have 10 or more frames of
TotalFrame_filtered = TotalFrame[TotalFrame['CellID'].map(TotalFrame['CellID'].value_counts()) >= 10].copy()

#get cell averages
avgdf_filtered = TotalFrame_filtered.groupby(['Treatment','CellID']).mean().reset_index(level='Treatment')
#change the rear length to abs
avgdf_filtered.loc[:,'LengthAlongTrajectoryRear'] = avgdf_filtered.LengthAlongTrajectoryRear.abs()

############### get list of metrics that are significant ttest of CELL AVERAGES ############
metriclist = ['Treatment','Cell_Volume','Volume_Front_Ratio','Cell_SurfaceArea','Cell_Sphericity','Cell_Aspect_Ratio',
               'LengthAlongTrajectory','LengthAlongTrajectoryFront','LengthAlongTrajectoryRear','WidthAlongTrajectory',
               'speed','protrusion_speed','retraction_speed'] #'Turn_Angle',
pclist = [x for x in TotalFrame.columns.to_list() if 'PC' in x and 'bin' not in x]
includelist = metriclist + pclist

#iterate through remaining columns and do two-way ttest between each drug and control
reslist = []
for col in includelist:
    for t in treatments[1:]:
        if col not in ['Treatment']:
            tempframe = avgdf_filtered[['Treatment', col]].dropna()
            tstat, pval = scipy.stats.ttest_ind(
                tempframe.loc[tempframe.Treatment=='DMSO', col].values,
                tempframe.loc[tempframe.Treatment==t, col].values
                )
            reslist.append({'metric': col, 'Treatment': t, 'pvalue': pval})
pvdf = pd.DataFrame(reslist).sort_values('Treatment')


#separately test statistics by treatment and metrics vs PCs
bothtreatsig = []
for tr in treatments[1:]:
    tpvdf = pvdf[pvdf.Treatment == tr].copy().reset_index(drop=True)
    metricdf = tpvdf[tpvdf.metric.isin(metriclist)].copy()
    pcdf = tpvdf[tpvdf.metric.isin(pclist)].copy()
    #correct pvalues for metrics
    metric_reject, metrics_pvcorr = multipletests(metricdf['pvalue'],method='fdr_bh')[:2]
    #correct pvalues for PCs
    PC_reject, PC_pvcorr = multipletests(pcdf['pvalue'],method='fdr_bh')[:2]
    #combine rejected hypotheses
    tempsigframe = pd.concat((metricdf[metric_reject], pcdf[PC_reject]), ignore_index=True)
    #add corrected PCs
    tempsigframe['pvcorr'] = np.concatenate((metrics_pvcorr[metric_reject], PC_pvcorr[PC_reject]))
    bothtreatsig.append(tempsigframe)
## combine treatments
sigframe = pd.concat(bothtreatsig, ignore_index=True)

#all significant comparisons
allsiglist = sigframe.metric.unique()




siglist = ['speed','PC1','PC7','Volume_Front_Ratio']

ylabels = [
           'Instantaneous\nSpeed (µm/s)',
           'PC1',
           'PC7',
           'Front-Rear\nVolume Ratio',]

############### CELL AVERAGES OF SIGNIFICANT METRICS #################################
colorlist = ['#9c836b','#faa7a7','#faf191']
sns.set_palette(palette=colorlist)

scale = int(len(siglist)/2)
linewid= 1.2

fig, axes = plt.subplots(1,len(siglist),figsize=(12,3))
for i, sig in enumerate(siglist):
    ax = axes.flatten()[i]
    sns.swarmplot(x = 'Treatment', y = sig, data = avgdf_filtered, size = 1.3, ax = ax, zorder = 1)
    sns.boxplot(x = 'Treatment', y = sig, data = avgdf_filtered, width = 0.5,
                boxprops={
                    'fill': False,
                    'linewidth': linewid,
                    'edgecolor': 'black',
                    'zorder':2
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
                    'linewidth': 0,
                    'color': 'black'
                    },
                showfliers=False, ax = ax, zorder = 2)
    
    #set ylim min to zero
    # ax.set_ylim(0, ax.get_ylim()[1])
    #tick stuff
    ax.set_ylabel(ylabels[i], fontsize = 16)#, labelpad=-0.5)
    ax.set_xlabel('')
    ax.set_xticklabels([x[:11]+'\n'+x[11:] if x == 'Para-Nitro-Blebbistatin' else x for x in treatments])
    # ax.set_xticks([])
    # Turn off all spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    #remove legends
    ax.legend_ = None


    
    ymin,ymax = ax.get_ylim()
    
    #get only the significantly different comparisons
    starframe = sigframe[sigframe.metric == sig].reset_index(drop=True)
    # #if the first and third groups are sig AND there's another sig difference
    # #say there's more than one level
    # combos = list(starframe['group1'] +starframe['group2'])
    # slevels = [True if treatments[0] in c and treatments[-1] in c and len(starframe) else False for c in combos]
    # for s, row in starframe.iterrows():
    #     slv = 1 if slevels[s] else 0
    #     xp = np.sort([treatments.index(y) for y in [row.group1, row.group2]])
    #     starinc = (ymax-ymin)*0.001
    #     barinc = (ymax-ymin)*0.08
    #     #star
    #     ax.text(xp.mean(), ymax+(barinc*slv), get_stars(row['p-adj']), fontsize = 12, ha='center')
    #     #bar
    #     ax.plot([xp[0]+0.1,xp[1]-0.1], [ymax+(barinc*slv),ymax+(barinc*slv)], color = 'black')

    ###plot stars for DMSO-CK666
    row = starframe[starframe.Treatment=='CK666']
    #print
    print(f'CK666 pval for {ylabels[i]} is {pvdf[(pvdf.metric == sig) & (pvdf.Treatment =="CK666")].pvalue.iloc[0]}')
    xp = np.array([0,2])
    slv = 1
    starinc = (ymax-ymin)*0.001
    barinc = (ymax-ymin)*0.08
    #star
    ax.text(xp.mean(), ymax+(barinc*slv), get_stars(row['pvcorr'].values), fontsize = 12, ha='center')
    #bar
    ax.plot([xp[0]+0.1,xp[1]-0.1], [ymax+(barinc*slv),ymax+(barinc*slv)], color = 'black')

    ### plot star or ns for DMSO-PNB
    row = starframe[starframe.Treatment=='Para-Nitro-Blebbistatin']
    #print
    print(f'Bleb pval for {ylabels[i]} is {pvdf[(pvdf.metric == sig) & (pvdf.Treatment =="Para-Nitro-Blebbistatin")].pvalue.iloc[0]}')
    pstar = 'n.s.' if row.empty else get_stars(row['pvcorr'].values)
    xp = np.array([0,1])
    slv = 0
    starinc = (ymax-ymin)*0.02 if pstar == 'n.s.' else (ymax-ymin)*0.001
    barinc = (ymax-ymin)*0.08
    #star
    nsfs = 10 if pstar=='n.s.' else 12
    ax.text(xp.mean(), ymax+starinc, pstar, fontsize = nsfs, ha='center')
    #bar
    ax.plot([xp[0]+0.1,xp[1]-0.1], [ymax+(barinc*slv),ymax+(barinc*slv)], color = 'black')

plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')

