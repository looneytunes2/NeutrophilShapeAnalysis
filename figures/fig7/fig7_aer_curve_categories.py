# -*- coding: utf-8 -*-
"""
Created on Mon May 12 11:29:15 2025

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
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp


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




derivthresh = 0.0007
treatments = ['Random']
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
    
    cell, tck , w = utils.get_aer_state(cell, time_interval, derivthresh)
    #append that cell
    allcells.append(cell)
    
derivframe = pd.concat(allcells).reset_index(drop=True)



# vlist = []
# for c, cell in derivframe.groupby(['CellID','Movie']):
#     vframe = utils.project_raw_smooth(
#             cell, #dataframe of a cell with raw and smoothened x,y,z positions
#             time_interval, #time between frames
#             1, #integer number of image intervals to calculate velocity 
#             )
#     vlist.append(vframe)
# vframe = pd.concat(vlist)

# maxlag = 6
# plist = []
# #add movie id
# derivframe['Movie'] = [x.split('-Subset')[0] for x in derivframe.cell.to_list()]
# for c, cell in derivframe.groupby(['CellID','Movie']):
#     cell, runs = utils.get_consecutive_timepoints(cell, 'time', time_interval)
#     fmin = cell.frame.min()
#     fmax = cell.frame.max()
#     # galvdict[c[0]][c[1]] = {}
#     plist.append({'aer_state':c[0],'CellID':c[1],'lag_min':0,'dot_prod':1})
#     for lag in range(2,int(maxlag + 2)):

#         # plist = []
#         for f in np.arange(fmin,fmax-lag):
#             lagpair = cell[cell.frame.isin([f,int(f+lag-1)])]
#             traj = lagpair[['Trajectory_X','Trajectory_Z','Trajectory_Z']].values
#             if len(traj)==2:
#                 # Normalize vectors to get unit direction vectors
#                 unitvecs = traj/np.linalg.norm(traj, axis = 1)[:, np.newaxis]
#                 # Calculate dot products of consecutive unit vectors with the given lag
#                 dot_products = np.sum(unitvecs[:-1] * unitvecs[1:], axis=1)
#                 plist.append({'aer_state':c[0],'CellID':c[1],'cell':lagpair.cell.iloc[-1],
#                               'lag_min':(lag-1)*time_interval/60,'dot_prod':dot_products[0]})
# df = pd.DataFrame(plist)

# derivframe = derivframe.merge(df[df.lag_min==2*time_interval/60][['cell','dot_prod']], on = 'cell', how = 'left')

########### only include columns of interest
includelist = ['speed', 'directional_autocorrelation','aer_state']



results = []
for col in includelist:
    if col != 'aer_state':
        grouped = derivframe[['aer_state',col]].dropna().groupby('aer_state')
        stat, pval  = stats.kruskal(*np.array(grouped[col].apply(list).to_list()))
        results.append(pd.DataFrame({'stat':col,'pval':pval}, index=[0]))
pdf = pd.concat(results)
reject, pvcorr = multipletests(pdf['pval'],method='fdr_bh')[:2]
sigframe = pdf.iloc[reject]

#tukeys tests for the significant stats
alldunn = []
for s in sigframe.stat.to_list():
    tempframe = derivframe[['aer_state',s]].dropna().groupby('aer_state')
    stateorder = [i for i, t in derivframe.groupby('aer_state')]
    # print(s, pairwise_tukeyhsd(tempframe[s].values,tempframe.aer_state.values))
    dunnframe = sp.posthoc_dunn(tempframe[s].apply(list).to_list(), p_adjust = 'fdr_bh')
    dunnframe = dunnframe.rename(columns = {1:stateorder[0],
                                            2:stateorder[1],
                                            3:stateorder[2]})
    dunnframe.index = stateorder
    dunnframe['stat'] = s
    alldunn.append(dunnframe)
    print(sp.posthoc_dunn(tempframe[s].apply(list).to_list(), p_adjust = 'fdr_bh'))
    if sp.posthoc_dunn(tempframe[s].apply(list).to_list(), p_adjust = 'fdr_bh').loc[2,3]<0.05:
        print(s)
alldunnframe = pd.concat(alldunn)
    
    

ylabels = ['Instantaneous Speed (µm/sec)','Persistence']
xlabels = ['Decreasing','Unchanging','Increasing']

# CoRo = math.ceil(math.sqrt(len(includelist)))
# fig, axes = plt.subplots(CoRo, CoRo, figsize=(3.5*CoRo,3*CoRo))#, sharex=True)

fig, axes = plt.subplots(1,len(ylabels),figsize = (3.5*len(ylabels),3))
linewid = 2

#set color palette
# colorlist = matplotlib.cm.Pastel2.colors[-3:]
sns.set_palette(palette=['#d14c45','#a8a8a8','#3e88ad'])

for i, ax in enumerate(axes.flatten()):
    # sns.swarmplot(data = avgcfdf, x='Treatment', y ='average_cf', color = 'grey', size = 3.5, alpha = 0.7, ax = ax)
    sns.violinplot(x = 'aer_state', y=includelist[i], data = derivframe, order = ['decreasing','unchanging','increasing'],
                   linewidth = 0, inner = None, ax=ax, )
    for u in range(len(xlabels)):
        ax.collections[u].set_edgecolor('black')
        ax.collections[u].set_edgecolor('black')
    sns.boxplot(x = 'aer_state', y=includelist[i], data = derivframe, order = ['decreasing','unchanging','increasing'],
                width = 0.15, color = 'white',showcaps=False, showfliers=False,
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
    
        
    ymin,ymax = ax.get_ylim()
    
    
    ###plot stars for DMSO-CK666
    star = alldunnframe[alldunnframe.stat == includelist[i]].loc['decreasing','increasing']
    pstar = get_stars(star)
    xp = np.array([0,2])
    slv = 1
    starinc = (ymax-ymin)*0.02 if pstar == 'n.s.' else (ymax-ymin)*0.001
    barinc = (ymax-ymin)*0.08
    #star
    nsfs = 10 if pstar=='n.s.' else 12
    ax.text(xp.mean(), ymax+(barinc*slv), pstar, fontsize = nsfs, ha='center')
    #bar
    ax.plot([xp[0]+0.1,xp[1]-0.1], [ymax+(barinc*slv),ymax+(barinc*slv)], color = 'black')

    ### plot star or ns for DMSO-PNB
    star = alldunnframe[alldunnframe.stat == includelist[i]].loc['decreasing','unchanging']
    pstar = get_stars(star)
    xp = np.array([0,1])
    slv = 0
    starinc = (ymax-ymin)*0.02 if pstar == 'n.s.' else (ymax-ymin)*0.001
    barinc = (ymax-ymin)*0.08
    #star
    nsfs = 10 if pstar=='n.s.' else 12
    ax.text(xp.mean(), ymax+starinc, pstar, fontsize = nsfs, ha='center')
    #bar
    ax.plot([xp[0]+0.1,xp[1]-0.1], [ymax+(barinc*slv),ymax+(barinc*slv)], color = 'black')
    
    ### plot star or ns for DMSO-PNB
    star = alldunnframe[alldunnframe.stat == includelist[i]].loc['unchanging','increasing']
    pstar = get_stars(star)
    xp = np.array([1,2])
    slv = 0
    starinc = (ymax-ymin)*0.02 if pstar == 'n.s.' else (ymax-ymin)*0.001
    barinc = (ymax-ymin)*0.08
    #star
    nsfs = 10 if pstar=='n.s.' else 12
    ax.text(xp.mean(), ymax+starinc, pstar, fontsize = nsfs, ha='center')
    #bar
    ax.plot([xp[0]+0.1,xp[1]-0.1], [ymax+(barinc*slv),ymax+(barinc*slv)], color = 'black')
    
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






# # get all the stats not plotted yet
# exincludelist = [x for x in sigframe.stat.to_list() if x not in includelist]

# ylabels = exincludelist

# # ['Cell Volume (µm$^3$)','Front-Rear Volume Ratio','Cell Surface Area (µm$^2$)','Cell Sphericity',
# #            'Cell Aspect Ratio','Length Along\nTrajectory (µm)','Forward Length Along\nTrajectory (µm)',
# #            'Rearward Length Along\nTrajectory (µm)','Width Along\nTrajectory (µm)','Turn Angle (°)',]
#           #'PC1','PC2','PC4','PC5','PC6','PC7','PC8']



# sns.set_palette(palette=['#d14c45','#a8a8a8','#3e88ad'])

# CoRo = math.ceil(math.sqrt(len(exincludelist)))
# fig, axes = plt.subplots(CoRo, CoRo, figsize=(3.5*CoRo,3*CoRo))#, sharex=True)


# for i, ax in enumerate(axes.flatten()):
#     if i<len(exincludelist):
#         # sns.swarmplot(data = avgcfdf, x='Treatment', y ='average_cf', color = 'grey', size = 3.5, alpha = 0.7, ax = ax)
#         sns.violinplot(x = 'aer_state', y=exincludelist[i], data = derivframe, order = ['decreasing','unchanging','increasing'],
#                        linewidth = linewid, inner = None, ax=ax, )
#         for u in range(len(xlabels)):
#             ax.collections[u].set_edgecolor('black')
#             ax.collections[u].set_edgecolor('black')
#         sns.boxplot(x = 'aer_state', y=exincludelist[i], data = derivframe, order = ['decreasing','unchanging','increasing'],
#                     width = 0.15, color = 'white', showcaps=False, showfliers=False,
#                     boxprops={
#                         'fill': 'white',
#                         'linewidth': linewid,
#                         'edgecolor': 'black',
#                         'zorder': 2
#                         },
#                     medianprops={
#                         'linewidth': linewid,
#                         'color': 'black'
#                         },
#                     whiskerprops={
#                         'linewidth': linewid,
#                         'color': 'black'
#                         },
#                     capprops={
#                         'linewidth': linewid,
#                         'color': 'black'
#                         },
#                     ax=ax)
        
        
#         ymin,ymax = ax.get_ylim()
        
        
#         ###plot stars for DMSO-CK666
#         star = alldunnframe[alldunnframe.stat == exincludelist[i]].loc['decreasing','increasing']
#         pstar = get_stars(star)
#         xp = np.array([0,2])
#         slv = 1
#         starinc = (ymax-ymin)*0.02 if pstar == 'n.s.' else (ymax-ymin)*0.001
#         barinc = (ymax-ymin)*0.08
#         #star
#         nsfs = 10 if pstar=='n.s.' else 12
#         ax.text(xp.mean(), ymax+(barinc*slv), pstar, fontsize = nsfs, ha='center')
#         #bar
#         ax.plot([xp[0]+0.1,xp[1]-0.1], [ymax+(barinc*slv),ymax+(barinc*slv)], color = 'black')

#         ### plot star or ns for DMSO-PNB
#         star = alldunnframe[alldunnframe.stat == exincludelist[i]].loc['decreasing','unchanging']
#         pstar = get_stars(star)
#         xp = np.array([0,1])
#         slv = 0
#         starinc = (ymax-ymin)*0.02 if pstar == 'n.s.' else (ymax-ymin)*0.001
#         barinc = (ymax-ymin)*0.08
#         #star
#         nsfs = 10 if pstar=='n.s.' else 12
#         ax.text(xp.mean(), ymax+starinc, pstar, fontsize = nsfs, ha='center')
#         #bar
#         ax.plot([xp[0]+0.1,xp[1]-0.1], [ymax+(barinc*slv),ymax+(barinc*slv)], color = 'black')
        
#         ### plot star or ns for DMSO-PNB
#         star = alldunnframe[alldunnframe.stat == exincludelist[i]].loc['unchanging','increasing']
#         pstar = get_stars(star)
#         xp = np.array([1,2])
#         slv = 0
#         starinc = (ymax-ymin)*0.02 if pstar == 'n.s.' else (ymax-ymin)*0.001
#         barinc = (ymax-ymin)*0.08
#         #star
#         nsfs = 10 if pstar=='n.s.' else 12
#         ax.text(xp.mean(), ymax+starinc, pstar, fontsize = nsfs, ha='center')
#         #bar
#         ax.plot([xp[0]+0.1,xp[1]-0.1], [ymax+(barinc*slv),ymax+(barinc*slv)], color = 'black')
        
        
        
#         # ax.text(0.925,0.33,'***', fontsize=12)
#         # ax.set_ylim(0,60)
#         ax.set_xticklabels(xlabels, fontsize=10)
    
#         #remove legends
#         ax.legend_ = None
#         ax.set_ylabel(ylabels[i], fontsize=12)
#         ax.set_xlabel('')
    
#         #remove parts of box
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)

#     else:
#         ax.remove()

# plt.tight_layout()


# plt.savefig(__file__.split('.')[0] + '_extras.png', dpi = 500, bbox_inches='tight')




