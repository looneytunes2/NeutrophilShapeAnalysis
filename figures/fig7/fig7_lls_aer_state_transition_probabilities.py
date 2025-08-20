# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 12:51:04 2025

@author: Aaron
"""

import numpy as np
import pandas as pd
from CustomFunctions import utils
from itertools import combinations_with_replacement
from scipy import interpolate
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

treatments = ['Random']
time_interval = 5 #sec/frame
whichpcs = [1,7]
ntrans = 1
derivthresh = 0.0007


#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_apply/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'
    

realaers = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_raw_transition_aer_cf.csv', index_col = 0)
#merge aer and cf info
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col = 0)
TotalFrame = pd.merge(FullFrame, realaers[['aer','angular_velocity','cell']],on='cell',how='left')
TotalFrame = TotalFrame.sort_values(['CellID','time'])

#open the bootstrapped realizations
bsaers = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_{ntrans}_transition_Area_Enclosing_Rates.csv', index_col = 0)

#only use aers that are within the range of observed time of the real cells
minmaxtime = TotalFrame.groupby('CellID').time.max().min()
itertime = bsaers.groupby('iter').cumulative_time.max()
longiters = itertime[itertime>=minmaxtime]
bsaers_long = bsaers[bsaers.iter.isin(longiters.index.to_list())].copy()


#define transition types so I can fill them in if they're absent
transtype = np.array(
            [['decreasing','decreasing'],
             ['decreasing','unchanging'],
             ['decreasing','increasing'],
             ['unchanging','decreasing'],
             ['unchanging','unchanging'],
             ['unchanging','increasing'],
             ['increasing','decreasing'],
             ['increasing','unchanging'],
             ['increasing','increasing']])



########### get transitions for real cells
eachcell = []
for i, cell in TotalFrame.groupby(['CellID']):
    cell, tck, w = utils.get_aer_state(cell, time_interval)
    cellnona = cell[~cell.aer_state.isna()]
    
    ####### get the aer state transitions from time point to time point without
    ###### the NaNs
    aerstatelist = []
    for x,y in zip(cell['aer_state'],cell.shift(-1)['aer_state']):
        if not pd.isna(x) and not pd.isna(y):
            aerstatelist.append((x,y))
    #get transition counts
    transitions, counts = np.unique(np.array(aerstatelist),axis = 0,return_counts=True)
    
    frm, to = transitions.T
    #empty probdf
    probdf = pd.DataFrame({'from':transtype[:,0], 'to':transtype[:,1],
                           'probability':np.zeros(len(transtype)),
                           'counts':np.zeros(len(transtype))})
    #fill with probabilities above 0
    for f, t, c in zip(frm, to, counts):
        #add counts
        probdf.loc[(probdf['from']==f) & (probdf['to']==t),'counts'] = c
        #add probabilities
        probdf.loc[(probdf['from']==f) & (probdf['to']==t),'probability'] = c/len(aerstatelist)
        
        
    probdf['CellID'] = i
    
    
    eachcell.append(probdf)


#### combine real cell aer state transitions into a dataframe
realcellprobs = pd.concat(eachcell, ignore_index = True)
realcellprobs['probability'] = realcellprobs['probability'].astype(float)
realcellprobs['transition_type'] = realcellprobs['from']+'_'+ realcellprobs['to']


######### get transitions for bootstrapped data
bscells = []
for i, cell in bsaers_long.groupby(['iter']):

    #get rid of NA in aer which will ruin cumulative sums etc.
    cellnona = cell[~cell.aer.isna()].copy()
    #### weight the points near gaps more
    diffs = cellnona.cumulative_time.diff().values
    #get the indicies of jumps
    gaps = np.where(diffs>time_interval)[0]
    #add the indices before jumps
    gaps = np.concatenate((gaps,gaps-1))
    w = np.ones(diffs.shape)
    w[gaps] = 3


    #interpolate for smoothening
    tck, u = interpolate.splprep(np.array((cellnona.cumulative_time.values,
                                           cellnona.aer.cumsum().values)),
                                 k=3, s = 1, w = w)#k=1, s=2, w = w)
    x, y = interpolate.splev(u, tck, der=0)
    #get the derivative of the smoothened curve
    deriv = np.gradient(y, x)
    #threshold with np.select
    threshs = [deriv>=derivthresh, deriv<=-derivthresh]
    choices = ['increasing', 'decreasing']
    statethresh = np.select(threshs, choices, default = 'unchanging')
    #add new values to dataframe
    cell.loc[cellnona.index,'aer_deriv'] = deriv
    cell.loc[cellnona.index,'aer_state'] = statethresh


    ####### get the aer state transitions from time point to time point
    aerstatetrans = [(x,y) for x,y in zip(cell['aer_state'],cell.shift(-1)['aer_state'])][:-1]
    transitions, counts = np.unique(np.array(aerstatetrans),axis = 0,return_counts=True)
    frm, to = transitions.T
    #empty probdf
    probdf = pd.DataFrame({'from':transtype[:,0], 'to':transtype[:,1], 'probability':np.zeros(len(transtype)), 'rate':np.zeros(len(transtype))})
    #fill with probabilities above 0
    for f, t, c in zip(frm, to, counts):
        probdf.loc[(probdf['from']==f) & (probdf['to']==t),'probability'] = c/len(aerstatetrans)
        probdf.loc[(probdf['from']==f) & (probdf['to']==t),'rate'] = c/(len(aerstatetrans)*time_interval)
        
    probdf['iter'] = i
    
    bscells.append(probdf)



#### combine bs cell aer state transitions into a dataframe
bscellprobs = pd.concat(bscells, ignore_index = True)
bscellprobs['probability'] = bscellprobs['probability'].astype(float)
bscellprobs['transition_type'] = bscellprobs['from']+'_'+ bscellprobs['to']




#define the colors to make the meshes
set1 = plt.cm.Set1
set2 = plt.cm.Set2
set3 = plt.cm.Set3
cmap = matplotlib.colors.ListedColormap(list(set3.colors)+[set2.colors[-2]] + [set1.colors[-1]])
sns.set_palette(cmap.colors)

#define plot order
po = ['decreasing_decreasing','decreasing_unchanging','decreasing_increasing',
      'unchanging_decreasing','unchanging_unchanging','unchanging_increasing',
      'increasing_decreasing','increasing_unchanging','increasing_increasing']
xtl = ['-\n-','-\n•','-\n+',
       '•\n-','•\n•','•\n+',
       '+\n-','+\n•','+\n+']

fig, ax = plt.subplots()
sns.stripplot(x='transition_type',y = 'probability', data = realcellprobs, hue = 'CellID', marker = 'o',
              order = po, linewidth=0.7, edgecolor = '0.5', ax = ax, zorder = 2,)
sns.violinplot(x='transition_type', y = 'probability', data = bscellprobs, linewidth = 0, palette = ['0.85'], inner=None,
               order = po, scale='count', ax = ax, zorder = 1)
ax.legend_ = None

ax.set_ylabel('Probability', fontsize = 16)
ax.set_xlabel('')
ax.set_xticklabels(xtl, fontsize = 12, ha='center')

ax.text(0.04,-0.056, 'From', transform=ax.transAxes, ha='right')
ax.text(0.04,-0.106, 'To', transform=ax.transAxes, ha='right')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()




plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')




#### save the average rates of bootstrapped state transitions
bscellprobs.groupby('transition_type').rate.mean().to_csv(__file__.split('.')[0] + '_bootstrapped_transition_rates.csv')
#### save the average real state transition rates
avgrealrates = realcellprobs.groupby('transition_type').counts.sum()/(realcellprobs.counts.sum()*time_interval)
avgrealrates.to_csv(__file__.split('.')[0] + '_raw_transition_rates.csv')