# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:17:38 2024

@author: Aaron
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import groupby

datadir = 'E:/Aaron/Combined_Confocal_PCA_nospeedoutliers/'
savedir = datadir + 'Galvanotaxis/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

################ open the bootstrap files
bsframe_sep_full = pd.read_csv(savedir+'PC1-PC8_bootstrapped_transitions.csv', index_col=0)
modes = bsframe_sep_full.Migration_Mode.unique()

#ensure that DMSO is the first in order
bsframe_sep_full['Migration_Mode'] = pd.Categorical(bsframe_sep_full.Migration_Mode, categories=['Random','Pre-Galvanotaxis','Galvanotaxis'], ordered=True)
bsframe_sep_full = bsframe_sep_full.sort_values(by='Migration_Mode')
print('Opened bootstrap file')



nbins = 11
############### 
    
#### get current field for different numbers of bootstrap realizations ######
####### this is for looking at data spread for the current field ############
nob = [200,500,1000,2000,3000,4000,5000]
boots = []
for n in nob:
    tempframe = bsframe_sep_full[bsframe_sep_full.bs_iteration<n] 
    print(len(tempframe))
    bsfield = []
    for m, mig in tempframe.groupby('Migration_Mode'):
        for x in range(nbins):
            for y in range(nbins):
                current = mig[(mig['x'] == x+1) & (mig['y'] == y+1)]
                js = np.array([[[(row.x_plus_rate - row.x_minus_rate)/2, 0],[0,(row.y_plus_rate - row.y_minus_rate)/2]] for i, row in current.iterrows()])
                avgjs = np.mean(js, axis = 0)
                evals, evecs = np.linalg.eigh(avgjs)
                bsfield.append({'x':x+1,
                                'y':y+1,
                                'eval1':evals[1],
                                'eval2':evals[0],
                               'evec1x':evecs[0,1],
                               'evec1y':evecs[1,1],
                               'evec2x':evecs[0,0],
                               'evec2y':evecs[1,0],
                              'Migration_Mode':m})
            
    bsfield_sep = pd.DataFrame(bsfield)
    boots.append([n,bsfield_sep])
    
    
bootdiffs = []
for i in range(len(boots)-1):
    tempdiffs = pd.DataFrame()
    more = boots[i+1][1].copy()
    less = boots[i][1].copy()
    tempdiffs['eval1diffs'] = more.eval1-less.eval1
    tempdiffs['eval2diffs'] = more.eval2-less.eval2
    tempdiffs['Migration_Mode'] = more.Migration_Mode
    bootdiffs.append(tempdiffs)
    
allbootdiffs = pd.concat(bootdiffs)
allbootdiffs['bootnumber'] = np.array([[nob[x]]*363 for x in range(len(bootdiffs))]).flatten().astype(str)
    

bootnumvars = []
for i, bn in allbootdiffs.groupby(['bootnumber','Migration_Mode']):
    print(i, bn.eval1diffs.var(), bn.eval2diffs.var())
    bootnumvars.append({'bootnumber':i[0],
                        'Migration_Mode':i[1],
                        'eval1diffvar':bn.eval1diffs.std(),
                        'eval2diffvar':bn.eval2diffs.std()})
allbootnumvars = pd.DataFrame(bootnumvars)
allbootnumvars['bootnumber'] = pd.Categorical(allbootnumvars.bootnumber, categories=[str(x) for x in nob], ordered=True)

fig, ax = plt.subplots()
sns.stripplot(data = allbootnumvars, x = 'bootnumber',y='eval1diffvar', hue = 'Migration_Mode', ax = ax)
ax.set_ylim(0,0.00000000002)






bootmeans = []
for b in boots:
    tempmeans = pd.DataFrame()
    tempmeans['eval1means'] = b.eval1.mean()
    tempdiffs['eval2means'] = b.eval2.mean()
    tempdiffs['Migration_Mode'] = b.Migration_Mode
    bootmeans.append(tempdiffs)
    
allbootdiffs = pd.concat(bootdiffs)
allbootdiffs['bootnumber'] = np.array([[nob[x]]*363 for x in range(len(bootdiffs))]).flatten().astype(str)
