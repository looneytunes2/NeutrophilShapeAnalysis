# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:56:17 2024

@author: Aaron
"""

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.transforms
import seaborn as sns
import math
from math import sqrt, factorial
import re
from itertools import groupby
import scipy
import random
from decimal import Decimal
from operator import itemgetter
import multiprocessing
from CustomFunctions import PCvisualization

def cllct_rslts(result):
    """Uses apply_async's callback to setup up a separate Queue for each process.
    This will allow us to collect the results from different threads."""
    results.extend(result)
def collect_results(result):
    """Uses apply_async's callback to setup up a separate Queue for each process.
    This will allow us to collect the results from different threads."""
    results.append(result)


#some functions that help find angles between planes

#https://keisan.casio.com/exec/system/1223596129
def plane_eq(points):
    p0 = points[0,:]
    p1 = points[1,:]
    p2 = points[2,:]
    v1 = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]]
    v2 = [p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]]
    abc = np.cross(v1, v2)
    d = np.array([abc[0]*p0[0], abc[1]*p0[1], abc[2]*p0[2]])
    return abc, d
# Function to find Angle
def distance(a1, b1, c1, a2, b2, c2):
     
    d = ( a1 * a2 + b1 * b2 + c1 * c2 )
    e1 = math.sqrt( a1 * a1 + b1 * b1 + c1 * c1)
    e2 = math.sqrt( a2 * a2 + b2 * b2 + c2 * c2)
    d = d / (e1 * e2)
    A = math.degrees(math.acos(d))
    return A

def angle(a1, b1, a2, b2):
     
    d = ( a1 * a2 + b1 * b2)
    e1 = math.sqrt( a1 * a1 + b1 * b1)
    e2 = math.sqrt( a2 * a2 + b2 * b2)
    d = d / (e1 * e2)
    A = math.degrees(math.acos(d))
    return A
def closest(lst, K):  
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]







#get directories and open separated datasets
from CustomFunctions.DetailedBalance import filter_dataframe
#define the time interval
time_interval = 10 #sec/frame

datadir = 'D:/Aaron/Data/Combined_Confocal_PCA_newrotation/'
savedir = datadir + 'Galvanotaxis/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

FullFrame = pd.read_csv(datadir + 'Shape_Metrics_transitionPCbins.csv', index_col=0)

nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())

#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)


#which dates of these experiments to include
dates = ['20231019',
        '20231020']
#grab only the data from the days of these experiments
TotalFrame = FullFrame.iloc[[i for i,x in enumerate(FullFrame.cell) if x.split('_')[0] in dates]]

#add migration mode
migmod = []
for f in TotalFrame['frame'].to_list():
    if f<180:
        migmod.append('Random')
    # cells in the 4 minutes after the current is turned on ar "pre" galv
    elif f>=180 and f<205:
        migmod.append('Pre-Galvanotaxis')
    else:
        migmod.append('Galvanotaxis') 

# TotalFrame = FullFrame[FullFrame.cell.isin(include)]
#add the treatment categories
TotalFrame['Migration_Mode'] = pd.Categorical(migmod, categories=['Random','Pre-Galvanotaxis','Galvanotaxis'], ordered=True)
TotalFrame = TotalFrame.sort_values(by='Migration_Mode')

#filter the dataset on speed
# TotalFrame = filter_dataframe(TotalFrame, 'speed', thresh = 0.07)


#remove the big flips
# TotalFrame = TotalFrame[abs(TotalFrame.Width_Rotation_Angle)<100]


#get all non-data
nonnum = []
for c in TotalFrame.columns.to_list():
    if TotalFrame[c].dtype == str or TotalFrame[c].dtype == object or TotalFrame[c].dtype == 'category':
        nonnum.append(c)
######### get 
higher = []
for i, cells in TotalFrame.groupby('CellID'):
    cells = cells.sort_values('frame').reset_index(drop = True)
    runs = list()
    #######https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
    for k, g in groupby(enumerate(cells['frame']), lambda ix: ix[0] - ix[1]):
        currentrun = list(map(itemgetter(1), g))
        list.append(runs, currentrun)

    for r in runs:
        r = np.array(r, dtype=int)
        cell = cells.iloc[[cells[cells.frame==y].index[0] for y in r]].drop(columns=nonnum)
        higher.append(cell.diff()[abs(cell.PC1.diff())>2])
        
        
higher = pd.concat(higher)


higher = abs(higher)

corrframe = higher.corr()



color_scale = pd.DataFrame({'color':list(sns.diverging_palette(20, 220, n=200).as_hex()),
              'value':list(np.arange(-1,1,2/200))})
fig, axes = plt.subplots(7, 8, figsize=(15,15))#, sharex=True)
for i, q in enumerate([x for x in higher.columns.to_list() if 'PC' not in x and 'Vec' not in x]):
    if higher[q].dtype != str and higher[q].dtype != object:
        ax = axes.flatten()[i]
        x = higher['PC1']
        y = higher[q]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        p_corr = corrframe.loc[q, 'PC1']
        color = color_scale.color.loc[color_scale.value == closest(list(color_scale.value), p_corr)].values[0]
        ax.scatter(x,y, color = color)
        ax.plot(x, intercept+slope*x, 'k')
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        # ax.text(0.5, 0.1, 'pcorr =\n'+str(round(p_corr,4)),transform=ax.transAxes)
        # ax.set_aspect('equal','box')
        maxlim = max(x.max(),y.max(), abs(x.min()), abs(y.min()))
        # ax.set_xlim(-maxlim,maxlim)
        # ax.set_ylim(-maxlim,maxlim)
        ax.set_title(q)
plt.tight_layout()

        