# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:31:54 2024

@author: Aaron
"""

import pandas as pd
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

time_interval = 10 #sec/frame

datadir = 'F:/Combined_Confocal_PCA_newrotation_newalign/'
savedir = datadir + 'alldata/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

TotalFrame = pd.read_csv(datadir + 'Shape_Metrics_transitionPCbins.csv', index_col=0)

centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)

nbins = np.max(TotalFrame[[x for x in TotalFrame.columns.to_list() if 'bin' in x]].to_numpy())



color_scale = pd.DataFrame({'color':list(sns.diverging_palette(20, 220, n=200).as_hex()),
              'value':list(np.arange(-1,1,2/200))})


#########Scatter plots for cell metrics and the PCs


#drop all of the non-numeric columns
coldrop = [x for x in TotalFrame.columns.to_list() if TotalFrame[x].dtype == 'O']
metric_frame = TotalFrame.drop(columns=coldrop)

#get pearson correlation matrix for TotalFrame without bins
totalcorr = metric_frame.corr()


cols = metric_frame.columns.to_list()
sp = math.ceil(math.sqrt(len(cols)))
fig, axes = plt.subplots(sp,sp,figsize=(sp*3,sp*3))

for i, ax in enumerate(axes.flatten()):
    if i<len(cols):

        rows = metric_frame.iloc[:,i][metric_frame.iloc[:,i].isna() == False].index.to_list()
        x = metric_frame.loc[rows,cols[i]]
        y = metric_frame.loc[rows,'PC2']
        coef = np.polyfit(x,y,1)
        poly1d_fn = np.poly1d(coef) 
        p_corr = totalcorr.loc['PC2', metric_frame.iloc[:,i].name]
        color = color_scale.color.loc[color_scale.value == closest(list(color_scale.value), p_corr)].values[0]
        ax.scatter(x,y, color = color)
        ax.plot(x, poly1d_fn(x), 'k')
        ax.set_xlabel(cols[i])
        ax.set_ylabel('PC2')
    else:
        ax.remove()

plt.tight_layout()
plt.savefig('C:/Users/Aaron/Desktop/PC2bigone.png', bbox_inches='tight')



difflist = []
for i, cell in TotalFrame.sort_values(by=['CellID','frame']).reset_index(drop=True).groupby('CellID'):
    difflist.append(cell.drop(columns=coldrop).diff())
    
    
    

#drop all of the non-numeric columns
diff_frame = pd.concat(difflist)

#get pearson correlation matrix for TotalFrame without bins
totalcorr = diff_frame.corr()


cols = metric_frame.columns.to_list()
sp = math.ceil(math.sqrt(len(cols)))
fig, axes = plt.subplots(sp,sp,figsize=(sp*3,sp*3))

for i, ax in enumerate(axes.flatten()):
    if i<len(cols):

        rows = diff_frame.iloc[:,i][diff_frame.iloc[:,i].isna() == False].index.to_list()
        x = diff_frame.loc[rows,cols[i]]
        y = diff_frame.loc[rows,'PC2']
        coef = np.polyfit(x,y,1)
        poly1d_fn = np.poly1d(coef) 
        p_corr = totalcorr.loc['PC2', diff_frame.iloc[:,i].name]
        color = color_scale.color.loc[color_scale.value == closest(list(color_scale.value), p_corr)].values[0]
        ax.scatter(x,y, color = color)
        ax.plot(x, poly1d_fn(x), 'k')
        ax.set_xlabel(cols[i])
        ax.set_ylabel('PC2')
    else:
        ax.remove()

plt.tight_layout()
plt.savefig('C:/Users/Aaron/Desktop/PC2bigonediff.png', bbox_inches='tight')
