# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:41:42 2024

@author: Aaron
"""


import os
import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter
import matplotlib.pyplot as plt
########## find a cell of interest ###########
savedir = 'F:/Combined_Confocal_PCA_newrotation_newalign/'
infrsavedir = savedir + 'Inframe_Videos/'

TotalFrame = pd.read_csv(savedir+'Shape_Metrics_transitionPCbins.csv', index_col=0)

#find the length of cell consecutive frames
results = []
for i, cells in TotalFrame.groupby('CellID'):
    cells = cells.sort_values('frame').reset_index(drop = True)
    runs = list()
    #######https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
    for k, g in groupby(enumerate(cells['frame']), lambda ix: ix[0] - ix[1]):
        currentrun = list(map(itemgetter(1), g))
        list.append(runs, currentrun)
    maxrun = max([len(l) for l in runs])
    actualrun = max(runs, key=len, default=[])
    results.append([i, maxrun, actualrun])
#find
stdf = pd.DataFrame(results, columns = ['CellID','length_of_run','actual_run']).sort_values('length_of_run', ascending=False).reset_index(drop=True)
stdf.CellID.head(30)




for x in range(30):
    #select cell from list above
    row = stdf.loc[x]
    print(row.CellID)
    #get the data related to this run of this cell
    data = TotalFrame[(TotalFrame.CellID==row.CellID) & (TotalFrame.frame.isin(row.actual_run))].sort_values('frame').reset_index(drop=True)
    
    
    N = 20
    plt.plot(np.convolve(data.speed, np.ones(N)/N, mode='valid'), label=x)

plt.legend()




val = np.convolve(data.speed, np.ones(N)/N, mode='valid')
ful = np.convolve(data.speed, np.ones(N)/N, mode='full')
np.where(ful == val[-1])[0]


def running_mean(x, N):
    cumsum = numpy.cumsum(numpy.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


spthresh = 0.05

for i, cells in df_track.groupby('cell'):
    cells = cells.reset_index(drop = True)
    ############ actually filter the runs ################
    runs = list()
    #######https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
    for k, g in groupby(enumerate(cells['frame']), lambda ix: ix[0] - ix[1]):
        currentrun = list(map(itemgetter(1), g))
        list.append(runs, currentrun)
    for r in runs:
        r = np.array(r, dtype=int)
        #skip runs less than 3 frames long
        if len(r)<2:
            pass
        else:
            cell = cells.iloc[[cells[cells.frame==y].index[0] for y in r]]
            N=20
            #shrink the convolution window if the track isn't long enough
            if len(cell)<N:
                N=round(len(cell)/3)
            ####### alternatively use:
            ####### con = np.convolve(np.nan_to_num(data.speed), np.ones(N)/N, mode='valid')
            con = running_mean(np.nan_to_num(cell.speed),N)
            abvthresh = np.where(con>spthresh)[0]
            indtopull = abvthresh + (N-1)
            if abvthresh[0] == 0:
                indtopull = np.insert(indtopull, 0, range(N-1))
            cellabv = cell.iloc[indtopull].copy()

con = np.convolve(np.nan_to_num(data.speed), np.ones(N)/N, mode='valid')
ful = np.convolve(np.nan_to_num(data.speed), np.ones(N)/N, mode='full')

#find the length of cell consecutive frames
results = []
for i, cells in TotalFrame.groupby('CellID'):
    cells = cells.sort_values('frame').reset_index(drop = True)
    runs = list()
    #######https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
    for k, g in groupby(enumerate(cells['frame']), lambda ix: ix[0] - ix[1]):
        currentrun = list(map(itemgetter(1), g))
        list.append(runs, currentrun)
    maxrun = min([len(l) for l in runs])
    actualrun = min(runs, key=len, default=[])
    results.append([i, maxrun, actualrun])
#find
stdf = pd.DataFrame(results, columns = ['CellID','length_of_run','actual_run']).sort_values('length_of_run', ascending=False).reset_index(drop=True)
stdf.CellID.head(30)



