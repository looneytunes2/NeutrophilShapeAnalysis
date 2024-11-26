# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:53:47 2024

@author: Aaron
"""

########### interpolate all transitions so that only individual transitions are made ###########
import os
from CustomFunctions.DetailedBalance import interpolate_2dtrajectory, get_transition_counts, bootstrap_trajectories
import itertools
import pandas as pd
import multiprocessing
import numpy as np
from itertools import groupby
from operator import itemgetter




def collect_results(result):
    """Uses apply_async's callback to setup up a separate Queue for each process.
    This will allow us to collect the results from different threads."""
    results.append(result)


def calc_transitions(df, #dataframe with all of the relevant PC bin values
                     whichpcs, #list of the two PCs to be used in the phase space
                     time_interval, #time in seconds between the frames of the video data
                     ):
    
    ##### calculate all the phase space transitions from frame to frame including interpolations
    
    if __name__ ==  '__main__':
        pool = multiprocessing.Pool(processes=60)
        results = []
        for i, cells in df.groupby('CellID'):
            cells = cells.sort_values('frame').reset_index(drop = True)
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

                    pool.apply_async(interpolate_2dtrajectory, args = (
                        time_interval,
                        cell.CellID.iloc[0],
                        cell.frame.to_list(),
                        cell[[f'PC{whichpcs[0]}bins',f'PC{whichpcs[1]}bins']].to_numpy(),
                        ),
                        callback = collect_results)
    pool.close()
    pool.join()
    
    
    transdf = pd.DataFrame(sum([r[0] for r in results],[]))
    transdf = transdf.sort_values(by = ['CellID','frame']).reset_index(drop=True)
    transpairsdf = pd.DataFrame(sum([r[1] for r in results],[]))

    return transdf, transpairsdf
        
    
def calc_transition_rates(transdf, #dataframe with transitions from calc_transitions
                          nbins, #how many bins are in the phase space
                          ):

    #calculate the total time observed during the experiment
    ttot = transdf.time_elapsed.sum()
        
    if __name__ ==  '__main__':
        pool = multiprocessing.Pool(processes=60)
        results = []
        for x in range(nbins):
            for y in range(nbins):
                fromm = transdf[(transdf['from_x'] == x+1) & (transdf['from_y'] == y+1)].reset_index(drop=True).to_dict()
                to = transdf[(transdf['to_x'] == x+1) & (transdf['to_y'] == y+1)].reset_index(drop=True).to_dict()
                pool.apply_async(get_transition_counts, args = (
                    x+1,
                    y+1,
                    fromm,
                    to,
                    ttot,
                    ),
                    callback = collect_results)
    pool.close()
    pool.join()

    trans_rate_df = pd.DataFrame(results)
    trans_rate_df = trans_rate_df.sort_values(by = ['x','y']).reset_index(drop=True)
        
    return trans_rate_df
    


def bs_multi(transpairsdf, #dataframe of transition pairs
             ttot, #total realtime in seconds for each bootstrap iteration
             bsiter, #number of bootstrap iterations
             nbins, #number of bins in the phase space
             ):

    start = 0
    stop = 300
    allresults = []
    while start<bsiter:
        if __name__ ==  '__main__':
            pool = multiprocessing.Pool(processes=60)
            results = []
            for x in range(start,stop):
                pool.apply_async(bootstrap_trajectories, args = (
                    transpairsdf,
                    ttot,
                    nbins,
                    ),
                    callback = collect_results)
            pool.close()
            pool.join()

            allresults.extend(results)

        start = stop + 1
        stop = stop + 300
        if stop>bsiter:
            stop = bsiter
        print(f'Finished {start} iterations, beginning the next {stop-start}')
    bslist = allresults.copy()
    # del results
    # del allresults
    bsframe_sep_full = pd.concat(bslist)
    iters = [[x]*(nbins**2) for x in range(int(len(bsframe_sep_full)/(nbins**2)))]#, len(migboot)
    bsframe_sep_full['bs_iteration'] = list(itertools.chain.from_iterable(iters))

    return bsframe_sep_full

def avgrates_and_errors(bsframe_sep_full, #dataframe with all of the phase space transition data
                        nbins, #number of bins in the phase space
                        ):

    bsfield = []

    for x in range(nbins):
        for y in range(nbins):
            current = bsframe_sep_full[(bsframe_sep_full['x'] == x+1) & (bsframe_sep_full['y'] == y+1)]
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
                           'evec2y':evecs[1,0]})

    bsfield_sep = pd.DataFrame(bsfield)
    
    return bsfield_sep

