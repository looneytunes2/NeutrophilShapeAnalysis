# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:15:45 2023

@author: Aaron
"""


import os 
import numpy as np
import pandas as pd
from CustomFunctions import linear_cycle_utils
from pathlib import Path


#get directories and open separated datasets


treatments = ['Random']
time_interval = 10 #sec/frame
whichpcs = [1,2]
origin = [7, 8]
binrange = 360/6
direction = 'clockwise'
zerostart = 'left'


#get directories and open separated datasets
basedir = Path('E:/Aaron/Combined_37C_Confocal_PCA_planar')
datadir = basedir.joinpath('Data_and_Figs')


FullFrame = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col=0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())
#open the centers of the binned PCs
centers = pd.read_csv(datadir.joinpath('PC_bin_centers.csv'), index_col=0)
#restrict dataframe to only random experiments
TotalFrame = FullFrame[FullFrame.Treatment=='Random']

angframe = linear_cycle_utils.linearize_cycle_continuous(
            TotalFrame, 
            centers,
            origin, 
            whichpcs,
            zerostart,
            direction,)



angframe =  linear_cycle_utils.bin_angular_coord(
        angframe,
        whichpcs,
        binrange,
        )



for t, treat in angframe.groupby('Treatment'):
    linear_cycle_utils.animate_linear_cycle_shcoeffs(
                            treat,
                            os.getcwd(),
                            t,
                            whichpcs,
                            binrange,
                            lmax = 10,
                            smooth = False
                            )

