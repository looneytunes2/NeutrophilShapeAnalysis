# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:15:45 2023

@author: Aaron
"""


import os 
import re
import numpy as np
import pandas as pd
import vtk
from vtk.util import numpy_support
from CustomFunctions import linear_cycle_utils

#get directories and open separated datasets


treatments = ['Random']
time_interval = 10 #sec/frame
whichpcs = [1,7]
origin = [9, 9]
binnum = 18
binrange = 360/binnum
direction = 'clockwise'
zerostart = 'left'


#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())
#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
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

