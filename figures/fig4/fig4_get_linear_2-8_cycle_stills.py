# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:15:45 2023

@author: Aaron
"""


import os 
import numpy as np
import pandas as pd
from neutrophil_shape.CustomFunctions import linear_cycle_utils
from neutrophil_shape.config.loader import load_config
from pathlib import Path


#get directories and open separated datasets


treatments = ['Random']
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'
whichpcs = (2,8)
pc_combos = config.common.pc_combos
origin = config.db_params.origins[pc_combos.index(whichpcs)]
binrange = 360/6
direction = 'clockwise'
zerostart = 'left'

#get directories and open separated datasets
basedir = config.common.savedir
datadir = basedir.joinpath('shape_data')

    
FullFrame = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col=0)
#open the centers of the binned PCs
centers = pd.read_csv(datadir.joinpath('PC_bin_centers.csv'), index_col=0)
TotalFrame = FullFrame[FullFrame.Treatment.isin(treatments)].copy()


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
                            Path(__file__).parent,
                            t,
                            whichpcs,
                            binrange,
                            lmax = 10,
                            smooth = False
                            )

