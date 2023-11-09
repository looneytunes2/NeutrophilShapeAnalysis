# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 13:28:07 2022

@author: Aaron
"""
import numpy as np
from CustomFunctions import PILRagg 




def correlate_representations(rep1, rep2):
    pcor = np.corrcoef(rep1.flatten(), rep2.flatten())
    # Returns Nan if rep1 or rep2 is empty.
    return pcor[0, 1]




def correlate(dirr, indexes, strs):
    rep1 = PILRagg.read_parameterized_intensity(dirr + indexes[0] + '_PILR.tiff')
    rep2 = PILRagg.read_parameterized_intensity(dirr + indexes[1] + '_PILR.tiff')
    rep1 = rep1[0,:,:]
    rep2 = rep2[0,:,:]
    return [indexes[0], indexes[1], correlate_representations(rep1, rep2), strs[0], strs[1]]





