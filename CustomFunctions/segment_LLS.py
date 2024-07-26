# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:58:51 2024

@author: Aaron
"""

import os
import numpy as np
from aicsimageio.czi_reader import CziReader
from aicsimageio.writers import OmeTiffWriter



def segment_LLS():
    

def segandinfo_LLS(celldir,
                   frame,
                   time,):
    cellname = os.path.basename(celldir).split('.')[0]
    cellimage = CziReader(celldir)
    celldata = cellimage.data[frame,:,:,:,:]
    
    return



