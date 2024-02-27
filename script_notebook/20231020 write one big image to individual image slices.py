# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:52:26 2023

@author: Aaron
"""

from aicsimageio.readers import OmeTiffReader
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
import os

folder = '//10.158.28.37/ExpansionHomesA/avlnas/HL60 Galv/20231019/20231019_488EGFP-CAAX_2/'
full_image = OmeTiffReader(folder+ '20231019_488EGFP-CAAX_2_MMStack_Pos0.ome.tif').data


savedir = folder + 'Default/'
if not os.path.exists(savedir):
    os.mkdir(savedir)
for t in range(full_image.shape[0]):
    for z in range(full_image.shape[-3]):
        OmeTiffWriter.save(full_image[t,0,z,:,:],savedir+f'img_channel000_position000_time{t:09}_z{z:03}.tif')