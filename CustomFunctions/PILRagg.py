# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 21:02:14 2022

@author: Aaron
"""
from pathlib import Path
import numpy as np
from aicsimageio import AICSImage
import tifffile
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
import vtk

def read_parameterized_intensity(im):
#     code, intensity_names = None, []
#     path = f"parameterization/representations/{index}.tif"
#     path = self.control.get_staging() / path

#     if path.is_file():
    code = AICSImage(im)
    code = code.data.squeeze()
    if code.ndim == 2:
        code = code.reshape(1, *code.shape)
#         if return_intensity_names:
#             return code, intensity_names
    return code

def normalize_representations(reps):
    # Expected shape is SCMN
    if reps.ndim != 4:
        raise ValueError(f"Input shape {reps.shape} does not match expected SCMN format.")
    count = np.sum(reps, axis=(-2,-1), keepdims=True)
    reps_norm = np.divide(reps, count, out=np.zeros_like(reps), where=count>0)
    return reps_norm



def write_ome_tif(path, img, channel_names=None, image_name=None):
    # path = Path(path)
    dims = [['X', 'Y', 'Z', 'C', 'T'][d] for d in range(img.ndim)]
    dims = ''.join(dims[::-1])
    name = path.stem if image_name is None else image_name
    OmeTiffWriter.save(img, path, dim_order=dims, image_name=name, channel_names=channel_names)
    return

def pilr_correlation(
        corrpair, #iterable len 2, [0] is "cell1" and [1] is the cell to correlate it with
        ):
    cell1 = tifffile.imread(corrpair[0])
    cell2 = tifffile.imread(corrpair[1])
    cor = np.corrcoef(cell1.flatten(), cell2.flatten())
    return [Path(corrpair[0]).stem.split('_PILR')[0], Path(corrpair[1]).stem.split('_PILR')[0], cor[0,1]]