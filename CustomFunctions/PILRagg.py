# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 21:02:14 2022

@author: Aaron
"""
import numpy as np
from aicsimageio import AICSImage
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
import vtk

def read_parameterized_intensity(im):
#     code, intensity_names = None, []
#     path = f"parameterization/representations/{index}.tif"
#     path = self.control.get_staging() / path

#     if path.is_file():
    code = AICSImage(im)
    intensity_names = code.channel_names
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


def read_vtk_polydata(path: str):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()


def write_ome_tif(path, img, channel_names=None, image_name=None):
    # path = Path(path)
    dims = [['X', 'Y', 'Z', 'C', 'T'][d] for d in range(img.ndim)]
    dims = ''.join(dims[::-1])
    name = path.stem if image_name is None else image_name
    OmeTiffWriter.save(img, path, dim_order=dims, image_name=name, channel_names=channel_names)
    return