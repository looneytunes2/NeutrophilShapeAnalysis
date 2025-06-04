# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:54:13 2024

@author: Aaron
"""

import numpy as np
from aicssegmentation.core.MO_threshold import MO
from CustomFunctions import shparam_mod, shtools_mod
from vtk.util import numpy_support
from scipy.spatial import KDTree

def quickcaaxseg(im):
    return MO(im,global_thresh_method = 'tri', object_minArea = 50000)




def front_recon_distance(row,
                         l_order = 10,):
    """
        Gets the distance between original and reconstructed mesh only in the
        positive x direction
    """
    
    cell_mesh = shparam_mod.read_vtk_polydata('E:/Aaron/Combined_37C_Confocal_PCA_s5/Meshes/'+row.cell+'_cell_mesh.vtp')
    shcoeffs = row[[x for x in row.index.to_list() if 'coeff' in x]]
    #get reconstruction errors both ways
    cell_recon, grid_recon = shtools_mod.get_reconstruction_from_coeffs(np.array(shcoeffs.to_list()).reshape(2,l_order+1,l_order+1))
    #get "front" coordinates of both meshes
    cellfront = numpy_support.vtk_to_numpy(cell_mesh.GetPoints().GetData())
    reconfront = numpy_support.vtk_to_numpy(cell_recon.GetPoints().GetData())
    cellfront = cellfront[np.where(cellfront[:,0]>0)]
    reconfront = reconfront[np.where(reconfront[:,0]>0)]
    #get average nearest distance from original mesh to reconstruction
    tree = KDTree(cellfront)
    d, idx = tree.query(reconfront)
    OriginaltoReconError = np.mean(d)
    #get average nearest distance from reconstruction to original mesh
    tree = KDTree(reconfront)
    d, idx = tree.query(cellfront)
    RecontoOriginalError = np.mean(d)
    
    row['FrontOriginaltoReconError'] = OriginaltoReconError
    row['FrontRecontoOriginalError'] = RecontoOriginalError
    return row