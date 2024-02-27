#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 09:30:52 2023

@author: vidya
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 22:41:51 2022

@author: Abhishek, 
Adapted closely from Dalmasso et al., 4D reconstruction of murine developmental
trajectories using spherical harmonics, Developmental Cell, 2022
"""

import time
import numpy as np
import vtk
import os
from scipy.spatial import KDTree
from vtk.util import numpy_support
from vedo import ProgressBar, Mesh, Points, write, utils
from vedo.volumeFromMesh import volumeFromMesh
from vedo.interpolateVolume import interpolateVolume
from Fearless.utils import voxelIntensity, forwardTransformation, samplePoints, inverseTransformations


st = time.time()

"""Convert a vtk mesh from the limb-data files into voxel data."""

# Path to folder which has the input meshes.
DataPath = '/home/vidya/Desktop/PhD/Dr.Theriot_Rotation/PC_time_analysis/Parametrisation/parametrization_data/all_images/SDspharm/'

cells = []
filelist_fl = list()
distances = [] # Used later to store the reconstruction distances.

radiusDiscretisation = 100 # Number of spheres
N = 25 # Number of samples of theta and phi to reconstruct spheres (total points = N*N).
lmax = 10 # Each sphere is reconstructed to the same value of lmax.
expo = 1.0



#filelist_fl is a list containing the names of each input mesh as a string.
for file in os.listdir(DataPath):
    if file.endswith(".vtp"):
        list.append(filelist_fl, file)
        
# "cells" containes the corresponding vedo mesh for each input mesh.
for i in range(0,len(filelist_fl)):
    x = filelist_fl[i]
    mesh = vtk.vtkXMLPolyDataReader()
    mesh.SetFileName(DataPath + x)
    mesh.Update()
    l = mesh.GetOutput()
    cells.append(Mesh(l))
    

totcells = len(filelist_fl)

for j in range(0,totcells):
    cells[j].filename = str(filelist_fl[j])
   

sampleSize = (40, 40, 40) # Number of voxels in the volume element.


# Collecting imgBounds of all cells.
allBounds = np.empty(shape=(6, totcells))

for j in range(0,totcells):
    allBounds[:, j] = cells[j].GetBounds()
  
# Finding max imgBounds.
allBoundsTmp = []
pb = ProgressBar(0, allBounds.shape[0], c=1)
for j in pb.range():
    if allBounds[j, 0] > 0:
        allBoundsTmp.append(np.max(allBounds[j, :]))
    else:
        allBoundsTmp.append(np.min(allBounds[j, :]))
    
# Making imgBounds a bit bigger, just in case.
#imgBOunds = [x * 1.2 for x in allBoundsTmp]

imgBOunds = [-29.310628509521482, 31.58730239868164, -29.14987564086914, 28.173216247558592, -31.161527252197263, 31.048661041259763]
# Matrix with all Clm of all cells for all spheres.
allClmMatrix = np.zeros((totcells, radiusDiscretisation,
                        2, lmax, lmax), dtype=np.float32, order='C')


pb = ProgressBar(0, totcells, c=1)
for j in pb.range(): #The whole code loops over the total number of cells.

    
    trimesh_test = utils.vedo2trimesh(cells[j])
    trimesh_test.fill_holes()
    cells[j] = utils.trimesh2vedo(trimesh_test)
  #  vol =   cells[j].signed_distance(dims=sampleSize)
        
    
    vol = volumeFromMesh(cells[j],
                         dims=sampleSize,
                         bounds=imgBOunds,
                         invert=False,  # invert sign
                         )
    # "vol" is the volume element whose voxels are the SD values from the mesh.
    # The number of voxels is (20,20,20). The bounds are the largest bounds from all the input images,
    # and hence is input image dependent.
        
###################################################################
    

   
                                                            
               
   
    # Computing voxel intensities at 625 points for 10 spheres. 
    # This is essentially the SD vales on the surface of each sphere.
    allIntensities = voxelIntensity(vol, expo, N, radiusDiscretisation)
    allIntensitiesShape = allIntensities.shape
  
#####################################
    """Path where the reconstructions are stored. Create this file before running the code."""
    path = 'reconstructions/'

##############################################

    # Forward transformations (SPHARM). Spherical harmonic expansion.
    Clm = forwardTransformation(allIntensities, N, lmax)

    allClmMatrix[j, :, :, :, :] = Clm

       

##############################################
    # Getting certain parameters from the volume element.
 
    pos = vol.center()
    rmax = vol.diagonal_size()/2
    volBounds = np.array(vol.GetBounds())

    # The same points at which the SD values were initially evaluated. This is for the reconstruction.
    samplePoints1 = samplePoints(vol, expo, N, radiusDiscretisation)

   
    ##############################################
    # Inverse transformations. Going back from spherical harmonic coefficients to SD values 
    # at 625 points for 10 spheres. 

    
    inverse_Matrix = inverseTransformations(
        allClmMatrix[j, :, :, :, :], allIntensitiesShape, N, lmax)

    intensitiesreshape = np.reshape(
        inverse_Matrix, inverse_Matrix.shape[0] * inverse_Matrix.shape[1])

        
##############################

    # Going back from the reconstructed intensities to a SD map.
    
    voxBin = 20  # Essentially the same as sampleSie.
    
    #Vedo point cloud of the points at which SD values were calculated in the previous step.
    apts = Points(samplePoints1)
    #Assigning the SD value of the appropriate point.
    apts.pointdata["scals"] = intensitiesreshape
    # Reconstructing the volume element.
    volume = interpolateVolume(points = apts, kernel='shepard', radius=(
        rmax/20), dims=(voxBin, voxBin, voxBin), bounds=volBounds)
        
    # Writing the output mesh.
    write(volume.isosurface(threshold=-0.01),
        path + 'Limb-rec_' + str(j+1) + '.vtk', binary=False)
          
        
    ##############################################



    #For calculating the reconstruction distances.
    
    # Path to folder which has the input meshes.
    DataPath = '/home/vidya/Desktop/PhD/Dr.Theriot_Rotation/PC_time_analysis/Parametrisation/parametrization_data/all_images/SDspharm/'

    cell_mesh = vtk.vtkXMLPolyDataReader()
    recon_mesh = vtk.vtkPolyDataReader()

    cell_mesh.SetFileName(DataPath + filelist_fl[j])
    recon_mesh.SetFileName(path + 'Limb-rec_' + str(j+1) + '.vtk')

    cell_mesh.Update()
    cell_me = cell_mesh.GetOutput()
    recon_mesh.Update()
    recon_me = recon_mesh.GetOutput()

    tree = KDTree(numpy_support.vtk_to_numpy(cell_me.GetPoints().GetData()))
    d, idx = tree.query(numpy_support.vtk_to_numpy(recon_me.GetPoints().GetData()))
    

    
    distances.append(np.mean(d))

print(distances)


end = time.time()
ti = (end-st)
print(ti)
