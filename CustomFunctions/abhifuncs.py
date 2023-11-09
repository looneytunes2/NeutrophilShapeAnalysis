# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 15:20:00 2022

@author: Aaron
"""

import numpy as np
import vtk
import os
from scipy.spatial import KDTree
from vtk.util import numpy_support
from vedo import ProgressBar, Mesh, Points, write, utils
from vedo.volumeFromMesh import volumeFromMesh
from vedo.interpolateVolume import interpolateVolume
from Fearless.utils import voxelIntensity, forwardTransformation, samplePoints, inverseTransformations
import csv




def getBounds(x):
    mesh = vtk.vtkXMLPolyDataReader()
    mesh.SetFileName(x)
    mesh.Update()
    l = mesh.GetOutput()
    return Mesh(l).GetBounds()
    




def abhiparam(
        savedir,
        meshdir,
        cell,
        sampleSize,
        imgBOunds,
        radiusDiscretisation,
        N,
        lmax,
        expo,
        ):
    
    
    mesh = vtk.vtkXMLPolyDataReader()
    mesh.SetFileName(meshdir + cell)
    mesh.Update()
    l = mesh.GetOutput()
    cells = Mesh(l)
    
    trimesh_test = utils.vedo2trimesh(cells)
    trimesh_test.fill_holes()
    cells = utils.trimesh2vedo(trimesh_test)
    
    vol = volumeFromMesh(cells,
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
  

##############################################

    # Forward transformations (SPHARM). Spherical harmonic expansion.
    Clm = forwardTransformation(allIntensities, N, lmax)



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
        Clm, allIntensitiesShape, N, lmax)

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
        savedir + cell.split('Cell_Mesh')[0] + 'newrecon.vtk', binary=False)
          
        
    ##############################################



    #For calculating the reconstruction distances.
    


    cell_mesh = vtk.vtkXMLPolyDataReader()
    recon_mesh = vtk.vtkPolyDataReader()

    cell_mesh.SetFileName(meshdir + cell)
    recon_mesh.SetFileName(savedir + cell.split('Cell_Mesh')[0] + 'newrecon.vtk')

    cell_mesh.Update()
    cell_me = cell_mesh.GetOutput()
    recon_mesh.Update()
    recon_me = recon_mesh.GetOutput()

    #get average nearest distance for this particular reconstruction
    tree = KDTree(numpy_support.vtk_to_numpy(cell_me.GetPoints().GetData()))
    d, idx = tree.query(numpy_support.vtk_to_numpy(recon_me.GetPoints().GetData()))
    RtOdistance = np.mean(d)

    #get average nearest distance for this particular reconstruction
    tree = KDTree(numpy_support.vtk_to_numpy(recon_me.GetPoints().GetData()))
    d, idx = tree.query(numpy_support.vtk_to_numpy(cell_me.GetPoints().GetData()))
    OtRdistance = np.mean(d)

    # np.save(savedir+cell.split('Cell_')[0]+'_Clm.npy', Clm)


    
    with open(savedir+ cell+ 'dists.csv', 'w') as f:
          
        # using csv.writer method from CSV package
        wr = csv.writer(f)
          
        wr.writerow(['OtRdistance', 'RtOdistance'])
        wr.writerow([OtRdistance, RtOdistance])
    f.close()
    
    return




          
        



def abhirecon(
        Clm,
        N,
        lmax,
        samplePoints1,
        rmax,
        volBounds,
        allIntensitiesShape,
        savefile
        ):

 
    

    ##############################################
    # Inverse transformations. Going back from spherical harmonic coefficients to SD values 
    # at 625 points for 10 spheres. 

    
    inverse_Matrix = inverseTransformations(
        Clm, allIntensitiesShape, N, lmax)

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
        savefile, binary=False)
          
        


