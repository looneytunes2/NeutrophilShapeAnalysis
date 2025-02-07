# -*- coding: utf-8 -*-
"""
Created on Mon May 31 18:03:57 2021

@author: Aaron
"""

import re
import os
import vtk
import warnings
import pyshtools
import numpy as np
import pandas as pd
from vtk.util import numpy_support
from vtkmodules.vtkFiltersCore import (
    vtkCleanPolyData,
    vtkTriangleFilter
)
from vtkmodules.vtkFiltersGeneral import vtkBooleanOperationPolyDataFilter
from vtkmodules.vtkFiltersSources import vtkCubeSource
from skimage import transform as sktrans
from scipy import signal
from scipy import interpolate as spinterp
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R


from . import shtools_mod, cytoparam_mod

from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from aicsimageio.readers.tiff_reader import TiffReader


def read_vtk_polydata(path: str):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()

def find_normal_width_peaks(
        impath,
        csvdir,
        xyres,
        zstep,
        sigma: float = 0,
        align_method: str = 'None',
        ):

    
    #get cell name from impath
    cell_name = impath.split('/')[-1].split('_segmented')[0]
    #read image
    im = TiffReader(impath)
    

    #if align_method is a numpy array, use that as the vector to align to
    if type(align_method) == np.ndarray:
        vec = align_method.copy()
        #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        xaxis = np.array([[1,0,0],[0,1,0], [0,0,1]]).astype('float64')
        upnorm = np.cross(vec,[1,0,0])
        # if upnorm[2]<0:
        #     upnorm = upnorm.copy() * -1
        sidenorm = np.cross(vec,upnorm)
        current_vec = np.stack((vec, sidenorm, upnorm), axis = 0)
        # current_vec = np.concatenate((current_vec,[5*vec]), axis = 0)
        rotationthing = R.align_vectors(xaxis, current_vec)
        #below is actual rotation matrix if needed
        #rot_mat = rotationthing[0].as_matrix()
        rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)
        Euler_Angles = np.array([rotthing_euler[0], rotthing_euler[1], rotthing_euler[2]])
    elif align_method == 'trajectory':
        #if the csvdir is a string read the csv file, if it's a dict turn it into a DataFrame
        if type(csvdir)==str:
            infopath = csvdir + cell_name + '_cell_info.csv'
            info = pd.read_csv(infopath, index_col=0)
        elif type(csvdir)==dict:
            info = pd.DataFrame(csvdir, index=[0])
        vec = np.array([info.Trajectory_X[0], info.Trajectory_Y[0], info.Trajectory_Z[0]])
        #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        xaxis = np.array([[1,0,0],[0,1,0], [0,0,1]]).astype('float64')
        upnorm = np.cross(vec,[1,0,0])
        # if upnorm[2]<0:
        #     upnorm = upnorm.copy() * -1
        sidenorm = np.cross(vec,upnorm)
        current_vec = np.stack((vec, sidenorm, upnorm), axis = 0)
        # current_vec = np.concatenate((current_vec,[5*vec]), axis = 0)
        rotationthing = R.align_vectors(xaxis, current_vec)
        #below is actual rotation matrix if needed
        #rot_mat = rotationthing[0].as_matrix()
        rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)
        Euler_Angles = np.array([rotthing_euler[0], rotthing_euler[1], rotthing_euler[2]])
        
    if len(im.shape)>3:
        image = im.data[0,:,:,:]
    else:
        image = im.data
    
    
    if len(image.shape) != 3:
        raise ValueError(
            "Incorrect dimensions: {}. Expected 3 dimensions.".format(image.shape)
        )


    # Binarize the input. We assume that everything that is not background will
    # be use for parametrization
    image_ = image.copy()
    image_[image_ > 0] = 1

    # Converting the input image into a mesh using regular marching cubes
    mesh, image_, centroid = shtools_mod.get_mesh_from_image(image=image_, sigma=sigma)
    
    #rotate and scale mesh
    #worked from https://kitware.github.io/vtk-examples/site/Python/PolyData/AlignTwoPolyDatas/
    #rotate around z axis
    transformation = vtk.vtkTransform()
    #rotate the shape
    transformation.RotateWXYZ(Euler_Angles[2], 0, 0, 1)
    transformation.RotateWXYZ(Euler_Angles[0], 1, 0, 0)
    #set scale to actual image scale
    transformation.Scale(xyres, xyres, zstep)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transformation)
    transformFilter.SetInputData(mesh)
    transformFilter.Update()
    mesh = transformFilter.GetOutput()

    
    #rotate around the x axis until you find the widest distance in y
    angles = np.arange(0,360,0.5)
    widths = np.empty(len(angles))
    for i, a in enumerate(angles):
        
        transformation = vtk.vtkTransform()
        #rotate the shape
        transformation.RotateWXYZ(a, 1, 0, 0)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(mesh)
        transformFilter.Update()
        rotatedmesh = transformFilter.GetOutput()
        coords = numpy_support.vtk_to_numpy(rotatedmesh.GetPoints().GetData())
        #store the average of the negative y coordinates
        widths[i] = coords[np.where(coords[:,1]<0)][:,1].mean()
    
    #get the angle that rotates the least to achieve a "width" peak
    both = np.concatenate((widths, widths))
    peaks, properties = signal.find_peaks(abs(both),prominence=0.11, width=55)
    angpeaks = np.concatenate((angles,angles))[peaks]
    tangpeaks = angpeaks.copy()
    tangpeaks = list(set(tangpeaks))
    # tangpeaks[tangpeaks>180] -= 360

    return [cell_name, tangpeaks]

def measure_volume_half(
        mesh,
        domain,
        ):
    
    #turn off warnings
    vtk.vtkObject.GlobalWarningDisplayOff()
    
    #Create a plane in the domain from the input
    plane = vtk.vtkPlane()

    if domain == 'x':
        plane.SetNormal(1,0,0)
    if domain == 'y':
        plane.SetNormal(0,1,0)
    if domain == 'z':
        plane.SetNormal(0,0,1)
    

    clip = vtk.vtkClipPolyData()
    clip.SetClipFunction(plane)
    clip.SetInputData(mesh)
    clip.Update()
    clipped = clip.GetOutput(0)
    
    #get the volume of the intersection
    CellMassProperties = vtk.vtkMassProperties()
    CellMassProperties.SetInputData(clipped)
    
    return CellMassProperties.GetVolume()



def get_shcoeffs_mod(
    image: np.array,
    img_name: str,
    lmax: int,
    xyres: float,
    zstep: float,
    Euler_Angles: np.array,
    exceptions_list: list,
    sigma: float = 0,
    normal_rotation_method: str = 'none',
    provided_normal_rotation_angle: float = 0,
    compute_lcc: bool = True,
    alignment_2d: bool = False,
    make_unique: bool = False,
):

    """Compute spherical harmonics coefficients that describe an object stored as
    an image.

        Calculates the spherical harmonics coefficients that parametrize the shape
        formed by the foreground set of voxels in the input image. The input image
        does not need to be binary and all foreground voxels (background=0) are used
        in the computation. Foreground voxels must form a single connected component.
        If you are sure that this is the case for the input image, you can set
        compute_lcc to False to speed up the calculation. In addition, the shape is
        expected to be centered in the input image.

        Parameters
        ----------
        image : ndarray
            Input image. Expected to have shape ZYX.
        lmax : int
            Order of the spherical harmonics parametrization. The higher the order
            the more shape details are represented.
        zstep : float
            Z step of the image
        xyres : float
            microns/pixel resolution
        Euler_Angles : numpy array
            (3,) array of angles for rotation of the shape
        normal_rotation_method: str
            "widest" is longest axis parallel to trajectory
        
        Returns
        -------
        coeffs_dict : dict
            Dictionary with the spherical harmonics coefficients and the mean square
            error between input and its parametrization
        grid_rec : ndarray
            Parametric grid representing sh parametrization
        image_ : ndarray
            Input image after pre-processing (lcc calculation, smooth and binarization).
        mesh : vtkPolyData
            Polydata representation of image_.
        grid_down : ndarray
            Parametric grid representing input object.
        transform : tuple of floats
            (xc, yc, zc, angle) if alignment_2d is True or
            (xc, yc, zc) if alignment_2d is False. (xc, yc, zc) are the coordinates
            of the shape centroid after alignment; angle is the angle used to align
            the image

        Other parameters
        ----------------
        sigma : float, optional
            The degree of smooth to be applied to the input image, default is 0 (no
            smooth)
        compute_lcc : bool, optional
            Whether to compute the largest connected component before appliying the
            spherical harmonic parametrization, default is True. Set compute_lcc to
            False in case you are sure the input image contains a single connected
            component. It is crucial that parametrization is calculated on a single
            connected component object.
        alignment_2d : bool
            Wheather the image should be aligned in 2d. Default is True.
        make_unique : bool
            Set true to make sure the alignment rotation is unique.

        Notes
        -----
        Alignment mode '2d' allows for keeping the z axis unchanged which might be
        important for some applications.

        Examples
        --------
            import numpy as np
            from aicsshparam import shparam, shtools

            img = np.ones((32,32,32), dtype=np.uint8)

            (coeffs, grid_rec), (image_, mesh, grid, transform) =
                shparam.get_shcoeffs(image=img, lmax=2)
            mse = shtools.get_reconstruction_error(grid, grid_rec)

            print('Coefficients:', coeffs)
        >>> Coefficients: {'shcoeffs_L0M0C': 18.31594310878251, 'shcoeffs_L0M1C': 0.0,
        'shcoeffs_L0M2C': 0.0, 'shcoeffs_L1M0C': 0.020438775421611564, 'shcoeffs_L1M1C':
        -0.0030960466571801513, 'shcoeffs_L1M2C': 0.0, 'shcoeffs_L2M0C':
        -0.0185688727281408, 'shcoeffs_L2M1C': -2.9925077712704384e-05,
        'shcoeffs_L2M2C': -0.009087503958673892, 'shcoeffs_L0M0S': 0.0,
        'shcoeffs_L0M1S': 0.0, 'shcoeffs_L0M2S': 0.0, 'shcoeffs_L1M0S': 0.0,
        'shcoeffs_L1M1S': 3.799611612562637e-05, 'shcoeffs_L1M2S': 0.0,
        'shcoeffs_L2M0S': 0.0, 'shcoeffs_L2M1S': 3.672543904347801e-07,
        'shcoeffs_L2M2S': 0.0002230857005948496}
            print('Error:', mse)
        >>> Error: 2.3738182456948795
    """

    if len(image.shape) != 3:
        raise ValueError(
            "Incorrect dimensions: {}. Expected 3 dimensions.".format(image.shape)
        )

    if image.sum() == 0:
        # raise ValueError("No foreground voxels found. Is the input image empty?")
        warnings.warn(
            "No foreground voxels found. Is the input image empty?" + str(img_name)
        )
        exceptions_list.extend(["No foreground voxels found. Is the input image empty?", img_name])

    # Binarize the input. We assume that everything that is not background will
    # be use for parametrization
    image_ = image.copy()
    image_[image_ > 0] = 1

    # Alignment
    if alignment_2d:
        # Align the points such that the longest axis of the 2d
        # xy max projected shape will be horizontal (along x)
        image_, angle = shtools_mod.align_image_2d(image=image_, make_unique=make_unique)
        image_ = image_.squeeze()

    # Converting the input image into a mesh using regular marching cubes
    mesh, image_, first_center = shtools_mod.get_mesh_from_image(image=image_, sigma=sigma)
    
    #rotate and scale mesh
    #worked from https://kitware.github.io/vtk-examples/site/Python/PolyData/AlignTwoPolyDatas/
    #rotate around z axis
    transformation = vtk.vtkTransform()
    #rotate the shape
    transformation.RotateWXYZ(Euler_Angles[2], 0, 0, 1)
    transformation.RotateWXYZ(Euler_Angles[0], 1, 0, 0)
    #set scale to actual image scale
    transformation.Scale(xyres, xyres, zstep)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transformation)
    transformFilter.SetInputData(mesh)
    transformFilter.Update()
    mesh = transformFilter.GetOutput()
    

    #set a normal rotation angle even if not rotating
    widestangle = 0
    
    #################### normal rotation by provided angle ###############
    if bool(type(normal_rotation_method) == float):
        #get the rotation angle with the most heavy negative y direction bias
        widestangle = normal_rotation_method
        
        #rotate around x by the widest y axis
        transformation = vtk.vtkTransform()
        #rotate the shape
        transformation.RotateWXYZ(widestangle, 1, 0, 0)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(mesh)
        transformFilter.Update()
        mesh = transformFilter.GetOutput()
        
        
    #################### find the absolute widest axis perpendicular to the trajectory ###############
    elif normal_rotation_method == 'widest':
        #rotate around the x axis until you find the widest distance in y
        angles = np.arange(0,360,0.5)
        widths = np.empty((len(angles),3))
        for i, a in enumerate(angles):
            
            transformation = vtk.vtkTransform()
            #rotate the shape
            transformation.RotateWXYZ(a, 1, 0, 0)
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(transformation)
            transformFilter.SetInputData(mesh)
            transformFilter.Update()
            rotatedmesh = transformFilter.GetOutput()
            coords = numpy_support.vtk_to_numpy(rotatedmesh.GetPoints().GetData())
            #first column is aboslute width, second is + width, third is - width
            widths[i,:] = np.array([[np.max(coords[:, 1]) - np.min(coords[:, 1]),
                                                      np.max(coords[:, 1]),
                                                      np.min(coords[:, 1])]])
            
        
        bigtwo = widths[np.where(widths[:, 0] == np.max(widths[:, 0]))]# & (widths[:, 2] == np.max(widths[:, 2])))
        bigone = bigtwo[np.where(bigtwo[:, 2] == np.min(bigtwo[:, 2]))]
        widestangle = angles[np.where((widths == (bigone)).all(axis=1))[0][0]]
        
        #rotate around x by the widest y axis
        transformation = vtk.vtkTransform()
        #rotate the shape
        transformation.RotateWXYZ(widestangle, 1, 0, 0)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(mesh)
        transformFilter.Update()
        mesh = transformFilter.GetOutput()


    ##################### find the "weighted" widest axis perpendicular to trajectory ##################
    elif normal_rotation_method == 'widest weighted':
        #rotate around the x axis until you find the widest distance in y
        angles = np.arange(0,360,0.5)
        widths = np.empty(len(angles))
        for i, a in enumerate(angles):
            
            transformation = vtk.vtkTransform()
            #rotate the shape
            transformation.RotateWXYZ(a, 1, 0, 0)
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(transformation)
            transformFilter.SetInputData(mesh)
            transformFilter.Update()
            rotatedmesh = transformFilter.GetOutput()
            coords = numpy_support.vtk_to_numpy(rotatedmesh.GetPoints().GetData())
            #store the average of the negative y coordinates
            widths[i] = coords[np.where(coords[:,1]<0)][:,1].mean()
        
        #get the rotation angle with the most heavy negative y direction bias
        widestangle = angles[np.where(widths==widths.min())[0][0]]
        
        #rotate around x by the widest y axis
        transformation = vtk.vtkTransform()
        #rotate the shape
        transformation.RotateWXYZ(widestangle, 1, 0, 0)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(mesh)
        transformFilter.Update()
        mesh = transformFilter.GetOutput()
        
        
    ##################### find the widest axis perpendicular to trajectory by volume ##################
    elif normal_rotation_method == 'widest volume':
        #rotate around the x axis until you find the widest distance in y
        angles = np.arange(0,360,0.5)
        widths = np.empty(len(angles))
        for i, a in enumerate(angles):
            
            transformation = vtk.vtkTransform()
            #rotate the shape
            transformation.RotateWXYZ(a, 1, 0, 0)
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(transformation)
            transformFilter.SetInputData(mesh)
            transformFilter.Update()
            rotatedmesh = transformFilter.GetOutput()

            #store the volume in the positive y direction
            widths[i] = measure_volume_half(rotatedmesh, 'y')
        
        #get the rotation angle with the most heavy negative y direction bias
        widestangle = angles[np.where(widths==widths.max())[0][0]]
        
        #rotate around x by the widest y axis
        transformation = vtk.vtkTransform()
        #rotate the shape
        transformation.RotateWXYZ(widestangle, 1, 0, 0)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(mesh)
        transformFilter.Update()
        mesh = transformFilter.GetOutput()


    elif normal_rotation_method == 'first wide':
        #rotate around the x axis until you find the widest distance in y
        angles = np.arange(0,360,0.5)
        widths = np.empty(len(angles))
        for i, a in enumerate(angles):
            
            transformation = vtk.vtkTransform()
            #rotate the shape
            transformation.RotateWXYZ(a, 1, 0, 0)
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(transformation)
            transformFilter.SetInputData(mesh)
            transformFilter.Update()
            rotatedmesh = transformFilter.GetOutput()
            coords = numpy_support.vtk_to_numpy(rotatedmesh.GetPoints().GetData())
            #store the average of the negative y coordinates
            widths[i] = coords[np.where(coords[:,1]<0)][:,1].mean()
        
        #get the angle that rotates the least to achieve a "width" peak
        both = np.concatenate((widths, widths))
        peaks, properties = signal.find_peaks(abs(both),prominence=0.13, width=70)
        angpeaks = np.concatenate((angles,angles))[peaks]
        tangpeaks = angpeaks.copy()
        tangpeaks[tangpeaks>180] -= 360
        widestangle = angpeaks[np.argmin(abs(tangpeaks))]

        #rotate around x by the widest y axis
        transformation = vtk.vtkTransform()
        #rotate the shape
        transformation.RotateWXYZ(widestangle, 1, 0, 0)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(mesh)
        transformFilter.Update()
        mesh = transformFilter.GetOutput()


    elif normal_rotation_method == 'widest and flip':
        volumes = []
        vtk.vtkObject.GlobalWarningDisplayOff()
        #rotate around the x axis until you find the widest distance in y
        angles = np.arange(0,360,0.5)
        widths = np.empty((0,3))
        for a in angles:
            
            transformation = vtk.vtkTransform()
            #rotate the shape
            transformation.RotateWXYZ(a, 1, 0, 0)
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(transformation)
            transformFilter.SetInputData(mesh)
            transformFilter.Update()
            rotatedmesh = transformFilter.GetOutput()
            coords = numpy_support.vtk_to_numpy(rotatedmesh.GetPoints().GetData())
            #first column is aboslute width, second is + width, third is - width
            widths = np.append(widths, np.array([[np.max(coords[:, 1]) - np.min(coords[:, 1]),
                                                      np.max(coords[:, 1]),
                                                      np.min(coords[:, 1])]]), axis = 0)
            
        
        bigtwo = widths[np.where(widths[:, 0] == np.max(widths[:, 0]))]# & (widths[:, 2] == np.max(widths[:, 2])))
        bigone = bigtwo[np.where(bigtwo[:, 2] == np.min(bigtwo[:, 2]))]
        widestangle = angles[np.where((widths == (bigone)).all(axis=1))[0][0]]
        
        #rotate around x by the widest y axis
        transformation = vtk.vtkTransform()
        #rotate the shape
        transformation.RotateWXYZ(widestangle, 1, 0, 0)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(mesh)
        transformFilter.Update()
        mesh = transformFilter.GetOutput()
        
        tri1 = vtk.vtkTriangleFilter()
        tri1.SetInputData(mesh)
        clean1 = vtk.vtkCleanPolyData()
        clean1.SetInputConnection(tri1.GetOutputPort())
        clean1.Update()
        mesh = clean1.GetOutput()
        
        #open example mesh to compare with current mesh
        FILE_NAME = 'C:/Users/Aaron/Documents/PythonScripts/Simulations/Example_Mesh.vtp'
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(FILE_NAME)
        reader.Update()
        example_mesh = reader.GetOutput()
    
    
        #get intersection volume with example
        booleanOperation = vtk.vtkBooleanOperationPolyDataFilter()
        booleanOperation.SetInputData(0, mesh)
        booleanOperation.SetInputData(1, example_mesh)
        booleanOperation.SetOperationToIntersection()
        booleanOperation.Update()
        Intersection = booleanOperation.GetOutput()
        
        massProperties = vtk.vtkMassProperties()
        massProperties.SetInputData(Intersection)
        volumes.append(massProperties.GetVolume())
    
        #flip banana 180 around y axis
        transformation = vtk.vtkTransform()
        transformation.RotateWXYZ(180, 0, 1, 0)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(mesh)
        transformFilter.Update()
        flipped_mesh1 = transformFilter.GetOutput()
    
    
        booleanOperation = vtk.vtkBooleanOperationPolyDataFilter()
        booleanOperation.SetOperationToIntersection()
        booleanOperation.SetInputData(0, example_mesh)
        booleanOperation.SetInputData(1, flipped_mesh1)
        booleanOperation.Update()
        Intersection = booleanOperation.GetOutput()
        
        massProperties = vtk.vtkMassProperties()
        massProperties.SetInputData(Intersection)
        volumes.append(massProperties.GetVolume())
    
        #flip banana 180 around y axis
        transformation = vtk.vtkTransform()
        transformation.RotateWXYZ(180, 1, 0, 0)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(flipped_mesh1)
        transformFilter.Update()
        flipped_mesh2 = transformFilter.GetOutput()
    
    
        booleanOperation = vtk.vtkBooleanOperationPolyDataFilter()
        booleanOperation.SetOperationToIntersection()
        booleanOperation.SetInputData(0, example_mesh)
        booleanOperation.SetInputData(1, flipped_mesh2)
        booleanOperation.Update()
        Intersection = booleanOperation.GetOutput()
        
        massProperties = vtk.vtkMassProperties()
        massProperties.SetInputData(Intersection)
        volumes.append(massProperties.GetVolume())
    
    
        #flip banana 180 around y axis
        transformation = vtk.vtkTransform()
        transformation.RotateWXYZ(180, 0, 1, 0)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(flipped_mesh2)
        transformFilter.Update()
        flipped_mesh3 = transformFilter.GetOutput()
    
    
        booleanOperation = vtk.vtkBooleanOperationPolyDataFilter()
        booleanOperation.SetOperationToIntersection()
        booleanOperation.SetInputData(0, example_mesh)
        booleanOperation.SetInputData(1, flipped_mesh3)
        booleanOperation.Update()
        Intersection = booleanOperation.GetOutput()
        
        massProperties = vtk.vtkMassProperties()
        massProperties.SetInputData(Intersection)
        volumes.append(massProperties.GetVolume())
    
        max_value = max(volumes)
        max_index = volumes.index(max_value)
        if max_index == 0:
            mesh = mesh
        elif max_index == 1:
            mesh = flipped_mesh1
        elif max_index == 2:
            mesh = flipped_mesh2
        elif max_index == 3:
            mesh = flipped_mesh3
        


    #rotate the mesh around x by a specific angle
    if provided_normal_rotation_angle>0:
        #rotate around x by the widest y axis
        transformation = vtk.vtkTransform()
        #rotate the shape
        transformation.RotateWXYZ(provided_normal_rotation_angle, 1, 0, 0)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(mesh)
        transformFilter.Update()
        mesh = transformFilter.GetOutput()
    
    
    
    if not image_[tuple([int(u) for u in first_center[::-1]])]:
        warnings.warn(
            "Mesh centroid seems to fall outside the object. This indicates\
        the mesh may not be a manifold suitable for spherical harmonics\
        parameterization." + str(img_name)
        )
        exceptions_list.extend(["Mesh centroid seems to fall outside the object. This indicates\
        the mesh may not be a manifold suitable for spherical harmonics\
        parameterization.", img_name])

    # Get coordinates of mesh points
    coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    #get the centroid
    second_center = coords.mean(axis=0, keepdims=True)
    #subtract centroid from coordinates
    coords -= second_center
    #separate the coordinates
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    transform = tuple((np.array(first_center).squeeze(), second_center.squeeze())) + ((angle,) if alignment_2d else ())

    # Translate and update mesh normals
    mesh = shtools_mod.update_mesh_points(mesh, x, y, z)

    # Cartesian to spherical coordinates convertion
    rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    lat = np.arccos(np.divide(z, rad, out=np.zeros_like(rad), where=(rad != 0)))
    lon = np.pi + np.arctan2(y, x)

    # Creating a meshgrid data from (lon,lat,r)
    points = np.concatenate(
        [np.array(lon).reshape(-1, 1), np.array(lat).reshape(-1, 1)], axis=1
    )

    grid_lon, grid_lat = np.meshgrid(
        np.linspace(start=0, stop=2 * np.pi, num=256, endpoint=True),
        np.linspace(start=0, stop=1 * np.pi, num=128, endpoint=True),
    )

    # Interpolate the (lon,lat,r) data into a grid
    grid = spinterp.griddata(points, rad, (grid_lon, grid_lat), method="nearest")

    # Fit grid data with SH. Look at pyshtools for detail.
    coeffs = pyshtools.expand.SHExpandDH(grid, sampling=2, lmax_calc=lmax)

    # Reconstruct grid. Look at pyshtools for detail.
    grid_rec = pyshtools.expand.MakeGridDH(coeffs, sampling=2)

    # Resize the input grid to match the size of the reconstruction
    grid_down = sktrans.resize(grid, output_shape=grid_rec.shape, preserve_range=True)

    # Create (l,m) keys for the coefficient dictionary
    lvalues = np.repeat(np.arange(lmax + 1).reshape(-1, 1), lmax + 1, axis=1)

    keys = []
    for suffix in ["C", "S"]:
        for (l, m) in zip(lvalues.flatten(), lvalues.T.flatten()):
            keys.append(f"shcoeffs_L{l}M{m}{suffix}")

    coeffs_dict = dict(zip(keys, coeffs.flatten()))

    return (coeffs_dict, grid_rec, widestangle, exceptions_list), (image_, mesh, grid_down, transform)

def get_shcoeffs_mesh(
        mesh,
        lmax: str):
    coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    # Translate and update mesh normals
    mesh = shtools_mod.update_mesh_points(mesh, x, y, z)

    # Cartesian to spherical coordinates convertion
    rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    lat = np.arccos(np.divide(z, rad, out=np.zeros_like(rad), where=(rad != 0)))
    lon = np.pi + np.arctan2(y, x)

    # Creating a meshgrid data from (lon,lat,r)
    points = np.concatenate(
        [np.array(lon).reshape(-1, 1), np.array(lat).reshape(-1, 1)], axis=1
    )

    grid_lon, grid_lat = np.meshgrid(
        np.linspace(start=0, stop=2 * np.pi, num=256, endpoint=True),
        np.linspace(start=0, stop=1 * np.pi, num=128, endpoint=True),
    )

    # Interpolate the (lon,lat,r) data into a grid
    grid = spinterp.griddata(points, rad, (grid_lon, grid_lat), method="nearest")

    # Fit grid data with SH. Look at pyshtools for detail.
    coeffs = pyshtools.expand.SHExpandDH(grid, sampling=2, lmax_calc=lmax)

    # Reconstruct grid. Look at pyshtools for detail.
    grid_rec = pyshtools.expand.MakeGridDH(coeffs, sampling=2)

    # Resize the input grid to match the size of the reconstruction
    grid_down = sktrans.resize(grid, output_shape=grid_rec.shape, preserve_range=True)

    # Create (l,m) keys for the coefficient dictionary
    lvalues = np.repeat(np.arange(lmax + 1).reshape(-1, 1), lmax + 1, axis=1)

    keys = []
    for suffix in ["C", "S"]:
        for (l, m) in zip(lvalues.flatten(), lvalues.T.flatten()):
            keys.append(f"shcoeffs_L{l}M{m}{suffix}")

    coeffs_dict = dict(zip(keys, coeffs.flatten()))

    return (coeffs_dict, grid_rec), (grid_down)


def get_shcoeffs_shiftres(
    img_name: str,
    image: np.array,
    lmax: int,
    xyres: float,
    zstep: float,
    exceptions_list: list,
    sigma: float = 0,
    compute_lcc: bool = True,
    alignment_2d: bool = True,
    make_unique: bool = False,
    center_align: bool = True,
    center_point: np.array = None,
):



    if len(image.shape) != 3:
        raise ValueError(
            "Incorrect dimensions: {}. Expected 3 dimensions.".format(image.shape)
        )

    if image.sum() == 0:
        # raise ValueError("No foreground voxels found. Is the input image empty?")
        warnings.warn(
            "No foreground voxels found. Is the input image empty?" + str(img_name)
        )
        exceptions_list.extend(["No foreground voxels found. Is the input image empty?", img_name])

    # Binarize the input. We assume that everything that is not background will
    # be use for parametrization
    image_ = image.copy()
    image_[image_ > 0] = 1

    # Alignment
    if alignment_2d:
        # Align the points such that the longest axis of the 2d
        # xy max projected shape will be horizontal (along x)
        image_, angle = shtools_mod.align_image_2d(image=image_, make_unique=make_unique)
        image_ = image_.squeeze()

    # Converting the input image into a mesh using regular marching cubes
    if center_align == False:
        mesh, image_, centroid = shtools_mod.get_mesh_from_image(image=image_, sigma=sigma, translate_to_origin=center_align, center=center_point)
    else:
        mesh, image_, centroid = shtools_mod.get_mesh_from_image(image=image_, sigma=sigma)
    
    
    if not image_[tuple([int(u) for u in centroid[::-1]])]:
        warnings.warn(
            "Mesh centroid seems to fall outside the object. This indicates\
        the mesh may not be a manifold suitable for spherical harmonics\
        parameterization." + str(img_name)
        )
        exceptions_list.extend(["Mesh centroid seems to fall outside the object. This indicates\
        the mesh may not be a manifold suitable for spherical harmonics\
        parameterization.", img_name])
    
    
    #rotate and scale mesh
    #worked from https://kitware.github.io/vtk-examples/site/Python/PolyData/AlignTwoPolyDatas/
    #rotate around z axis
    transformation = vtk.vtkTransform()
    #set scale to actual image scale
    transformation.Scale(xyres, xyres, zstep)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transformation)
    transformFilter.SetInputData(mesh)
    transformFilter.Update()
    mesh = transformFilter.GetOutput()
    

    # Get coordinates of mesh points
    coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    transform = centroid + ((angle,) if alignment_2d else ())

    # Translate and update mesh normals
    mesh = shtools_mod.update_mesh_points(mesh, x, y, z)

    # Cartesian to spherical coordinates convertion
    rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    lat = np.arccos(np.divide(z, rad, out=np.zeros_like(rad), where=(rad != 0)))
    lon = np.pi + np.arctan2(y, x)

    # Creating a meshgrid data from (lon,lat,r)
    points = np.concatenate(
        [np.array(lon).reshape(-1, 1), np.array(lat).reshape(-1, 1)], axis=1
    )

    grid_lon, grid_lat = np.meshgrid(
        np.linspace(start=0, stop=2 * np.pi, num=256, endpoint=True),
        np.linspace(start=0, stop=1 * np.pi, num=128, endpoint=True),
    )

    # Interpolate the (lon,lat,r) data into a grid
    grid = spinterp.griddata(points, rad, (grid_lon, grid_lat), method="nearest")

    # Fit grid data with SH. Look at pyshtools for detail.
    coeffs = pyshtools.expand.SHExpandDH(grid, sampling=2, lmax_calc=lmax)

    # Reconstruct grid. Look at pyshtools for detail.
    grid_rec = pyshtools.expand.MakeGridDH(coeffs, sampling=2)

    # Resize the input grid to match the size of the reconstruction
    grid_down = sktrans.resize(grid, output_shape=grid_rec.shape, preserve_range=True)

    # Create (l,m) keys for the coefficient dictionary
    lvalues = np.repeat(np.arange(lmax + 1).reshape(-1, 1), lmax + 1, axis=1)

    keys = []
    for suffix in ["C", "S"]:
        for (l, m) in zip(lvalues.flatten(), lvalues.T.flatten()):
            keys.append(f"shcoeffs_L{l}M{m}{suffix}")

    coeffs_dict = dict(zip(keys, coeffs.flatten()))

    return (coeffs_dict, grid_rec, exceptions_list), (image_, mesh, grid_down, transform)



def axis_order(ax):
    if ax == 'X':
        return [1, 0, 0]
    if ax == 'Y':
        return [0, 1, 0]
    if ax == 'Z':
        return [0, 0, 1]


def get_shcoeffs_ReverseNanerRotation(
    image: np.array,
    lmax: int,
    xyres: float,
    zstep: float,
    row: pd.core.series.Series,
    sigma: float = 0,
    compute_lcc: bool = True,
    alignment_2d: bool = True,
    make_unique: bool = False,
):


    if len(image.shape) != 3:
        raise ValueError(
            "Incorrect dimensions: {}. Expected 3 dimensions.".format(image.shape)
        )

    if image.sum() == 0:
        raise ValueError("No foreground voxels found. Is the input image empty?")

    # Binarize the input. We assume that everything that is not background will
    # be use for parametrization
    image_ = image.copy()
    image_[image_ > 0] = 1

    # Alignment
    if alignment_2d:
        # Align the points such that the longest axis of the 2d
        # xy max projected shape will be horizontal (along x)
        image_, angle = shtools_mod.align_image_2d(image=image_, make_unique=make_unique)
        image_ = image_.squeeze()

    # Converting the input image into a mesh using regular marching cubes
    mesh, image_, centroid = shtools_mod.get_mesh_from_image(image=image_, sigma=sigma)
    
    Third_rot = axis_order(row.Third)
    Second_rot = axis_order(row.Second)
    First_rot = axis_order(row.First)
    Euler_angles = np.array([row[row.First], row[row.Second], row[row.Third]])
    
    #rotate and scale mesh
    #worked from https://kitware.github.io/vtk-examples/site/Python/PolyData/AlignTwoPolyDatas/
    #rotate around z axis
    transformation = vtk.vtkTransform()
    #rotate the shape
    transformation.RotateWXYZ(90, 1, 0, 0)
    transformation.RotateWXYZ(Euler_angles[0], First_rot[0], First_rot[1], First_rot[2])
    transformation.RotateWXYZ(Euler_angles[1], Second_rot[0], Second_rot[1], Second_rot[2])
    transformation.RotateWXYZ(Euler_angles[2], Third_rot[0], Third_rot[1], Third_rot[2])
    #set scale to actual image scale
    transformation.Scale(xyres, xyres, zstep)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transformation)
    transformFilter.SetInputData(mesh)
    transformFilter.Update()
    mesh = transformFilter.GetOutput()
    
    
    
    
    if not image_[tuple([int(u) for u in centroid[::-1]])]:
        warnings.warn(
            "Mesh centroid seems to fall outside the object. This indicates\
        the mesh may not be a manifold suitable for spherical harmonics\
        parameterization."
        )

    # Get coordinates of mesh points
    coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    transform = centroid + ((angle,) if alignment_2d else ())

    # Translate and update mesh normals
    mesh = shtools_mod.update_mesh_points(mesh, x, y, z)

    # Cartesian to spherical coordinates convertion
    rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    lat = np.arccos(np.divide(z, rad, out=np.zeros_like(rad), where=(rad != 0)))
    lon = np.pi + np.arctan2(y, x)

    # Creating a meshgrid data from (lon,lat,r)
    points = np.concatenate(
        [np.array(lon).reshape(-1, 1), np.array(lat).reshape(-1, 1)], axis=1
    )

    grid_lon, grid_lat = np.meshgrid(
        np.linspace(start=0, stop=2 * np.pi, num=256, endpoint=True),
        np.linspace(start=0, stop=1 * np.pi, num=128, endpoint=True),
    )

    # Interpolate the (lon,lat,r) data into a grid
    grid = spinterp.griddata(points, rad, (grid_lon, grid_lat), method="nearest")

    # Fit grid data with SH. Look at pyshtools for detail.
    coeffs = pyshtools.expand.SHExpandDH(grid, sampling=2, lmax_calc=lmax)

    # Reconstruct grid. Look at pyshtools for detail.
    grid_rec = pyshtools.expand.MakeGridDH(coeffs, sampling=2)

    # Resize the input grid to match the size of the reconstruction
    grid_down = sktrans.resize(grid, output_shape=grid_rec.shape, preserve_range=True)

    # Create (l,m) keys for the coefficient dictionary
    lvalues = np.repeat(np.arange(lmax + 1).reshape(-1, 1), lmax + 1, axis=1)

    keys = []
    for suffix in ["C", "S"]:
        for (l, m) in zip(lvalues.flatten(), lvalues.T.flatten()):
            keys.append(f"shcoeffs_L{l}M{m}{suffix}")

    coeffs_dict = dict(zip(keys, coeffs.flatten()))

    return (coeffs_dict, grid_rec), (image_, mesh, grid_down, transform)







def recondistances(
        img_name,
        cell,
        nucleus,
        l_order_num,
        xyres,
        zstep,
        Euler_Angles,
        ):
    
    ###### makes spherical harmonic reconstructions of the cell at various values of l 
    ###### starting with the segmented image of the cell
    
    
    distances = pd.DataFrame()
    grid_se = pd.DataFrame()
    nuc_distances = pd.DataFrame()
    nuc_grid_se = pd.DataFrame()
    
    ########### do a bunch of reconstructions for the cell #################
    for l_order in range(l_order_num):
        ordernumber = 'Lmax '+ str(l_order)
        (coeffs, grid_rec, widestangle), (image_, cell_mesh, grid, transform) = get_shcoeffs_mod(
            image = cell,
            lmax = l_order,
            xyres = xyres,
            zstep = zstep,
            Euler_Angles =  Euler_Angles,
            sigma = 2,
            normal_rotation_method='widest',
        )
        #get standard error of reconstruction from grid
        grid_se[ordernumber] = pd.Series(shtools_mod.get_reconstruction_error(grid, grid_rec))

        #put all the coefficients into a pandas dataframe
        cell_coeffs = pd.DataFrame([coeffs.values()],
                                  index = [img_name.replace('_segmented.tiff','')],
                                  columns = list(coeffs.keys()))

        #split up all the cofficients so they can be fed to the get_reconstruction_from_coeffs function
        coeff_names = list(cell_coeffs.columns)
        L_coeffs = len(np.unique(re.findall('L\d*', ''.join(coeff_names))))
        M_coeffs = len(np.unique(re.findall('M\d*', ''.join(coeff_names))))
        #reconstruct from coefficients
        cell_recon, grid_recon = shtools_mod.get_reconstruction_from_coeffs(np.array(cell_coeffs).reshape(2,L_coeffs,M_coeffs))
        #get average nearest distance for this particular reconstruction
        tree = KDTree(numpy_support.vtk_to_numpy(cell_mesh.GetPoints().GetData()))
        d, idx = tree.query(numpy_support.vtk_to_numpy(cell_recon.GetPoints().GetData()))
        distances[ordernumber] = pd.Series(np.mean(d))

        #################do all of the same reconstructions for the nucleus##############
        (coeffs, grid_rec, widestangle), (image_, nuc_mesh, grid, transform) = get_shcoeffs_mod(
            image = nucleus,
            lmax = l_order,
            xyres = xyres,
            zstep = zstep,
            Euler_Angles =  Euler_Angles,
            sigma = 2,
            compute_lcc = True,
            alignment_2d = False,
            make_unique = False,
        )
        #get standard error of reconstruction from grid
        nuc_grid_se[ordernumber] = pd.Series(shtools_mod.get_reconstruction_error(grid, grid_rec))

        #put all the coefficients into a pandas dataframe
        nuc_coeffs = pd.DataFrame([coeffs.values()],
                                  index = [img_name.replace('_segmented.tiff','')],
                                  columns = list(coeffs.keys()))
        #split up all the cofficients so they can be fed to the get_reconstruction_from_coeffs function
        coeff_names = list(nuc_coeffs.columns)
        L_coeffs = len(np.unique(re.findall('L\d*', ''.join(coeff_names))))
        M_coeffs = len(np.unique(re.findall('M\d*', ''.join(coeff_names))))
        #reconstruct from coefficients
        nuc_recon, grid_recon = shtools_mod.get_reconstruction_from_coeffs(np.array(nuc_coeffs).reshape(2,L_coeffs,M_coeffs))
        #get average nearest distance for this particular reconstruction
        tree = KDTree(numpy_support.vtk_to_numpy(nuc_mesh.GetPoints().GetData()))
        d, idx = tree.query(numpy_support.vtk_to_numpy(nuc_recon.GetPoints().GetData()))
        nuc_distances[ordernumber] = pd.Series(np.mean(d))
    
    return([img_name, distances.to_dict(), grid_se.to_dict(), nuc_distances.to_dict(), nuc_grid_se.to_dict()])


def recondistancesmesh(
        meshdir,
        img_name,
        l_order_num,
        ):

    ###### makes spherical harmonic reconstructions of the cell at various values of l 
    ###### starting with the surface mesh of the cell
    
    mesh = vtk.vtkXMLPolyDataReader()
    mesh.SetFileName(meshdir + img_name)
    mesh.Update()
    mesh = mesh.GetOutput()
    
    OtRdistances = {}
    RtOdistances = {}
    grid_se = {}
    
    ########### do a bunch of reconstructions for the cell #################
    for l_order in range(l_order_num):
        ordernumber = 'Lmax '+ str(l_order)
        (coeffs, grid_rec), (grid) = get_shcoeffs_mesh(
            mesh = mesh,
            lmax = l_order,
        )
        #get standard error of reconstruction from grid
        grid_se[ordernumber] = shtools_mod.get_reconstruction_error(grid, grid_rec)

        #put all the coefficients into a pandas dataframe
        cell_coeffs = pd.DataFrame([coeffs.values()],
                                  index = [img_name.replace('_segmented.tiff','')],
                                  columns = list(coeffs.keys()))

        #split up all the cofficients so they can be fed to the get_reconstruction_from_coeffs function
        coeff_names = list(cell_coeffs.columns)
        L_coeffs = len(np.unique(re.findall('L\d*', ''.join(coeff_names))))
        M_coeffs = len(np.unique(re.findall('M\d*', ''.join(coeff_names))))
        #reconstruct from coefficients
        cell_recon, grid_recon = shtools_mod.get_reconstruction_from_coeffs(np.array(cell_coeffs).reshape(2,L_coeffs,M_coeffs))
        #get average nearest distance for this particular reconstruction
        tree = KDTree(numpy_support.vtk_to_numpy(mesh.GetPoints().GetData()))
        d, idx = tree.query(numpy_support.vtk_to_numpy(cell_recon.GetPoints().GetData()))
        OtRdistances[ordernumber] = np.mean(d)


        #get average nearest distance for this particular reconstruction
        tree = KDTree(numpy_support.vtk_to_numpy(cell_recon.GetPoints().GetData()))
        d, idx = tree.query(numpy_support.vtk_to_numpy(mesh.GetPoints().GetData()))
        RtOdistances[ordernumber] = np.mean(d)

    return([img_name, OtRdistances, RtOdistances, grid_se])



def shcoeffs_and_PILR(
        impath: str,
        savedir: str,
        xyres: float,
        zstep: float,
        str_name: str,
        exceptions_list: list,
        normal_rotation_method: str,
        l_order: int,
        sigma: float = 0,
        align_method: str = 'None',
        ):

    """
        Parameters
        ----------
        impath : str
            Input image path. Expected to have shape CZYX. Channel order must be Membrane, Nucleus, Structure
        savedir : str
            Directory that will contain the Mesh and PILR folders
        xyres : float
            microns/pixel resolution of the image
        zstep : float
            Z step of the image
        normal_rotation_method : str
            "widest" is longest axis parallel to trajectory
        exceptions_list : List
            List to append names of images that cannot be represented well by SH
        str_name : str
            String detailing the name of the intracellular structure in the image
        l_order : int
            l order for SH transformation
        
        Returns
        -------
        coeffs_dict : dict
            Dictionary with the spherical harmonics coefficients and the mean square
            error between input and its parametrization
        grid_rec : ndarray
            Parametric grid representing sh parametrization
        image_ : ndarray
            Input image after pre-processing (lcc calculation, smooth and binarization).
        mesh : vtkPolyData
            Polydata representation of image_.
        grid_down : ndarray
            Parametric grid representing input object.
        transform : tuple of floats
            (xc, yc, zc, angle) if alignment_2d is True or
            (xc, yc, zc) if alignment_2d is False. (xc, yc, zc) are the coordinates
            of the shape centroid after alignment; angle is the angle used to align
            the image

        Other parameters
        ----------------
        sigma : float, optional
            The degree of smooth to be applied to the input image, default is 0 (no
            smooth)
        compute_lcc : bool, optional
            Whether to compute the largest connected component before appliying the
            spherical harmonic parametrization, default is True. Set compute_lcc to
            False in case you are sure the input image contains a single connected
            component. It is crucial that parametrization is calculated on a single
            connected component object.
        alignment_2d : bool
            Wheather the image should be aligned in 2d. Default is True.
        make_unique : bool
            Set true to make sure the alignment rotation is unique. 
            """
    #get cell name from impath
    cell_name = impath.split('/')[-1].split('_segmented')[0]
    #read image
    im = TiffReader(impath)
    imd = im.data

    #read euler angles for alignment
    infopath = '/'.join(impath.split('/')[:-1]) + '/' + cell_name + '_cell_info.csv'
    if align_method == 'trajectory':
        info = pd.read_csv(infopath, index_col=0)
        vec = np.array([info.Trajectory_X[0], info.Trajectory_Y[0], info.Trajectory_Z[0]])
        #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        xaxis = np.array([[0,0,0],[1,0,0], [5,0,0]]).astype('float64')
        current_vec = np.stack(([0,0,0],vec), axis = 0)
        current_vec = np.concatenate((current_vec,[5*vec]), axis = 0)
        rotationthing = R.align_vectors(xaxis, current_vec)
        #below is actual rotation matrix if needed
        #rot_mat = rotationthing[0].as_matrix()
        rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)
        euler_angles = np.array([rotthing_euler[0], rotthing_euler[1], rotthing_euler[2]])


    #get cell mesh and coeffs
    (coeffs_mem, grid_rec, widestangle, exceptions_list), (image_, cell_mesh, grid, centroid_mem) = get_shcoeffs_mod(        
        image = imd[0,:,:,:],
        img_name= cell_name,
        lmax = 16,
        xyres = xyres/xyres,
        zstep = zstep/xyres,
        Euler_Angles = euler_angles,
        exceptions_list = exceptions_list,
        sigma = sigma,
        normal_rotation_method = normal_rotation_method,
        )
    
    
    #get nuc coeffs
    (coeffs_nuc, _, _, nuc_list), (_, nuc_mesh, _, centroid_nuc) = get_shcoeffs_mod(
        image = imd[2,:,:,:],
        img_name = cell_name,
        lmax = 16,
        xyres = xyres/xyres,
        zstep = zstep/xyres,
        Euler_Angles = euler_angles,
        provided_normal_rotation_angle = widestangle,
        exceptions_list = [],
        sigma = sigma,
        )
    
    #get nuc_mesh in correct posistion relative to the cell
    nuc_mesh, _, _ = shtools_mod.get_mesh_from_image_and_rotate(
        image = imd[2,:,:,:],
        xyres = xyres/xyres,
        zstep = zstep/xyres,
        Euler_Angles = euler_angles,
        provided_normal_rotation_angle = widestangle,
        translate_to_origin=False,
        center = np.array(centroid_mem),
    )
    
    #get centroid of the structure mesh
    str_mesh, _, str_cent = shtools_mod.get_mesh_from_image(
        image = imd[1,:,:,:],
        sigma = 0,
        lcc = False
        )
    
    #get structure mesh
    str_mesh, _, cent = shtools_mod.get_mesh_from_image_and_rotate(
        image = imd[1,:,:,:],
        xyres = xyres/xyres,
        zstep = zstep/xyres,
        Euler_Angles = euler_angles,
        provided_normal_rotation_angle = widestangle,
        translate_to_origin=False,
        lcc = False,
        center = np.array(centroid_mem)
    )
    
    
    #get voxelized intracellular structure image
    img, origin = cytoparam_mod.voxelize_meshes([cell_mesh,nuc_mesh,str_mesh])
    
    #get rotated segmentented str signal alone
    strimg = img.copy()
    strimg[strimg<3]=0
    strimg[strimg>0]=255
    
    #########parameterize cell
    aicstif = cytoparam_mod.cellular_mapping(
        coeffs_mem = coeffs_mem,
        centroid_mem = centroid_mem,
        coeffs_nuc = coeffs_nuc,
        centroid_nuc = centroid_nuc,
        nisos = [32,32],
        images_to_probe = [[str_name,strimg]],
        )
          
    
    #Save PILR
    pilrf = savedir+'PILRs/'
    if not os.path.exists(pilrf):
        os.makedirs(pilrf)
    if os.path.exists(pilrf+cell_name+'_PILR.tif'):
        os.remove(pilrf+cell_name+'_PILR.tif')
    OmeTiffWriter.save(aicstif.get_image_data('CZYX', S=0, T=0), pilrf+cell_name+'_PILR.tif', dim_order='CZYX', channel_names=aicstif.channel_names)
    
    
    
    # remove file if it already exists
    meshf = savedir+'Meshes/'
    if not os.path.exists(meshf):
        os.makedirs(meshf)
    #save cell mesh
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(meshf + cell_name + '_cell_mesh.vtp')
    writer.SetInputData(cell_mesh)
    writer.Write()
    
    #save nuclear mesh
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(meshf + cell_name + '_nuc_mesh.vtp')
    writer.SetInputData(nuc_mesh)
    writer.Write()
    
    #save str mesh
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(meshf + cell_name + '_str_mesh.vtp')
    writer.SetInputData(str_mesh)
    writer.Write()
    
    
    
    #scale cell and nuc meshes so that I can take some shape stats
    #set transform and apply to cell
    transformation = vtk.vtkTransform()
    transformation.Scale(xyres, xyres, xyres)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transformation)
    transformFilter.SetInputData(cell_mesh)
    transformFilter.Update()
    cell_mesh = transformFilter.GetOutput()
    #set transform and apply to nucleus
    transformation = vtk.vtkTransform()
    transformation.Scale(xyres, xyres, xyres)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transformation)
    transformFilter.SetInputData(nuc_mesh)
    transformFilter.Update()
    nuc_mesh = transformFilter.GetOutput()

    
    
    #Get physical properties of both cell and nucleus
    CellMassProperties = vtk.vtkMassProperties()
    CellMassProperties.SetInputData(cell_mesh)
    NucMassProperties = vtk.vtkMassProperties()
    NucMassProperties.SetInputData(nuc_mesh)

    #get cell major, minor, and mini axes
    cell_coords = numpy_support.vtk_to_numpy(cell_mesh.GetPoints().GetData())
    cov = np.cov(cell_coords.T)
    cell_evals, cell_evecs = np.linalg.eig(cov)
    cell_sort_indices = np.argsort(cell_evals)[::-1]
    rotationthing = R.align_vectors(np.array([[0,0,1],[0,1,0],[1,0,0]]), cell_evecs.T)
    cell_coords = rotationthing[0].apply(cell_coords)

    #get nucleus major, minor, and mini axes
    nuc_coords = numpy_support.vtk_to_numpy(nuc_mesh.GetPoints().GetData())
    cov = np.cov(nuc_coords.T)
    nuc_evals, nuc_evecs = np.linalg.eig(cov)
    nuc_sort_indices = np.argsort(nuc_evals)[::-1]
    rotationthing = R.align_vectors(np.array([[0,0,1],[0,1,0],[1,0,0]]), nuc_evecs.T)
    nuc_coords = rotationthing[0].apply(nuc_coords)

    
    
    #Shape stats dict
    Shape_Stats = {'cell': cell_name,
                   'Euler_angles_X': euler_angles[0],
                   'Euler_angles_Y':euler_angles[1],
                   'Euler_angles_Z':euler_angles[2],
                  'Cell_Centroid_X': centroid_mem[0],
                  'Cell_Centroid_Y': centroid_mem[1],
                  'Cell_Centroid_Z': centroid_mem[2],
                   'Cell_Volume': CellMassProperties.GetVolume(),
                   'Cell_SurfaceArea': CellMassProperties.GetSurfaceArea(),
                   'Cell_MajorAxis': np.max(cell_coords[:,2])-np.min(cell_coords[:,2]),
                   'Cell_MajorAxis_Vec_X': cell_evecs[:, cell_sort_indices[0]][0],
                   'Cell_MajorAxis_Vec_Y': cell_evecs[:, cell_sort_indices[0]][1],
                   'Cell_MajorAxis_Vec_Z': cell_evecs[:, cell_sort_indices[0]][2],
                   'Cell_MinorAxis': np.max(cell_coords[:,1])-np.min(cell_coords[:,1]),
                   'Cell_MinorAxis_Vec_X': cell_evecs[:, cell_sort_indices[1]][0],
                   'Cell_MinorAxis_Vec_Y': cell_evecs[:, cell_sort_indices[1]][1],
                   'Cell_MinorAxis_Vec_Z': cell_evecs[:, cell_sort_indices[1]][2],
                   'Cell_MiniAxis': np.max(cell_coords[:,0])-np.min(cell_coords[:,0]),
                   'Cell_MiniAxis_Vec_X': cell_evecs[:, cell_sort_indices[2]][0],
                   'Cell_MiniAxis_Vec_Y': cell_evecs[:, cell_sort_indices[2]][1],
                   'Cell_MiniAxis_Vec_Z': cell_evecs[:, cell_sort_indices[2]][2],
                   'Nucleus_Centroid_X': centroid_nuc[0],
                   'Nucleus_Centroid_Y': centroid_nuc[1],
                   'Nucleus_Centroid_Z': centroid_nuc[2],
                   'Nucleus_Volume': NucMassProperties.GetVolume(),
                   'Nucleus_SurfaceArea': NucMassProperties.GetSurfaceArea(),
                   'Nucleus_MajorAxis': np.max(nuc_coords[:,2])-np.min(nuc_coords[:,2]),
                   'Nucleus_MajorAxis_Vec_X': nuc_evecs[:, nuc_sort_indices[0]][0],
                   'Nucleus_MajorAxis_Vec_Y': nuc_evecs[:, nuc_sort_indices[0]][1],
                   'Nucleus_MajorAxis_Vec_Z': nuc_evecs[:, nuc_sort_indices[0]][2],
                   'Nucleus_MinorAxis': np.max(nuc_coords[:,1])-np.min(nuc_coords[:,1]),
                   'Nucleus_MinorAxis_Vec_X': nuc_evecs[:, nuc_sort_indices[1]][0],
                   'Nucleus_MinorAxis_Vec_Y': nuc_evecs[:, nuc_sort_indices[1]][1],
                   'Nucleus_MinorAxis_Vec_Z': nuc_evecs[:, nuc_sort_indices[1]][2],
                   'Nucleus_MiniAxis': np.max(nuc_coords[:,0])-np.min(nuc_coords[:,0]),
                   'Nucleus_MiniAxis_Vec_X': nuc_evecs[:, nuc_sort_indices[2]][0],
                   'Nucleus_MiniAxis_Vec_Y': nuc_evecs[:, nuc_sort_indices[2]][1],
                   'Nucleus_MiniAxis_Vec_Z': nuc_evecs[:, nuc_sort_indices[2]][2]
                    }

    
    return Shape_Stats, coeffs_mem, coeffs_nuc, exceptions_list



def shcoeffs_and_PILR_nonuc(
        impath: str,
        direct: str,
        xyres: float,
        zstep: float,
        str_name: str,
        exceptions_list: list,
        normal_rotation_method: str,
        l_order: int,
        nisos: int,
        pilr_method: str,
        sigma: float = 0,
        align_method: str = 'None',
        ):

    """
        Parameters
        ----------
        impath : str
            Input image path. Expected to have shape CZYX. Channel order must be Membrane, Nucleus, Structure
        savedir : str
            Directory that will contain the Mesh and PILR folders
        xyres : float
            microns/pixel resolution of the image
        zstep : float
            Z step of the image
        normal_rotation_method : str
            "widest" is longest axis parallel to trajectory
        exceptions_list : List
            List to append names of images that cannot be represented well by SH
        str_name : str
            String detailing the name of the intracellular structure in the image
        l_order : int
            l order for SH transformation
        
        Returns
        -------
        coeffs_dict : dict
            Dictionary with the spherical harmonics coefficients and the mean square
            error between input and its parametrization
        grid_rec : ndarray
            Parametric grid representing sh parametrization
        image_ : ndarray
            Input image after pre-processing (lcc calculation, smooth and binarization).
        mesh : vtkPolyData
            Polydata representation of image_.
        grid_down : ndarray
            Parametric grid representing input object.
        transform : tuple of floats
            (xc, yc, zc, angle) if alignment_2d is True or
            (xc, yc, zc) if alignment_2d is False. (xc, yc, zc) are the coordinates
            of the shape centroid after alignment; angle is the angle used to align
            the image

        Other parameters
        ----------------
        sigma : float, optional
            The degree of smooth to be applied to the input image, default is 0 (no
            smooth)
        compute_lcc : bool, optional
            Whether to compute the largest connected component before appliying the
            spherical harmonic parametrization, default is True. Set compute_lcc to
            False in case you are sure the input image contains a single connected
            component. It is crucial that parametrization is calculated on a single
            connected component object.
        alignment_2d : bool
            Wheather the image should be aligned in 2d. Default is True.
        make_unique : bool
            Set true to make sure the alignment rotation is unique. 
            """
    
    #get cell name from impath
    impath = direct + 'processed_images/' + impath
    print(impath)
    cell_name = impath.split('/')[-1].split('_segmented')[0]
    #read image
    im = TiffReader(impath)
    
    #read euler angles for alignment
    infopath = direct + 'processed_data/' + cell_name + '_cell_info.csv'
    #if align_method is a numpy array, use that as the vector to align to
    if type(align_method) == np.ndarray:
        vec = align_method.copy()
        #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        xaxis = np.array([[1,0,0], [0,1,0], [0,0,1]]).astype('float64')
        upnorm = np.cross(vec,[1,0,0])
        # if upnorm[2]<0:
        #     upnorm = upnorm.copy() * -1
        sidenorm = np.cross(vec,upnorm)
        current_vec = np.stack((vec, sidenorm, upnorm), axis = 0)
        # current_vec = np.concatenate((current_vec,[5*vec]), axis = 0)
        rotationthing = R.align_vectors(xaxis, current_vec)
        #below is actual rotation matrix if needed
        #rot_mat = rotationthing[0].as_matrix()
        rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)
        euler_angles = np.array([rotthing_euler[0], rotthing_euler[1], rotthing_euler[2]])
    elif align_method == 'trajectory':
        info = pd.read_csv(infopath, index_col=0)
        vec = np.array([info.Trajectory_X[0], info.Trajectory_Y[0], info.Trajectory_Z[0]])
        #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        xaxis = np.array([[1,0,0], [0,1,0], [0,0,1]]).astype('float64')
        upnorm = np.cross(vec,[1,0,0])
        # if upnorm[2]<0:
        #     upnorm = upnorm.copy() * -1
        sidenorm = np.cross(vec,upnorm)
        current_vec = np.stack((vec, sidenorm, upnorm), axis = 0)
        # current_vec = np.concatenate((current_vec,[5*vec]), axis = 0)
        rotationthing = R.align_vectors(xaxis, current_vec)
        #below is actual rotation matrix if needed
        #rot_mat = rotationthing[0].as_matrix()
        rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)
        euler_angles = np.array([rotthing_euler[0], rotthing_euler[1], rotthing_euler[2]])
        
    if len(im.shape)>3:
        if im.shape[0]>2:
            ci = im.data[0,:,:,:]
            si = im.data[1:,:,:,:]
        else:
            ci = im.data[0,:,:,:]
            si = im.data[1,:,:,:][np.newaxis]
    else:
        ci = im.data
        
    #get cell mesh and coeffs
    (coeffs_mem, grid_rec, widestangle, exceptions_list), (image_, cell_mesh, grid, centroid_mem) = get_shcoeffs_mod(        
        image = ci,
        img_name= cell_name,
        lmax = l_order,
        xyres = xyres/xyres, #use pixels and not microns because I'll need pixel dimensions for PILRs
        zstep = zstep/xyres,
        Euler_Angles = euler_angles,
        exceptions_list = exceptions_list,
        sigma = sigma,
        normal_rotation_method = normal_rotation_method,
        )
    
    
    #get reconstruction errors both ways
    cell_recon, grid_recon = shtools_mod.get_reconstruction_from_coeffs(np.array(list(coeffs_mem.values())).reshape(2,l_order+1,l_order+1))
    #get average nearest distance from original mesh to reconstruction
    tree = KDTree(numpy_support.vtk_to_numpy(cell_mesh.GetPoints().GetData()))
    d, idx = tree.query(numpy_support.vtk_to_numpy(cell_recon.GetPoints().GetData()))
    OriginaltoReconError = np.mean(d)
    #get average nearest distance from reconstruction to original mesh
    tree = KDTree(numpy_support.vtk_to_numpy(cell_recon.GetPoints().GetData()))
    d, idx = tree.query(numpy_support.vtk_to_numpy(cell_mesh.GetPoints().GetData()))
    RecontoOriginalError = np.mean(d)
    
    #make sure nucleus is handled as segmented
    if str_name == 'nucleus':
        pilr_method = 'threshold'
    
    
    if ('si' in locals()) and (si.max()>0):
        #create inner sphere
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(0.0, 0.0, 0.0)
        sphereSource.SetRadius(nisos[0]/2)
        # Make the surface smooth.
        sphereSource.SetPhiResolution(100)
        sphereSource.SetThetaResolution(100)
        sphereSource.Update()
        spherepoly = sphereSource.GetOutput()
        
        
        (sphere_coeffs, grid_rec), (grid_down) = get_shcoeffs_mesh(
                spherepoly,
                lmax= l_order)

            
        
        images_to_probe = []
        if pilr_method == 'threshold':
            #set levels strings for structure thrsholds
            if len(si)>1:
                levels = ['low','mid','high']
            else:
                levels = ['mid']
            #for each structure threshold try to get a PILR
            for n, s in enumerate(si):
    
                #provide escape for cells with no signal in the "structure channel"
                if np.max(s) == 0:
                    #get voxelized intracellular structure image
                    img, origin = cytoparam_mod.voxelize_meshes([cell_mesh])
                
                    #get rotated segmentented str signal alone
                    strimg = img.copy()
                    strimg[strimg>0]=255
                
                else:
                    #get structure mesh
                    str_mesh, _, cent = shtools_mod.get_mesh_from_image(
                        image = s,
                        translate_to_origin=False,
                        lcc = False,
                        center = np.array(centroid_mem)[0]
                        )
                    #euler rotation and scaling
                    str_mesh = shtools_mod.rotate_and_scale_mesh(str_mesh,
                                                      euler_angles,
                                                      np.array([xyres, xyres, zstep])/xyres)
                    #widest angle rotation
                    str_mesh = shtools_mod.rotate_and_scale_mesh(str_mesh,
                                                      rotations = np.array([widestangle,0,0]))
                    #adjust the structure center to the final position of the cell after rotation
                    coords = numpy_support.vtk_to_numpy(str_mesh.GetPoints().GetData())
                    coords -= np.array(centroid_mem)[1]
                    str_mesh = shtools_mod.update_mesh_points(str_mesh, coords[:, 0], coords[:, 1], coords[:, 2])
                
                    #get voxelized intracellular structure image
                    img, origin = cytoparam_mod.voxelize_meshes([cell_mesh,str_mesh])
                    
                    #get rotated segmentented str signal alone
                    strimg = img.copy()
                    strimg[strimg<2]=0
                    strimg[strimg>0]=255
                    images_to_probe.append([str_name,strimg])
                    
                    #scale structure mesh
                    #set transform and apply
                    meshf = direct + 'meshes/'
                    transformation = vtk.vtkTransform()
                    transformation.Scale(xyres, xyres, xyres)
                    transformFilter = vtk.vtkTransformPolyDataFilter()
                    transformFilter.SetTransform(transformation)
                    transformFilter.SetInputData(str_mesh)
                    transformFilter.Update()
                    str_mesh = transformFilter.GetOutput()
                    #save str mesh
                    writer = vtk.vtkXMLPolyDataWriter()
                    writer.SetFileName(meshf + cell_name + f'_str_mesh_{levels[n]}.vtp')
                    writer.SetInputData(str_mesh)
                    writer.Write()
                    
            #########parameterize cell
            aicstif = cytoparam_mod.cellular_mapping(
                coeffs_mem = coeffs_mem,
                centroid_mem = abs(origin)[0],
                coeffs_nuc = sphere_coeffs,
                centroid_nuc = abs(origin)[0],
                nisos = nisos,
                images_to_probe = images_to_probe,
                )
            #Save PILR
            pilrf = direct+'PILRs/'
            if os.path.exists(pilrf+cell_name+'_PILR.ome.tiff'):
                os.remove(pilrf+cell_name+'_PILR.ome.tiff')
            OmeTiffWriter.save(aicstif.get_image_data('CZYX', S=0, T=0), pilrf+cell_name+'_PILR.ome.tiff', dim_order='CZYX', channel_names=aicstif.channel_names)
            
        elif pilr_method == 'raw':
            ######### translate coordinates to membrane centroid
            #open the raw data
            rawpath = impath.split('_segmented')[0] + '_raw.tiff'
            #read image
            raw = TiffReader(rawpath).data
            memseg = np.where(ci>0)
            intensities = np.tile(raw[1][memseg],3)
            #add some half points
            memsegmore = []
            for m in memseg:
                memsegmore.append(np.concatenate((m,m-0.25,m+0.25)))
            memcent = np.mean(memsegmore, axis = 1)
            centcoords = [memsegmore[i]-m for i, m in enumerate(memcent)]
            ########## rotate coordinates
            #first rotate toward trajectory (coords are flipped from zyx to xyz)
            rotcoords = rotationthing[0].apply(np.flip(np.array(centcoords).T, axis = 1))
            #then do width rotation (coords are flipped back to zyx from xyz)
            widrot = R.from_euler('xyz',np.array([widestangle,0,0]), degrees = True)
            widrotcoords = np.flip(widrot.apply(rotcoords), axis = 1)
            ######### move coordinates to origin as 0,0,0 (size of image plus pad 1)
            rotimg, origin = cytoparam_mod.voxelize_meshes([cell_mesh])
            rotseg = np.where(rotimg>0)
            rotcent = np.mean(rotseg, axis = 1)
            zerocoords = np.subtract(widrotcoords, -rotcent)#origin[::-1])
            ######### turn coordinates into int
            zerointcoords = np.round(zerocoords).astype(np.int16)
            ######### apply scalars to new array
            strimg = np.zeros(rotimg.shape)
            #combine coords with intensities
            coordint = np.hstack((zerointcoords,intensities.reshape(-1,1)))
            for z in range(strimg.shape[-3]):
                zcoords = coordint[np.where(coordint[:,0] == z)]
                for y in np.unique(zcoords[:,1]):
                    ycoords = zcoords[np.where(zcoords[:,1] == y)]
                    for x in np.unique(ycoords[:,2]):
                        xcoords = ycoords[np.where(ycoords[:,2] == x)]
                        strimg[z,y,x] = np.mean(xcoords[:,-1])
            ####### normalize the strimg
            strimg = strimg-intensities.min()
            strimg = strimg/strimg.max()
            strimg[strimg<0] = 0
            images_to_probe.append([str_name,strimg])
    
            #########parameterize cell
            aicstif = cytoparam_mod.cellular_mapping(
                coeffs_mem = coeffs_mem,
                centroid_mem = abs(origin)[0],
                coeffs_nuc = sphere_coeffs,
                centroid_nuc = abs(origin)[0],
                nisos = nisos,
                images_to_probe = images_to_probe,
                )
                  
            #Save PILR
            pilrf = direct+'PILRs/'
            if os.path.exists(pilrf+cell_name+'_PILR.ome.tiff'):
                os.remove(pilrf+cell_name+'_PILR.ome.tiff')
            OmeTiffWriter.save(aicstif.get_image_data('CZYX', S=0, T=0), pilrf+cell_name+'_PILR.ome.tiff', dim_order='CZYX', channel_names=aicstif.channel_names)
            
        
    #now that PILR has been made,
    #scale cell and nuc meshes so that I can take some shape stats
    #set transform and apply to cell
    transformation = vtk.vtkTransform()
    transformation.Scale(xyres, xyres, xyres)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transformation)
    transformFilter.SetInputData(cell_mesh)
    transformFilter.Update()
    cell_mesh = transformFilter.GetOutput()
    
    
    
    
    # remove file if it already exists
    meshf = direct + 'meshes/'
    #save cell mesh
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(meshf + cell_name + '_cell_mesh.vtp')
    writer.SetInputData(cell_mesh)
    writer.Write()
    
    
    #Get physical properties of cell
    CellMassProperties = vtk.vtkMassProperties()
    CellMassProperties.SetInputData(cell_mesh)
    
    
    
    #get cell major, minor, and mini axes using the segmented image
    cell_coords = numpy_support.vtk_to_numpy(cell_mesh.GetPoints().GetData())
    ##### measure length of cell along the trajectory axis
    trajlen = np.max(cell_coords[:,0])-np.min(cell_coords[:,0])
    trajwid = np.max(cell_coords[:,1])-np.min(cell_coords[:,1])
    #remove duplicate coordinates
    duplicates = pd.DataFrame(cell_coords).duplicated().to_numpy()
    mask = np.ones(len(cell_coords), dtype=bool)
    mask[duplicates] = False
    cell_coords = cell_coords[mask,:]
    #get covariance matrix and find eigenvalues and vectors
    cov = np.cov(cell_coords.T)
    cell_evals, cell_evecs = np.linalg.eig(cov)
    #make sure that the eigenvalues and vectors are in the order of highest to lowest
    idx = np.argsort(cell_evals)[::-1]
    cell_evals = cell_evals[idx]
    cell_evecs = cell_evecs[:,idx]
    #rotate the cell coordinates to align the major axis with the x, the minor axis to the y and the "mini" axis to the z
    rotationthing = R.align_vectors(np.array([[1,0,0],[0,1,0]]), cell_evecs.T[:2,:])
    cell_coords = rotationthing[0].apply(cell_coords)
    
    
    #measure the volume of the cell in the x domain
    FrontVolume = measure_volume_half(cell_mesh, 'x')
    RightVolume = measure_volume_half(cell_mesh, 'y')
    TopVolume = measure_volume_half(cell_mesh, 'z')
    

    
    
    #Shape stats dict
    Shape_Stats = {'cell': cell_name,
                   'Euler_angles_X': euler_angles[0],
                   'Euler_angles_Y':euler_angles[1],
                   'Euler_angles_Z':euler_angles[2],
                   'Width_Rotation_Angle': widestangle,
                  'Cell_Centroid_X': centroid_mem[0][0],
                  'Cell_Centroid_Y': centroid_mem[0][1],
                  'Cell_Centroid_Z': centroid_mem[0][2],
                   'Cell_Volume': CellMassProperties.GetVolume(),
                    'Cell_Volume_Front': FrontVolume,
                    'Cell_Volume_Right': RightVolume,
                    'Cell_Volume_Top': TopVolume,
                   'Cell_SurfaceArea': CellMassProperties.GetSurfaceArea(),
                   'Cell_MajorAxis': np.max(cell_coords[:,0])-np.min(cell_coords[:,0]),
                   'Cell_MajorAxis_Vec_X': cell_evecs[0,0],
                   'Cell_MajorAxis_Vec_Y': cell_evecs[1,0],
                   'Cell_MajorAxis_Vec_Z': cell_evecs[2,0],
                   'Cell_MinorAxis': np.max(cell_coords[:,1])-np.min(cell_coords[:,1]),
                   'Cell_MinorAxis_Vec_X': cell_evecs[0,1],
                   'Cell_MinorAxis_Vec_Y': cell_evecs[1,1],
                   'Cell_MinorAxis_Vec_Z': cell_evecs[2,1],
                   'Cell_MiniAxis': np.max(cell_coords[:,2])-np.min(cell_coords[:,2]),
                   'Cell_MiniAxis_Vec_X': cell_evecs[0,2],
                   'Cell_MiniAxis_Vec_Y': cell_evecs[1,2],
                   'Cell_MiniAxis_Vec_Z': cell_evecs[2,2],
                   'OriginaltoReconError': OriginaltoReconError,
                   'RecontoOriginalError': RecontoOriginalError,
                   'LengthAlongTrajectory': trajlen,
                   'WidthAlongTrajectory': trajwid
                    }


    return Shape_Stats, coeffs_mem, exceptions_list


