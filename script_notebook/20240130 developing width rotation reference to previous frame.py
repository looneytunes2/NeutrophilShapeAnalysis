# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:16:19 2024

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


from CustomFunctions import shparam_mod, shtools_mod, cytoparam_mod

from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from aicsimageio.readers.tiff_reader import TiffReader

from itertools import groupby
from operator import itemgetter

import multiprocessing

def collect_results(result):
    """Uses apply_async's callback to setup up a separate Queue for each process.
    This will allow us to collect the results from different threads."""
    results.append(result)


def find_normal_width_peaks(
        impath: str,
        xyres: float,
        zstep: float,
        sigma: float = 0,
        align_method: str = 'None',
        ):

    
    #get cell name from impath
    cell_name = impath.split('/')[-1].split('_segmented')[0]
    #read image
    im = TiffReader(impath)
    
    #read euler angles for alignment
    infopath = '/'.join(impath.split('/')[:-1]) + '/' + cell_name + '_cell_info.csv'
    #if align_method is a numpy array, use that as the vector to align to
    if type(align_method) == np.ndarray:
        vec = align_method.copy()
        #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        xaxis = np.array([[0,0,0],[1,0,0], [5,0,0]]).astype('float64')
        current_vec = np.stack(([0,0,0],vec), axis = 0)
        current_vec = np.concatenate((current_vec,[5*vec]), axis = 0)
        rotationthing = R.align_vectors(xaxis, current_vec)
        #below is actual rotation matrix if needed
        #rot_mat = rotationthing[0].as_matrix()
        rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)
        Euler_Angles = np.array([rotthing_euler[0], rotthing_euler[1], rotthing_euler[2]])
    elif align_method == 'trajectory':
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
    peaks, properties = signal.find_peaks(abs(both),prominence=0.13, width=70)
    angpeaks = np.concatenate((angles,angles))[peaks]
    tangpeaks = angpeaks.copy()
    tangpeaks[tangpeaks>180] -= 360

    return [cell_name, tangpeaks]





############ FIND WIDTH ROTATIONS THAT DEPEND ON PREVIOUS FRAMES TO LIMIT ROTATION FLIPPING ################


savedir = 'D:/Aaron/Data/Galvanotaxis_Confocal_40x_30C_10s/Processed_Data/'
    
xyres = 0.3394 #um / pixel 
zstep = 0.7 # um
align_method = 'trajectory'
sigma = 0

imlist = []
for o in os.listdir(savedir):
    if 'segmented' in o:
        cellid = o.split('_frame')[0]
        if cellid not in imlist:
            imlist.append(cellid)
            
allresults = []
for i in imlist:
    if __name__ ==  '__main__':
        results = []
        pool = multiprocessing.Pool(processes=60)
        for j in os.listdir(savedir):
            if (i+'_' in j) and ('segmented' in j):
                #get path to segmented image
                impath = savedir + j
    
                #put in the pool
                pool.apply_async(shparam_mod.find_normal_width_peaks, args = (
                    impath,
                    xyres,
                    zstep,
                    sigma,
                    align_method,
                    ),             
                    callback = collect_results)
    
        pool.close()
        pool.join()

    results.sort(key=lambda x: float(re.findall('(?<=frame_)\d*', x[0])[0]))
    tempframe = pd.DataFrame(results, columns = ['CellID','Width_Peaks'])
    tempframe['frame'] = [float(re.findall('(?<=frame_)\d*', x[0])[0]) for x in results]
    
    runs = list()
    #######https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
    for k, g in groupby(enumerate(tempframe['frame']), lambda ix: ix[0] - ix[1]):
        currentrun = list(map(itemgetter(1), g))
        list.append(runs, currentrun)

    
    #find the minima in each frame that are closest to the minimum chosen in the last frame
    #aka the one that results in the least amount of consecutive rotation
    allmins = []
    for xx in runs:
        runframe = tempframe[tempframe.frame.isin(xx)]
        for y, wp in enumerate(runframe.Width_Peaks.to_list()):
            if y == 0:
                allmins.append(wp[np.argmin(abs(wp))])
            else:
                allmins.append(wp[np.argmin(abs(wp-(allmins[-1])))])
    
    #add all mins to tempframe
    tempframe['Closest_minimums'] = allmins
    
    allresults.append(tempframe)
    

#save the shape metrics dataframe
bigdf = pd.concat(allresults)
bigdf.to_csv(datadir + 'Shape_Metrics.csv')






