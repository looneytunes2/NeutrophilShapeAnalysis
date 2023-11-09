# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 12:34:46 2022

@author: Aaron
"""


import math
import os
import pandas as pd
import numpy as np
from scipy import interpolate, ndimage, signal
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance
from aicsimageio.writers import OmeTiffWriter

# function for core algorithm
from aicssegmentation.core.utils import hole_filling, topology_preserving_thinning
from aicssegmentation.core.vessel import filament_2d_wrapper, filament_3d_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_slice_by_slice, image_smoothing_gaussian_3d
from skimage.morphology import remove_small_objects     # function for post-processing (size filter)
from aicssegmentation.core.MO_threshold import MO

from aicsshparam import shtools

from itertools import groupby
from operator import itemgetter
import skimage.measure
from skimage import transform
from skimage.filters import threshold_otsu, threshold_triangle

import vtk
from vtk.util import numpy_support

def znomore(number, im_temp_whole,):
    if number > im_temp_whole.shape[-3]:
        number = im_temp_whole.shape[-3]
    if number < 0:
        number = 0
    return number

def xynomore(number, im_temp_whole,):
    if number > im_temp_whole.shape[-2]:
        number = im_temp_whole.shape[-2]
    if number < 0:
        number = 0
    return number

def xzexpand(img, size, ordr, z):
    ex = transform.resize(img, size, order=ordr, preserve_range=True)
    return ex


def twodholefill(thresh, hole_min, hole_max):
    YZ = thresh.swapaxes(0,2)
    YZ_fill = hole_filling(YZ, hole_min, hole_max, fill_2d=True)
    XZ = thresh.swapaxes(0,1)
    XZ_fill = hole_filling(XZ, hole_min, hole_max, fill_2d=True)
    both_fill = np.logical_or(XZ_fill.swapaxes(1, 0), YZ_fill.swapaxes(2,0))
    filled = np.logical_or(thresh, both_fill)
    return filled



def segment_caax(img):
    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [1, 9]
    gaussian_smoothing_sigma = 2
    ################################
    # intensity normalization
    struct_img = intensity_normalization(img, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_slice_by_slice(struct_img, sigma=gaussian_smoothing_sigma)
    
    # step 1: Masked-Object (MO) Thresholding
    # thresh_img, object_for_debug = MO(structure_img_smooth[:,:,:], global_thresh_method='tri', local_adjust = 1.5, object_minArea=1200, return_object=True)
    thresh_img = structure_img_smooth > threshold_triangle(structure_img_smooth)
    thresh_img = remove_small_objects(thresh_img, min_size=1200)
    
    # fill in the holes
    hole_max = 1000
    hole_min = 1
    thresh_img_fill = twodholefill(thresh_img, hole_min, hole_max)
    # Step 2: Perform topology-preserving thinning
    # thin_dist_preserve = 0.5
    # thin_dist = 2
    # bw_thin = topology_preserving_thinning(thresh_img_fill>0, thin_dist_preserve, thin_dist)
    
    # structure_img_smooth = edge_preserving_smoothing_3d(struct_img)
    ################################
    ## PARAMETERS for this step ##
    f2_param = [[2, 0.1]]
    ################################

    fil_img = filament_2d_wrapper(structure_img_smooth, f2_param)

    # fill in the holes
    hole_max = 1000
    hole_min = 1
    fil_img_fill = hole_filling(fil_img, hole_min, hole_max) 

    ################################
    ## PARAMETERS for this step ##
    minArea = 6000
    ################################
    #combine the two segmentations
    seg = thresh_img_fill + fil_img_fill
    # create segmentation mask               
    seg = remove_small_objects(seg>0, min_size=minArea, connectivity=1, in_place=False)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255
    return seg
    
def segment_nucleus(img):
    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [1, 11]
    gaussian_smoothing_sigma = 1.5
    ################################
    # intensity normalization
    struct_img = intensity_normalization(img, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter slice by slice 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    
    # step 1: Masked-Object (MO) Thresholding
    thresh_img, object_for_debug = MO(structure_img_smooth[:,:,:], global_thresh_method='tri', object_minArea=1200, local_adjust = 1.5, return_object=True)

    # fill in the holes
    hole_max = 1000
    hole_min = 2
    seg = hole_filling(thresh_img, hole_min, hole_max)
    
    
    ################################
    ## PARAMETERS for this step ##
    # set minimum area to just less that largest object
    im_labeled, n_labels = skimage.measure.label(
                              seg, background=0, return_num=True)
    if n_labels > 1:
        im_props = skimage.measure.regionprops(im_labeled)
        tempdf = pd.DataFrame([])
        for count, prop in enumerate(im_props):
            area = prop.area
            tempdata = {'cell':count, 'area':area}
            tempdf = tempdf.append(tempdata, ignore_index=True)
        minArea = tempdf.sort_values(by='area', ascending=False).loc[0,['area']][0]-1
    else:
        minArea = 6000
    ################################
    #combine the two segmentations
    # create segmentation mask               
    seg = remove_small_objects(seg>0, min_size=minArea, connectivity=1, in_place=False)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255
    return seg



def segment_caax_final(img):
    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [1, 9]
    gaussian_smoothing_sigma = 1
    ################################
    # intensity normalization
    struct_img = intensity_normalization(img, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_slice_by_slice(struct_img, sigma=gaussian_smoothing_sigma)
    
    #otsu threshold
    thresh_img = structure_img_smooth > threshold_otsu(structure_img_smooth)
    
    # fill in the holes
    hole_max = 1000
    hole_min = 1
    thresh_img_fill = twodholefill(thresh_img, hole_min, hole_max)
    # Step 2: Perform topology-preserving thinning
    # thin_dist_preserve = 0.5
    # thin_dist = 2
    # bw_thin = topology_preserving_thinning(thresh_img_fill>0, thin_dist_preserve, thin_dist)
    
    # structure_img_smooth = edge_preserving_smoothing_3d(struct_img)
    ################################
    ## PARAMETERS for this step ##
    f3_param = [[1.5, 0.2]]
    ################################
    
    fil_img = filament_3d_wrapper(structure_img_smooth, f3_param)

    # fill in the holes
    hole_max = 1000
    hole_min = 1
    fil_img_fill = hole_filling(fil_img, hole_min, hole_max) 

    ################################
    ## PARAMETERS for this step ##
    #minArea = 6000
    ################################
    #combine the two segmentations
    seg = thresh_img_fill + fil_img_fill
    
    # set minimum area to just less that largest object
    im_labeled, n_labels = skimage.measure.label(
                              seg, background=0, return_num=True)
    if n_labels > 1:
        im_props = skimage.measure.regionprops(im_labeled)
        tempdf = pd.DataFrame([])
        for count, prop in enumerate(im_props):
            area = prop.area
            tempdata = {'cell':count, 'area':area}
            tempdf = tempdf.append(tempdata, ignore_index=True)
        minArea = int(tempdf.area.max()-2)
    else:
        minArea = 15000
    
    # create segmentation mask               
    seg = remove_small_objects(im_labeled, min_size=minArea, connectivity=1, in_place=False)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255
    return seg




def segment_cells_func2(
    cell,
    im_temp_whole,
    savedir,
    image_name,
    xy_buffer: int,
    z_buffer: int,
    xyres: float,
    zstep: float,
    framerange: int,
    framemethod: str,
    norm_rot: str,
    nuc_mask: bool
):

    """
        Parameters
        ----------
        image : ndarray
            Input image. 
        zstep : float
            Z step of the image
        xyres : float
            microns/pixel resolution
        framemethod : str
            "first" gives first available frame
            "fast" gives fastest cell frame
            "fast nucleus" gives fastest nuclear frame
        Returns
        -------

        Other parameters
        ----------------

        Notes
        -----


    """
    
    temp = pd.DataFrame()
    runs = list()
    #######https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
    for k, g in groupby(enumerate(cell['frame']), lambda ix: ix[0] - ix[1]):
        currentrun = list(map(itemgetter(1), g))
        list.append(runs, currentrun)
    whichframes = np.array(max(enumerate(runs), key = lambda tup: len(tup[1]))[1], dtype=int)
    pullrows = [cell.loc[cell['frame'] == i] for i in whichframes]
    temp = temp.append(pullrows, ignore_index=True)
    temp = temp.reset_index(drop = True)
    #decide which frame of the available frames to take the shape of
    if framemethod == 'fast':
        fastest = temp['dist'][1:].idxmax()
        #if the fastest frame is the first or last frame find second fastest
        if fastest == temp.index[0] or fastest == temp.index[len(temp)-1]:
            fastest2 = temp.drop(fastest).reset_index(drop = True)['dist'][1:].idxmax()
            temp = temp.iloc[int(fastest2-1):int(fastest2+2),:]
            temp = temp.reset_index(drop = True)
        else:
            temp = temp.iloc[int(fastest-1):int(fastest+2),:]
            temp = temp.reset_index(drop = True)
    # if framemethod == 'fast nucleus':
    #     fastest = temp['nuc_dist'][1:].idxmax()
    #     #if the fastest frame is the first or last frame find second fastest
    #     if fastest == temp.index[0] or fastest == temp.index[len(temp)-1]:
    #         fastest2 = temp.drop(fastest).reset_index(drop = True)['nuc_dist'][1:].idxmax()
    #         temp = temp.iloc[int(fastest2-1):int(fastest2+2),:]
    #         temp = temp.reset_index(drop = True)
    #     else:
    #         temp = temp.iloc[int(fastest-1):int(fastest+2),:]
    #         temp = temp.reset_index(drop = True)
    
    #get first three available frames
    if framemethod == 'first':
        temp = temp[0:framerange]
        temp = temp.reset_index(drop = True)
    if framemethod == 'all':
        pass

    #segment the cell channel and get centroid
    df = pd.DataFrame()
    for t, row in temp.iterrows():

        xmincrop = int(xynomore(row.x_min-xy_buffer, im_temp_whole))
        ymincrop = int(xynomore(row.y_min-xy_buffer, im_temp_whole))
        zmincrop = int(znomore(row.z_min-z_buffer, im_temp_whole))
        
        
        temp_im = im_temp_whole[int(row.frame),
            0,
            zmincrop:int(znomore(row.z_max+z_buffer, im_temp_whole,)+1),
            ymincrop:int(xynomore(row.y_max+xy_buffer, im_temp_whole,)+1),
            xmincrop:int(xynomore(row.x_max+xy_buffer, im_temp_whole,)+1)]
        
        seg_caax = segment_caax(temp_im)
        
        
        # Label binary image; backward kwarg says value in im_bw to consider backgr.
        im_labeled, n_labels = skimage.measure.label(
                                  seg_caax, background=0, return_num=True)

        # Get properties
        cent = skimage.measure.regionprops(im_labeled)[0].centroid
        data = {'image': image_name,
                'cell': row.cell,
                'frame':row.frame,
                'x':(cent[2]+xmincrop)*xyres, 
                'y':(cent[1]+ymincrop)*xyres, 
                'z':(cent[0]+zmincrop)*zstep,
                'cropx':cent[2], 
                'cropy':cent[1], 
                'cropz':cent[0]}
        df = df.append(data, ignore_index=True)
        
    #add new distances from cropped image
    #############find distance travelled##################
    longdistmatrix = distance.pdist(df[['x','y','z']])
    shortdistmatrix = distance.squareform(longdistmatrix)
    shortdistmatrix = np.array(shortdistmatrix)
    dist = pd.Series([])
    for count, i in enumerate(shortdistmatrix):
        if count == 0:
            tmp = pd.Series([0])
            dist = dist.append(tmp, ignore_index=True)
        else:
            tmp = pd.Series(shortdistmatrix[count,count-1])
            dist = dist.append(tmp, ignore_index=True)
    df['dist'] = dist
        
    
    # ############# SEGMENT THE NUCLEUS ####################
    # seg_nucleus = np.zeros(temp_im[:,1,:,:,:].shape).astype(bool)
    # for t in range(temp_im.shape[0]):
        
    #set the k order for interpolation to the max possible
    if len(df)<6:
        kay = len(df)-1
    else:
        kay = 5
    
    #     seg_nucleus[t,:,:,:] = segment_nucleus(temp_im[t,1,:,:,:])
    pos = df[['x','y','z']]
    if bool(pos[pos.duplicated()].index.tolist()):
        ######### FIND CELL TRAJECTORY AND EULER ANGLES ################
        # if there is duplicate positions
        dups = pos[pos.duplicated()].index.tolist()
        pos_drop = pos.drop(dups, axis = 0)
        #get trajectories without the duplicates
        tck, u = interpolate.splprep(pos_drop.to_numpy().T, k=kay)
        yderv = interpolate.splev(u,tck,der=1)
        yderv2 = interpolate.splev(u,tck,der=2)
        traj = np.vstack(yderv).T
        #re-insert duplicate row that was dropped
        traj = np.insert(traj, dups, traj[[d-1 for d in dups],:], axis=0)
        acc = np.vstack(yderv2).T
        #re-insert duplicate row that was dropped
        acc = np.insert(acc, dups, [0,0,0], axis=0)
    else:
        ######### FIND CELL TRAJECTORY AND EULER ANGLES ################
        #no duplicate positions
        #interpolate and get tangent at midpoint
        tck, u = interpolate.splprep(pos.to_numpy().T, k=kay)
        yderv = interpolate.splev(u,tck,der=0)
        traj = np.vstack(yderv).T
        
        
    for j in range(1,traj.shape[0]-1):
        
        temp_df = df.iloc[j,:]
        row = temp.iloc[list(temp.frame == temp_df.frame)].squeeze()
        
        vec = traj[j,:]

        #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        xaxis = np.array([[0,0,0],[1,0,0], [5,0,0]]).astype('float64')
        current_vec = np.stack(([0,0,0],vec), axis = 0)
        current_vec = np.concatenate((current_vec,[5*vec]), axis = 0)
        rotationthing = R.align_vectors(xaxis, current_vec)
        #below is actual rotation matrix if needed
        #rot_mat = rotationthing[0].as_matrix()
        rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)
        euler_angles = np.array([rotthing_euler[0], rotthing_euler[1], rotthing_euler[2]])
    

        
        #get current cropped image
        xmincrop = int(xynomore(row.x_min-xy_buffer, im_temp_whole))
        ymincrop = int(xynomore(row.y_min-xy_buffer, im_temp_whole))
        zmincrop = int(znomore(row.z_min-z_buffer, im_temp_whole))
        raw_img = im_temp_whole[int(row.frame),
                    :,
                    zmincrop:int(znomore(row.z_max+z_buffer, im_temp_whole,)+1),
                    ymincrop:int(xynomore(row.y_max+xy_buffer, im_temp_whole,)+1),
                    xmincrop:int(xynomore(row.x_max+xy_buffer, im_temp_whole,)+1)]
        
        
        #expand z to match xy resolution as closely as possible
        size = (round(raw_img.shape[-3]*zstep/xyres), raw_img.shape[-1])
        adj_zstep = raw_img.shape[-3]*zstep/round(raw_img.shape[-3]*zstep/xyres)
        raw_img_ex = np.zeros((raw_img.shape[0],size[0], raw_img.shape[-2], raw_img.shape[-1]))
        for c in range(raw_img.shape[0]):
            for i in range(raw_img.shape[-2]):
                raw_img_ex[c,:,i,:] = xzexpand(raw_img[c,:,i,:], size, 1, i)
        
        
        # segment cropped image
        seg_rimg = np.zeros(raw_img_ex.shape)
        seg_rimg[0,:,:,:] = segment_caax_final(raw_img_ex[0,:,:,:])
        seg_rimg[1,:,:,:] = segment_nucleus(raw_img_ex[1,:,:,:])
        
        
        
        #get surface meshes
        cell_mesh, image_, centroid = shtools.get_mesh_from_image(image=seg_rimg[0,:,:,:], sigma=2)
        nuc_mesh, image_, centroid = shtools.get_mesh_from_image(image=seg_rimg[1,:,:,:], sigma=2)
        
        
        
        #rotate and scale the cell mesh
        transformation = vtk.vtkTransform()
        #rotate the shape
        transformation.RotateWXYZ(euler_angles[2], 0, 0, 1)
        transformation.RotateWXYZ(euler_angles[0], 1, 0, 0)
        #set scale to actual image scale
        transformation.Scale(xyres, xyres, adj_zstep)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(cell_mesh)
        transformFilter.Update()
        cell_mesh = transformFilter.GetOutput()
        
        #rotate and scale the nuclear mesh
        transformation = vtk.vtkTransform()
        #rotate the shape
        transformation.RotateWXYZ(euler_angles[2], 0, 0, 1)
        transformation.RotateWXYZ(euler_angles[0], 1, 0, 0)
        #set scale to actual image scale
        transformation.Scale(xyres, xyres, zstep)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(nuc_mesh)
        transformFilter.Update()
        nuc_mesh = transformFilter.GetOutput()
        
        
        
        if norm_rot == 'none':
            pass
        
        if norm_rot == 'widest':
            widestangle = 0
            #rotate around the x axis until you find the widest distance in y
            angles = np.arange(0,360,0.5)
            widths = np.empty((0,3))
            for a in angles:
                
                transformation = vtk.vtkTransform()
                #rotate the shape
                transformation.RotateWXYZ(a, 1, 0, 0)
                transformFilter = vtk.vtkTransformPolyDataFilter()
                transformFilter.SetTransform(transformation)
                transformFilter.SetInputData(cell_mesh)
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
            
            
            #rotate the cell
            transformation = vtk.vtkTransform()
            transformation.RotateWXYZ(widestangle, 1, 0, 0)
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(transformation)
            transformFilter.SetInputData(cell_mesh)
            transformFilter.Update()
            cell_mesh = transformFilter.GetOutput()
            
            #rotate the nucleus
            transformation = vtk.vtkTransform()
            transformation.RotateWXYZ(widestangle, 1, 0, 0)
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(transformation)
            transformFilter.SetInputData(nuc_mesh)
            transformFilter.Update()
            nuc_mesh = transformFilter.GetOutput()
            
            
        #save the info about cell
        info = pd.DataFrame({'cell': temp_df.cell,
                'frame': temp_df.frame,
                'dist': temp_df.dist,
                'trajectory_vec': vec,
                'euler_angles': euler_angles,
                'normal_rotation_angle': widestangle
                })
        #file name template
        cell_name = image_name + '_cell_' + str(int(cell['cell'].iloc[0])) + '_frame_' + str(int(info.frame.iloc[0]))
        info.to_csv(savedir + cell_name + '_cell_info.csv')
        
        

        #crop the segmented image
        im_labeled, n_labels = skimage.measure.label(
                                  seg_rimg[0,:,:,:], background=0, return_num=True)
        im_props = skimage.measure.regionprops(im_labeled)
        bound = im_props[0].bbox
        seg_rimg = seg_rimg[:,
                            bound[0]-5:bound[3]+5,
                            bound[1]-5:bound[4]+5,
                            bound[2]-5:bound[5]+5]
        
        
        #SAVE SEGMENTED IMAGE
        out=seg_rimg.astype(np.uint8)
        out[out>0]=255
        
        #include nucleus mask if wanted
        if nuc_mask == True:
            out[0,:,:,:] = out[0,:,:,:] + out[1,:,:,:]
            # out = np.array(out, dtype = 'bool')
            # fill in the holes
            hole_max = 2000
            hole_min = 1
            out[0,:,:,:] = hole_filling(out[0,:,:,:], hole_min, hole_max)
            out=out.astype(np.uint8)
            out[out>0]=255
        
        # remove file if it already exists
        seg_file = savedir + cell_name + '_segmented.tiff'
        if os.path.exists(seg_file):
            os.remove(seg_file)
        writer = OmeTiffWriter(seg_file)
        writer.save(out, dimension_order = "CZYX")
        
        
        #CROP THE RAW IMAGE
        rimg = raw_img[:,
                    bound[0]-5:bound[3]+5,
                    bound[1]-5:bound[4]+5,
                    bound[2]-5:bound[5]+5]       
        #SAVE THE RAW IMAGE
        raw_file = savedir + cell_name + '_raw.tiff'
        if os.path.exists(raw_file):
            os.remove(raw_file)
        writer = OmeTiffWriter(raw_file)
        writer.save(rimg, dimension_order = "CZYX")
        
        
        #save cell and nuclear meshes
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(savedir + 'Meshes/' + cell_name + 'Cell_Mesh.vtp')
        writer.SetInputData(cell_mesh)
        writer.Write()
        
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(savedir + 'Meshes/' + cell_name + 'Nuclear_Mesh.vtp')
        writer.SetInputData(nuc_mesh)
        writer.Write()
        
        
        
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

        #get original centroids
        cz, cy, cx = np.where(out[0,:,:,:])
        nz, ny, nx = np.where(out[1,:,:,:])

        #get intensity features
        cell_int = rimg[0,:,:,:]
        # nucleus_int = im_temp_whole_int[1,:,:,:]
        mem_feat = get_intensity_features(cell_int, out[0,:,:,:])
        mem_keylist = [x for x in list(mem_feat) if not x.endswith('lcc')]
        
        #Append shape metrics to dataframe
        temp_shape = {'cell': cell_name,
                      'Cell_Centroid_X': cx.mean(),
                      'Cell_Centroid_Y': cy.mean(),
                      'Cell_Centroid_Z': cz.mean(),
                       'Cell_Volume': CellMassProperties.GetVolume(),
                       'Cell_SurfaceArea': CellMassProperties.GetSurfaceArea(),
                       'Cell_NormalizedShapeIndex': CellMassProperties.GetNormalizedShapeIndex(),
                       'Cell_MajorAxis': np.max(nuc_coords[:,2])-np.min(nuc_coords[:,2]),
                       'Cell_MajorAxis_Vec': cell_evecs[:, cell_sort_indices[0]],
                       'Cell_MinorAxis': np.max(nuc_coords[:,1])-np.min(nuc_coords[:,1]),
                       'Cell_MinorAxis_Vec': cell_evecs[:, cell_sort_indices[1]],
                       'Cell_MiniAxis': np.max(nuc_coords[:,0])-np.min(nuc_coords[:,0]),
                       'Cell_MiniAxis_Vec': cell_evecs[:, cell_sort_indices[2]],
                       'Cell_'+mem_keylist[0]: mem_feat[mem_keylist[0]],
                       'Cell_'+mem_keylist[1]: mem_feat[mem_keylist[1]],
                       'Cell_'+mem_keylist[2]: mem_feat[mem_keylist[2]],
                       'Cell_'+mem_keylist[3]: mem_feat[mem_keylist[3]],
                       'Cell_'+mem_keylist[4]: mem_feat[mem_keylist[4]],
                       'Cell_'+mem_keylist[5]: mem_feat[mem_keylist[5]],
                       'Nucleus_Centroid_X': nx.mean(),
                       'Nucleus_Centroid_Y': ny.mean(),
                       'Nucleus_Centroid_Z': nz.mean(),
                       'Nucleus_Volume': NucMassProperties.GetVolume(),
                       'Nucleus_SurfaceArea': NucMassProperties.GetSurfaceArea(),
                       'Nucleus_NormalizedShapeIndex': NucMassProperties.GetNormalizedShapeIndex(),
                       'Nucleus_MajorAxis': np.max(nuc_coords[:,2])-np.min(nuc_coords[:,2]),
                       'Nucleus_MajorAxis_Vec': nuc_evecs[:, nuc_sort_indices[0]],
                       'Nucleus_MinorAxis': np.max(nuc_coords[:,1])-np.min(nuc_coords[:,1]),
                       'Nucleus_MinorAxis_Vec': nuc_evecs[:, nuc_sort_indices[1]],
                       'Nucleus_MiniAxis': np.max(nuc_coords[:,0])-np.min(nuc_coords[:,0]),
                       'Nucleus_MiniAxis_Vec': nuc_evecs[:, nuc_sort_indices[2]]
                        }
        Shape_Stats = Shape_Stats.append(temp_shape, ignore_index=True)
    
    
    
#save the coefficient dataframes
cell_coeffs.to_csv(savedir + 'Cell_Coefficients.csv')
nuclear_coeffs.to_csv(savedir + 'Nuclear_Coefficients.csv')
#save the shape metrics dataframe
Shape_Stats.to_csv(savedir + 'Shape_Metrics.csv')

#save list of cells that don't have centroid in shape
pd.Series([filelist_fl[x] for x in cell_outlist]).to_csv(savedir + 'ListToExclude.csv')
        
        