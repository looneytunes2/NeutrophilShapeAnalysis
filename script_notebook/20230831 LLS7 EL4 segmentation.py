# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:35:12 2023

@author: Aaron
"""

# # some_file.py
# import sys
# # insert at 1, 0 is the script path (or '' in REPL)
# import_path = 'C:/Users/Aaron/Documents/PythonScripts/CustomFunctions'
# sys.path.insert(1, import_path)

# import ExpandImage

import math
import os
import pandas as pd
import numpy as np
from scipy import ndimage, signal
from scipy.spatial.transform import Rotation as R
from aicsimageio.writers import OmeTiffWriter
from aicsimageio.readers.tiff_reader import TiffReader

# function for core algorithm
from aicssegmentation.core.utils import hole_filling, topology_preserving_thinning
from aicssegmentation.core.vessel import filament_3d_wrapper, filament_2d_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_3d
from skimage.morphology import remove_small_objects     # function for post-processing (size filter)
from CustomFunctions.track_functions import MO


from CustomFunctions import shparam_mod, shtools_mod



import skimage.measure
from skimage import transform


import vtk
from vtk.util import numpy_support


from aicsimageio.readers.tiff_reader import TiffReader

dirr = 'E:/Aaron/LLS7 EL-4 cells/Cells/'
imname = '20230824_EL4GFPCAAX_sirDNA-01_processed-Cell3.tif'
im = TiffReader(dirr + imname)

img = im.data[0,:,:,:]

################################
## PARAMETERS for this step ##
intensity_scaling_param = [0]
gaussian_smoothing_sigma = 2
################################
# intensity normalization
struct_img = intensity_normalization(img, scaling_param=intensity_scaling_param)
# smoothing with 2d gaussian filter 
structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)

# step 1: Masked-Object (MO) Thresholding
thresh_img, object_for_debug = MO(structure_img_smooth, global_thresh_method='tri', object_minArea=3000, local_adjust = 0.95, return_object=True)

OmeTiffWriter.save(thresh_img.astype(np.uint8),'C:/Users/Aaron/Documents/Python Scripts/temp/tcellthresh.ome.tiff')


f2_param = [[1,0.3]]
################################

fil_img = filament_2d_wrapper(structure_img_smooth, f2_param)

OmeTiffWriter.save(fil_img.astype(np.uint8),'C:/Users/Aaron/Documents/Python Scripts/temp/tcellfil.ome.tiff')

seg = thresh_img + fil_img

OmeTiffWriter.save(seg.astype(np.uint8),'C:/Users/Aaron/Documents/Python Scripts/temp/tcellboth.ome.tiff')

# fill in the holes
hole_max = 8000
hole_min = 1
seg = hole_filling(seg, hole_min, hole_max)#, fill_2d=True) 


# Step 2: Perform topology-preserving thinning
thin_dist_preserve = 1.5
thin_dist = 1
seg = topology_preserving_thinning(seg, thin_dist_preserve, thin_dist)


OmeTiffWriter.save(seg.astype(np.uint8),'C:/Users/Aaron/Documents/Python Scripts/temp/tcellbothfill.ome.tiff')



def segment_caax_el4(img):
    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [0]
    gaussian_smoothing_sigma = 2
    ################################
    # intensity normalization
    struct_img = intensity_normalization(img, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)

    # step 1: Masked-Object (MO) Thresholding
    thresh_img, object_for_debug = MO(structure_img_smooth[:,:,:], global_thresh_method='tri', object_minArea=3000, local_adjust = 0.95, return_object=True)

    # structure_img_smooth = edge_preserving_smoothing_3d(struct_img)
    ################################
    ## PARAMETERS for this step ##
    f2_param = [[1,0.3]]
    ################################

    fil_img = filament_2d_wrapper(structure_img_smooth, f2_param)


    seg = thresh_img + fil_img

    # fill in the holes
    hole_max = 8000
    hole_min = 1
    seg = hole_filling(seg, hole_min, hole_max) 


    # Step 2: Perform topology-preserving thinning
    thin_dist_preserve = 1.5
    thin_dist = 1
    seg = topology_preserving_thinning(seg, thin_dist_preserve, thin_dist)

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
        # create segmentation mask               
        seg = remove_small_objects(im_labeled, min_size=minArea, connectivity=1, in_place=False)
        seg = seg.astype(np.uint8)
        seg[seg > 0] = 255
    else:
        seg = seg.astype(np.uint8)
        seg[seg > 0] = 255

    return seg




def nomore(dim, number, im_shape,):
    if dim == 'x':
        dim = int(1)
    if dim == 'y':
        dim = int(2)
    if dim == 'z':
        dim = int(3)
    if number > im_shape[-dim]:
        number = im_shape[-dim]
    if number < 0:
        number = 0
    return number

def xzexpand(img, size, ordr, z):
    ex = transform.resize(img, size, order=ordr, preserve_range=True)
    return ex

    
def segment_nucleus(img):
    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [1, 4]
    gaussian_smoothing_sigma = 2
    ################################
    # intensity normalization
    struct_img = intensity_normalization(img, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter slice by slice 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    
    # step 1: Masked-Object (MO) Thresholding
    thresh_img, object_for_debug = MO(structure_img_smooth[:,:,:], global_thresh_method='tri', object_minArea=1200, return_object=True)
    
    # fill in the holes
    hole_max = 6000
    hole_min = 2
    seg = hole_filling(thresh_img, hole_min, hole_max)
    
    
    # Step 2: Perform topology-preserving thinning
    thin_dist_preserve = 2
    thin_dist = 1
    seg = topology_preserving_thinning(seg, thin_dist_preserve, thin_dist)
    
    
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
    intensity_scaling_param = [1, 6]
    gaussian_smoothing_sigma = 2
    ################################
    # intensity normalization
    struct_img = intensity_normalization(img, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    
    # step 1: Masked-Object (MO) Thresholding
    thresh_img, object_for_debug = MO(structure_img_smooth[:,:,:], global_thresh_method='tri', object_minArea=3000, return_object=True)

    # fill in the holes
    hole_max = 15000
    hole_min = 1
    thresh_img_fill = hole_filling(thresh_img, hole_min, hole_max)
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
    hole_max = 8000
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



def segment_caax_norot(img):
    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [1, 6]
    gaussian_smoothing_sigma = 2
    ################################
    # intensity normalization
    struct_img = intensity_normalization(img, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)

    # step 1: Masked-Object (MO) Thresholding
    thresh_img, object_for_debug = MO(structure_img_smooth[:,:,:], global_thresh_method='tri', object_minArea=3000, local_adjust = 0.92, return_object=True)

    # # fill in the holes
    # hole_max = 15000
    # hole_min = 1
    # thresh_img_fill = hole_filling(thresh_img, hole_min, hole_max)


    # structure_img_smooth = edge_preserving_smoothing_3d(struct_img)
    ################################
    ## PARAMETERS for this step ##
    # f3_param = [[1.5, 0.25]]
    # f2_param = [[1,0.22],[2, 0.17]]
    f2_param = [[1,0.3]]
    ################################

    fil_img = filament_2d_wrapper(structure_img_smooth, f2_param)


    seg = thresh_img + fil_img

    # fill in the holes
    hole_max = 8000
    hole_min = 1
    seg = hole_filling(seg, hole_min, hole_max) 


    # Step 2: Perform topology-preserving thinning
    thin_dist_preserve = 1.5
    thin_dist = 1
    seg = topology_preserving_thinning(seg, thin_dist_preserve, thin_dist)



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
        # create segmentation mask               
        seg = remove_small_objects(im_labeled, min_size=minArea, connectivity=1, in_place=False)
        seg = seg.astype(np.uint8)
        seg[seg > 0] = 255
    else:
        seg = seg.astype(np.uint8)
        seg[seg > 0] = 255

    return seg



def segment_cytoGFP(img):
    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [1, 3]
    gaussian_smoothing_sigma = 1
    ################################
    # intensity normalization
    struct_img = intensity_normalization(img, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    
    # step 1: Masked-Object (MO) Thresholding
    thresh_img = MO(structure_img_smooth, global_thresh_method='tri', local_adjust=0.93 ,object_minArea=20)
    
    
    # set minimum area to just less that largest object
    im_labeled, n_labels = skimage.measure.label(
                              thresh_img, background=0, return_num=True)
    if n_labels > 1:
        im_props = skimage.measure.regionprops(im_labeled)
        tempdf = pd.DataFrame([])
        for count, prop in enumerate(im_props):
            area = prop.area
            tempdata = {'cell':count, 'area':area}
            tempdf = tempdf.append(tempdata, ignore_index=True)
        minArea = int(tempdf.area.max()-2)
        # create segmentation mask               
        seg = remove_small_objects(im_labeled, min_size=minArea, connectivity=1, in_place=False)
        seg = seg.astype(np.uint8)
        seg[seg > 0] = 255
    else:
        seg = seg.astype(np.uint8)
        seg[seg > 0] = 255
    
    return seg


def segment_actinGFP_norot(img):
    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [0]
    gaussian_smoothing_sigma = 1.5
    ################################
    # intensity normalization
    struct_img = intensity_normalization(img, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    
    # step 1: Masked-Object (MO) Thresholding
    thresh_img = MO(structure_img_smooth[:,:,:], global_thresh_method='tri', local_adjust=1.2 ,object_minArea=20)
    
    
    seg = thresh_img.astype(np.uint8)
    seg[seg > 0] = 255
    
    return seg

def caax_noise_fill(caax_ch):
    #find the intensity peak of the noise
    hist = np.histogram(caax_ch[caax_ch>70], bins=list(range(0,round(np.max(caax_ch)),10)), range=(0,round(np.max(caax_ch))))
    peaks, properties = signal.find_peaks(hist[0],prominence=100000, width=0.5)
    #find positions in the image with values at or below noise level
    noise_positions = np.where(np.logical_and(caax_ch>properties['left_ips'][0]*10 , caax_ch<properties['right_ips'][0]*11))
    noise_sample = caax_ch[noise_positions[0],noise_positions[1],noise_positions[2]]
    noise_max = np.max(noise_sample)
    #get sample of noise from values
    r_fill = np.random.choice(noise_sample, len(caax_ch[caax_ch<np.max(noise_sample)]))
    return noise_max, r_fill

    
def nuc_noise_fill(nuc_ch):
    #find the intensity peak of the noise
    hist = np.histogram(nuc_ch[nuc_ch>50], bins=list(range(0,round(np.max(nuc_ch)),10)), range=(0,round(np.max(nuc_ch))))
    peaks, properties = signal.find_peaks(hist[0], prominence=100000,width=0.5)
    #find positions in the image with values at or below noise level
    noise_positions = np.where(np.logical_and(nuc_ch>properties['left_ips'][0]*10 , nuc_ch<properties['right_ips'][0]*11))
    noise_sample = nuc_ch[noise_positions[0],noise_positions[1],noise_positions[2]]
    noise_max = np.max(noise_sample)
    #get sample of noise from values
    r_fill = np.random.choice(noise_sample, len(nuc_ch[nuc_ch<np.max(noise_sample)]))
    return noise_max, r_fill
    

def segment_cells_alignchemgradient_iSIM(
    chem_vec,
    vec,
    temp_df,
    raw_img,
    image_name,
    savedir,
    cell,
    xyres,
    zstep,
    norm_rot,
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
    
    
    #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    xaxis = np.array([[0,0,0],[1,0,0], [5,0,0]]).astype('float64')
    current_vec = np.stack(([0,0,0],chem_vec), axis = 0)
    current_vec = np.concatenate((current_vec,[5*chem_vec]), axis = 0)
    rotationthing = R.align_vectors(xaxis, current_vec)
    #below is actual rotation matrix if needed
    #rot_mat = rotationthing[0].as_matrix()
    rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)
    euler_angles = pd.DataFrame({'x':[rotthing_euler[0]], 'y':[rotthing_euler[1]], 'z': [rotthing_euler[2]]})

    
    
    #translate the cell to the middle of the image based on the cell centroid
    trans = np.array([temp_df.cropx-raw_img.shape[-1]/2, 
                      temp_df.cropy-raw_img.shape[-2]/2, 
                      temp_df.cropz-raw_img.shape[-3]/2])
    #before translating, pad edges of image so that real image doesn't get cut off
    exp = math.ceil(np.max(abs(trans)))+3
    timg = np.pad(raw_img, ((0,0),(exp,exp),(exp,exp),(exp,exp)), 'constant', constant_values=(0))
    #actually translate the image
    timg = ndimage.shift(timg, [0, -trans[2], -trans[1], -trans[0]])
    
    
    #expand z to match xy resolution as closely as possible
    size = (round(timg.shape[-3]*zstep/xyres), timg.shape[-1])
    adj_zstep = timg.shape[-3]*zstep/round(timg.shape[-3]*zstep/xyres)
    timg_ex = np.zeros((timg.shape[0],size[0], timg.shape[-2], timg.shape[-1]))
    for c in range(timg.shape[0]):
        for i in range(timg.shape[-2]):
            timg_ex[c,:,i,:] = xzexpand(timg[c,:,i,:], size, 1, i)


    
    #rotate image around x-axis
    rimg = ndimage.rotate(timg_ex, -euler_angles.x[0], axes=(1,2))
    #rotate image around z-axis
    rimg = ndimage.rotate(rimg, -euler_angles.z[0], axes=(2,3))


    # norm_rot = 'none'
    if norm_rot == 'none':
    
        #fill in the expanded parts of the image with similar noise to the rest of the image
        noise_max, r_fill = caax_noise_fill(rimg[1,:,:,:])
        rimg[1,:,:,:][rimg[1,:,:,:]<noise_max] = r_fill
        
        
        #fill in the expanded parts of the image with similar noise to the rest of the image
        noise_max, r_fill = nuc_noise_fill(rimg[0,:,:,:])
        rimg[0,:,:,:][rimg[0,:,:,:]<noise_max] = r_fill
    
    if norm_rot == 'widest':
        #fill in the expanded parts of the image with similar noise to the rest of the image
        noise_max, r_fill = caax_noise_fill(rimg[1,:,:,:])
        rimg_copy = rimg.copy()
        rimg_copy[1,:,:,:][rimg_copy[1,:,:,:]<noise_max] = r_fill
        
        # get mesh to rotate for normal alignment
        temp_seg = segment_caax_final(rimg_copy[1,:,:,:])
        mesh, image_, centroid = shtools_mod.get_mesh_from_image(image=temp_seg, sigma=2)
        
        
        #rotate and scale mesh
        #worked from https://kitware.github.io/vtk-examples/site/Python/PolyData/AlignTwoPolyDatas/
        #rotate around z axis
        transformation = vtk.vtkTransform()
        #set scale to actual image scale
        transformation.Scale(xyres, xyres, adj_zstep)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(mesh)
        transformFilter.Update()
        mesh = transformFilter.GetOutput()
        
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
        
        
        #rotate image around x-axis so that the widest part of the cell is pointing towards the -y direction
        rimg = ndimage.rotate(rimg, widestangle, axes=(1,2))
    
        #fill in the expanded parts of the image with similar noise to the rest of the image
        noise_max, r_fill = caax_noise_fill(rimg[1,:,:,:])
        rimg[1,:,:,:][rimg[1,:,:,:]<noise_max] = r_fill
        
        
        #fill in the expanded parts of the image with similar noise to the rest of the image
        noise_max, r_fill = nuc_noise_fill(rimg[0,:,:,:])
        rimg[0,:,:,:][rimg[0,:,:,:]<noise_max] = r_fill
    
    #save the info about cell
    info = pd.DataFrame({'cell': temp_df.cell,
            'frame': temp_df.frame,
            'x': temp_df.x,
            'y': temp_df.y,
            'z': temp_df.z,
            'dist': temp_df.dist,
            'velocity_vec': vec,
            'deviation_angle':temp_df.deviation_angle,
            'euler_angles': rotthing_euler,
            'normal_rotation_angle': widestangle,
            'persistence': temp_df.persistence,
            'activity': temp_df.activity
            })
    info.to_csv(savedir + image_name + '_cell_' + str(int(cell['cell'].iloc[0])) + '_frame_' + str(int(info.frame.iloc[0])) + '_cell_info.csv')
    
    
    # segment rotated image
    seg_rimg = np.zeros(rimg.shape)
    seg_rimg[1,:,:,:] = segment_caax_final(rimg[1,:,:,:])
    seg_rimg[0,:,:,:] = segment_nucleus(rimg[0,:,:,:])
    
    
    #crop the segmented image
    im_labeled, n_labels = skimage.measure.label(
                              seg_rimg[1,:,:,:], background=0, return_num=True)
    im_props = skimage.measure.regionprops(im_labeled)
    bound = im_props[0].bbox
    seg_rimg = seg_rimg[:,
                        bound[0]-5:bound[3]+5,
                        bound[1]-5:bound[4]+5,
                        bound[2]-5:bound[5]+5]
    
    
    #save segmented image
    out=seg_rimg.astype(np.uint8)
    out[out>0]=255
    # remove file if it already exists
    im_file = savedir + image_name + '_cell_' + str(int(cell['cell'].iloc[0])) + '_frame_' + str(int(info.frame.iloc[0])) + '_segmented.tiff'
    if os.path.exists(im_file):
        os.remove(im_file)
    OmeTiffWriter.save(out, im_file, dimension_order = "CZYX")
    
    
    #crop the raw image
    rimg = rimg[:,
                bound[0]-5:bound[3]+5,
                bound[1]-5:bound[4]+5,
                bound[2]-5:bound[5]+5]       
    #save raw image
    raw_file = savedir + image_name + '_cell_' + str(int(cell['cell'].iloc[0])) + '_frame_' + str(int(info.frame.iloc[0])) + '_raw.tiff'
    if os.path.exists(raw_file):
        os.remove(raw_file)
    OmeTiffWriter.save(rimg, raw_file, dimension_order = "CZYX")




def segment_cells_aligntrajectory_iSIM(
    vec,
    temp_df,
    raw_img,
    image_name,
    savedir,
    cell,
    xyres,
    zstep,
    norm_rot,
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
    
    
    #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    xaxis = np.array([[0,0,0],[1,0,0], [5,0,0]]).astype('float64')
    current_vec = np.stack(([0,0,0],vec), axis = 0)
    current_vec = np.concatenate((current_vec,[5*vec]), axis = 0)
    rotationthing = R.align_vectors(xaxis, current_vec)
    #below is actual rotation matrix if needed
    #rot_mat = rotationthing[0].as_matrix()
    rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)
    euler_angles = pd.DataFrame({'x':[rotthing_euler[0]], 'y':[rotthing_euler[1]], 'z': [rotthing_euler[2]]})

    
    
    #translate the cell to the middle of the image based on the cell centroid
    trans = np.array([temp_df.cropx-raw_img.shape[-1]/2, 
                      temp_df.cropy-raw_img.shape[-2]/2, 
                      temp_df.cropz-raw_img.shape[-3]/2])
    #before translating, pad edges of image so that real image doesn't get cut off
    exp = math.ceil(np.max(abs(trans)))+3
    timg = np.pad(raw_img, ((0,0),(exp,exp),(exp,exp),(exp,exp)), 'constant', constant_values=(0))
    #actually translate the image
    timg = ndimage.shift(timg, [0, -trans[2], -trans[1], -trans[0]])
    
    
    #expand z to match xy resolution as closely as possible
    size = (round(timg.shape[-3]*zstep/xyres), timg.shape[-1])
    adj_zstep = timg.shape[-3]*zstep/round(timg.shape[-3]*zstep/xyres)
    timg_ex = np.zeros((timg.shape[0],size[0], timg.shape[-2], timg.shape[-1]))
    for c in range(timg.shape[0]):
        for i in range(timg.shape[-2]):
            timg_ex[c,:,i,:] = xzexpand(timg[c,:,i,:], size, 1, i)


    
    #rotate image around x-axis
    rimg = ndimage.rotate(timg_ex, -euler_angles.x[0], axes=(1,2))
    #rotate image around z-axis
    rimg = ndimage.rotate(rimg, -euler_angles.z[0], axes=(2,3))


    # norm_rot = 'none'
    if norm_rot == 'none':
    
        #fill in the expanded parts of the image with similar noise to the rest of the image
        noise_max, r_fill = caax_noise_fill(rimg[1,:,:,:])
        rimg[1,:,:,:][rimg[1,:,:,:]<noise_max] = r_fill
        
        
        #fill in the expanded parts of the image with similar noise to the rest of the image
        noise_max, r_fill = nuc_noise_fill(rimg[0,:,:,:])
        rimg[0,:,:,:][rimg[0,:,:,:]<noise_max] = r_fill
    
    if norm_rot == 'widest':
        #fill in the expanded parts of the image with similar noise to the rest of the image
        noise_max, r_fill = caax_noise_fill(rimg[1,:,:,:])
        rimg_copy = rimg.copy()
        rimg_copy[1,:,:,:][rimg_copy[1,:,:,:]<noise_max] = r_fill
        
        # get mesh to rotate for normal alignment
        temp_seg = segment_caax_final(rimg_copy[1,:,:,:])
        mesh, image_, centroid = shtools_mod.get_mesh_from_image(image=temp_seg, sigma=2)
        
        
        #rotate and scale mesh
        #worked from https://kitware.github.io/vtk-examples/site/Python/PolyData/AlignTwoPolyDatas/
        #rotate around z axis
        transformation = vtk.vtkTransform()
        #set scale to actual image scale
        transformation.Scale(xyres, xyres, adj_zstep)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(mesh)
        transformFilter.Update()
        mesh = transformFilter.GetOutput()
        
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
        
        
        #rotate image around x-axis so that the widest part of the cell is pointing towards the -y direction
        rimg = ndimage.rotate(rimg, widestangle, axes=(1,2))
    
        #fill in the expanded parts of the image with similar noise to the rest of the image
        noise_max, r_fill = caax_noise_fill(rimg[1,:,:,:])
        rimg[1,:,:,:][rimg[1,:,:,:]<noise_max] = r_fill
        
        
        #fill in the expanded parts of the image with similar noise to the rest of the image
        noise_max, r_fill = nuc_noise_fill(rimg[0,:,:,:])
        rimg[0,:,:,:][rimg[0,:,:,:]<noise_max] = r_fill
    
    #save the info about cell
    info = pd.DataFrame({'cell': temp_df.cell,
            'frame': temp_df.frame,
            'x': temp_df.x,
            'y': temp_df.y,
            'z': temp_df.z,
            'dist': temp_df.dist,
            'velocity_vec': vec,
            'deviation_angle':temp_df.deviation_angle,
            'euler_angles': rotthing_euler,
            'normal_rotation_angle': widestangle,
            'persistence': temp_df.persistence,
            'activity': temp_df.activity
            })
    info.to_csv(savedir + image_name + '_cell_' + str(int(cell['cell'].iloc[0])) + '_frame_' + str(int(info.frame.iloc[0])) + '_cell_info.csv')
    
    
    # segment rotated image
    seg_rimg = np.zeros(rimg.shape)
    seg_rimg[1,:,:,:] = segment_caax_final(rimg[1,:,:,:])
    seg_rimg[0,:,:,:] = segment_nucleus(rimg[0,:,:,:])
    
    
    #crop the segmented image
    im_labeled, n_labels = skimage.measure.label(
                              seg_rimg[1,:,:,:], background=0, return_num=True)
    im_props = skimage.measure.regionprops(im_labeled)
    bound = im_props[0].bbox
    seg_rimg = seg_rimg[:,
                        bound[0]-5:bound[3]+5,
                        bound[1]-5:bound[4]+5,
                        bound[2]-5:bound[5]+5]
    
    
    #save segmented image
    out=seg_rimg.astype(np.uint8)
    out[out>0]=255
    # remove file if it already exists
    im_file = savedir + image_name + '_cell_' + str(int(cell['cell'].iloc[0])) + '_frame_' + str(int(info.frame.iloc[0])) + '_segmented.tiff'
    if os.path.exists(im_file):
        os.remove(im_file)
    OmeTiffWriter.save(out, im_file, dimension_order = "CZYX")
    
    
    #crop the raw image
    rimg = rimg[:,
                bound[0]-5:bound[3]+5,
                bound[1]-5:bound[4]+5,
                bound[2]-5:bound[5]+5]       
    #save raw image
    raw_file = savedir + image_name + '_cell_' + str(int(cell['cell'].iloc[0])) + '_frame_' + str(int(info.frame.iloc[0])) + '_raw.tiff'
    if os.path.exists(raw_file):
        os.remove(raw_file)
    OmeTiffWriter.save(rimg, raw_file, dimension_order = "CZYX")
  
     
def segment_cells_noalign_iSIM(
    vec,
    temp_df,
    raw_img,
    image_name,
    savedir,
    cell,
    xyres,
    zstep,
    norm_rot,
):
    

    
    #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    xaxis = np.array([[0,0,0],[1,0,0], [5,0,0]]).astype('float64')
    current_vec = np.stack(([0,0,0],vec), axis = 0)
    current_vec = np.concatenate((current_vec,[5*vec]), axis = 0)
    rotationthing = R.align_vectors(xaxis, current_vec)
    #below is actual rotation matrix if needed
    #rot_mat = rotationthing[0].as_matrix()
    rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)

    

    # norm_rot = 'none'
    if norm_rot == 'none':
    
        simg = raw_img.copy()
    
    if norm_rot == 'widest':
        
        #translate the cell to the middle of the image based on the cell centroid
        trans = np.array([temp_df.cropx-raw_img.shape[-1]/2, 
                          temp_df.cropy-raw_img.shape[-2]/2, 
                          temp_df.cropz-raw_img.shape[-3]/2])
        #before translating, pad edges of image so that real image doesn't get cut off
        exp = math.ceil(np.max(abs(trans)))+3
        timg = np.pad(raw_img, ((0,0),(exp,exp),(exp,exp),(exp,exp)), 'constant', constant_values=(0))
        #actually translate the image
        timg = ndimage.shift(timg, [0, -trans[2], -trans[1], -trans[0]])
        
        
        #expand z to match xy resolution as closely as possible
        size = (round(timg.shape[-3]*zstep/xyres), timg.shape[-1])
        adj_zstep = timg.shape[-3]*zstep/round(timg.shape[-3]*zstep/xyres)
        timg_ex = np.zeros((timg.shape[0],size[0], timg.shape[-2], timg.shape[-1]))
        for c in range(timg.shape[0]):
            for i in range(timg.shape[-2]):
                timg_ex[c,:,i,:] = xzexpand(timg[c,:,i,:], size, 1, i)
        
        
        #fill in the expanded parts of the image with similar noise to the rest of the image
        noise_max, r_fill = caax_noise_fill(timg_ex[1,:,:,:])
        timg_ex_copy = timg_ex.copy()
        timg_ex_copy[1,:,:,:][timg_ex_copy[1,:,:,:]<noise_max] = r_fill
        
        # get mesh to rotate for normal alignment
        temp_seg = segment_caax_final(timg_ex_copy[1,:,:,:])
        mesh, image_, centroid = shtools_mod.get_mesh_from_image(image=temp_seg, sigma=2)
        
        
        #rotate and scale mesh
        #worked from https://kitware.github.io/vtk-examples/site/Python/PolyData/AlignTwoPolyDatas/
        #rotate around z axis
        transformation = vtk.vtkTransform()
        #set scale to actual image scale
        transformation.Scale(xyres, xyres, adj_zstep)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(mesh)
        transformFilter.Update()
        mesh = transformFilter.GetOutput()
        
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
        
        
        #rotate image around x-axis so that the widest part of the cell is pointing towards the -y direction
        simg = ndimage.rotate(timg_ex, widestangle, axes=(1,2))
    
        #fill in the expanded parts of the image with similar noise to the rest of the image
        noise_max, r_fill = caax_noise_fill(simg[1,:,:,:])
        simg[1,:,:,:][simg[1,:,:,:]<noise_max] = r_fill
        
        
        #fill in the expanded parts of the image with similar noise to the rest of the image
        noise_max, r_fill = nuc_noise_fill(simg[0,:,:,:])
        simg[0,:,:,:][simg[0,:,:,:]<noise_max] = r_fill
    
    #save the info about cell
    info = pd.DataFrame({'cell': temp_df.cell,
            'frame': temp_df.frame,
            'x': temp_df.x,
            'y': temp_df.y,
            'z': temp_df.z,
            'dist': temp_df.dist,
            'velocity_vec': vec,
            'deviation_angle':temp_df.deviation_angle,
            'euler_angles': rotthing_euler,
            'normal_rotation_angle': widestangle,
            'persistence': temp_df.persistence,
            'activity': temp_df.activity
            })
    info.to_csv(savedir + image_name + '_cell_' + str(int(cell['cell'].iloc[0])) + '_frame_' + str(int(info.frame.iloc[0])) + '_cell_info.csv')
    
    
    # segment rotated image
    seg = np.zeros(simg.shape)
    seg[1,:,:,:] = segment_caax_final(simg[1,:,:,:])
    seg[0,:,:,:] = segment_nucleus(simg[0,:,:,:])
    
    
    #crop the segmented image
    im_labeled, n_labels = skimage.measure.label(
                              seg[1,:,:,:], background=0, return_num=True)
    im_props = skimage.measure.regionprops(im_labeled)
    bound = im_props[0].bbox
    seg = seg[:,
                bound[0]-5:bound[3]+5,
                bound[1]-5:bound[4]+5,
                bound[2]-5:bound[5]+5]
    
    
    #save segmented image
    out=seg.astype(np.uint8)
    out[out>0]=255
    # remove file if it already exists
    im_file = savedir + image_name + '_cell_' + str(int(cell['cell'].iloc[0])) + '_frame_' + str(int(info.frame.iloc[0])) + '_segmented.tiff'
    if os.path.exists(im_file):
        os.remove(im_file)
    OmeTiffWriter.save(out, im_file, dimension_order = "CZYX")
    
    
    #crop the raw image
    simg = simg[:,
                bound[0]-5:bound[3]+5,
                bound[1]-5:bound[4]+5,
                bound[2]-5:bound[5]+5]       
    #save raw image
    raw_file = savedir + image_name + '_cell_' + str(int(cell['cell'].iloc[0])) + '_frame_' + str(int(info.frame.iloc[0])) + '_raw.tiff'
    if os.path.exists(raw_file):
        os.remove(raw_file)
    OmeTiffWriter.save(simg, raw_file, dimension_order = "CZYX")
    
    
        
def accurate_centroid(temp_im, row, image_name, xyres, zstep, xmincrop, ymincrop, zmincrop,):
    seg_caax = segment_caax_final(temp_im)


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
    return data
        
        
def get_intensity_features(img, seg):
    features = {}
    input_seg = seg.copy()
    input_seg = (input_seg>0).astype(np.uint8)
    input_seg_lcc = skimage.measure.label(input_seg)
    for mask, suffix in zip([input_seg, input_seg_lcc], ['', '_lcc']):
        values = img[mask>0].flatten()
        if values.size:
            features[f'intensity_mean{suffix}'] = values.mean()
            features[f'intensity_std{suffix}'] = values.std()
            features[f'intensity_1pct{suffix}'] = np.percentile(values, 1)
            features[f'intensity_99pct{suffix}'] = np.percentile(values, 99)
            features[f'intensity_max{suffix}'] = values.max()
            features[f'intensity_min{suffix}'] = values.min()
        else:
            features[f'intensity_mean{suffix}'] = np.nan
            features[f'intensity_std{suffix}'] = np.nan
            features[f'intensity_1pct{suffix}'] = np.nan
            features[f'intensity_99pct{suffix}'] = np.nan
            features[f'intensity_max{suffix}'] = np.nan
            features[f'intensity_min{suffix}'] = np.nan
    return features





def segment_cells_rotafter(
    raw_img,
    row,
    image_name,
    savedir,
    xyres: float,
    zstep: float,
    xmincrop, 
    ymincrop, 
    zmincrop,
    xmaxcrop, 
    ymaxcrop, 
    zmaxcrop,
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
        Returns
        -------

        Other parameters
        ----------------

        Notes
        -----


    """

    #file name template
    cell_name = image_name + f'_cell_{int(row.cell)}_frame_{int(row.frame)}'


    
    #expand z to match xy resolution as closely as possible
    size = [round(raw_img.shape[-3]*zstep/xyres)]+list(raw_img.shape[-2:])
    adj_zstep = raw_img.shape[-3]*zstep/round(raw_img.shape[-3]*zstep/xyres)
    raw_img_ex = np.zeros(([raw_img.shape[0]]+size))
    for c in range(raw_img.shape[0]):
        raw_img_ex[c,:,:,:] = transform.resize(raw_img[c,:,:,:],size, preserve_range=True)
    
    
    # segment cropped image
    seg_rimg = np.zeros(raw_img_ex.shape)
    seg_rimg[0,:,:,:] = segment_caax_norot(raw_img_ex[0,:,:,:])
    if 'actin' in cell_name:
        seg_rimg[1,:,:,:] = segment_actinGFP_norot(raw_img_ex[1,:,:,:])
        strr = 'actin'
    if 'Hoechst' in cell_name:
        seg_rimg[1,:,:,:] = segment_nucleus(raw_img_ex[1,:,:,:])
        strr = 'nucleus'
    if 'DNA' in cell_name:
        seg_rimg[1,:,:,:] = segment_nucleus(raw_img_ex[1,:,:,:])
        strr = 'nucleus'
    if 'GFP++' in cell_name:
        seg_rimg[1,:,:,:] = segment_cytoGFP(raw_img_ex[1,:,:,:])
        strr = 'cytoGFP'
    

    #ensure there is no structure segmentation outside the membrane segmentation
    # seg_rimg[1,:,:,:] = np.logical_and(seg_rimg[0,:,:,:], seg_rimg[1,:,:,:])



    #get intensity features
    # nucleus_int = im_temp_whole_int[1,:,:,:]
    mem_feat = get_intensity_features(raw_img_ex[0,:,:,:], seg_rimg[0,:,:,:])
    mem_keylist = [x for x in list(mem_feat) if not x.endswith('lcc')]
    str_feat = get_intensity_features(raw_img_ex[1,:,:,:], seg_rimg[1,:,:,:])
    str_keylist = [x for x in list(str_feat) if not x.endswith('lcc')]

    #crop the segmented image
    im_labeled, n_labels = skimage.measure.label(
                              seg_rimg[0,:,:,:], background=0, return_num=True)
    im_props = skimage.measure.regionprops(im_labeled)
    bound = im_props[0].bbox
    seg_rimg = seg_rimg[:,
                        max(0,bound[0]-1):min(seg_rimg.shape[-3],bound[3])+1,
                        max(0,bound[1]-1):min(seg_rimg.shape[-2],bound[4])+1,
                        max(0,bound[2]-1):min(seg_rimg.shape[-1],bound[5])+1]


    #get original centroids
    cent = im_props[0].centroid
    

    #SAVE SEGMENTED IMAGE
    out=seg_rimg.astype(np.uint8)
    out[out>0]=255
    
    
    # remove file if it already exists
    seg_file = savedir + cell_name + '_segmented.tiff'
    if os.path.exists(seg_file):
        os.remove(seg_file)
    OmeTiffWriter.save(out, seg_file, dimension_order = "CZYX")
    
    
   
    #SAVE THE RAW IMAGE
    raw_file = savedir + cell_name + '_raw.tiff'
    if os.path.exists(raw_file):
        os.remove(raw_file)
    OmeTiffWriter.save(raw_img_ex, raw_file, dimension_order = "CZYX")
    
    
    
    #save the info about cell
    data = {'image': image_name,
            'cell': cell_name,
            'structure': strr,
            'frame': row.frame,
            'x':(cent[2]+xmincrop)*xyres, 
            'y':(cent[1]+ymincrop)*xyres, 
            'z':(cent[0]+zmincrop)*zstep,
            'cropx (pixels)':cent[2], 
            'cropy (pixels)':cent[1], 
            'cropz (pixels)':cent[0],
            'Cell_'+mem_keylist[0]: mem_feat[mem_keylist[0]],
            'Cell_'+mem_keylist[1]: mem_feat[mem_keylist[1]],
            'Cell_'+mem_keylist[2]: mem_feat[mem_keylist[2]],
            'Cell_'+mem_keylist[3]: mem_feat[mem_keylist[3]],
            'Cell_'+mem_keylist[4]: mem_feat[mem_keylist[4]],
            'Cell_'+mem_keylist[5]: mem_feat[mem_keylist[5]],
            'Structure_'+str_keylist[0]: str_feat[str_keylist[0]],
            'Structure_'+str_keylist[1]: str_feat[str_keylist[1]],
            'Structure_'+str_keylist[2]: str_feat[str_keylist[2]],
            'Structure_'+str_keylist[3]: str_feat[str_keylist[3]],
            'Structure_'+str_keylist[4]: str_feat[str_keylist[4]],
            'Structure_'+str_keylist[5]: str_feat[str_keylist[5]],
            }
    
    return data





def segment_cells_rotafter_VV(
    cellstack,
    structstack,
    row,
    image_name,
    savedir,
    xyres,
    zstep,
    xmincrop, 
    ymincrop, 
    zmincrop,
    xmaxcrop, 
    ymaxcrop, 
    zmaxcrop,
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
        Returns
        -------

        Other parameters
        ----------------

        Notes
        -----


    """

    cell = TiffReader(cellstack)
    struct = TiffReader(structstack)
    
    #construct cropped raw image
    raw_img = np.stack((cell.data[zmincrop:zmaxcrop,
                            ymincrop:ymaxcrop,
                            xmincrop:xmaxcrop]
                        ,struct.data[zmincrop:zmaxcrop,
                            ymincrop:ymaxcrop,
                            xmincrop:xmaxcrop]))
    
    #file name template
    cell_name = image_name + f'_cell_{int(row.cell)}_frame_{int(row.frame)+1}'


    
    #expand z to match xy resolution as closely as possible
    size = [round(raw_img.shape[-3]*zstep/xyres)]+list(raw_img.shape[-2:])
    adj_zstep = raw_img.shape[-3]*zstep/round(raw_img.shape[-3]*zstep/xyres)
    raw_img_ex = np.zeros(([raw_img.shape[0]]+size))
    for c in range(raw_img.shape[0]):
        raw_img_ex[c,:,:,:] = transform.resize(raw_img[c,:,:,:],size, preserve_range=True)
    
    
    # segment cropped image
    seg_rimg = np.zeros(raw_img_ex.shape)
    seg_rimg[0,:,:,:] = segment_caax_norot(raw_img_ex[0,:,:,:])
    if 'actin' in cell_name:
        seg_rimg[1,:,:,:] = segment_actinGFP_norot(raw_img_ex[1,:,:,:])
        strr = 'actin'
    if 'Hoechst' in cell_name:
        seg_rimg[1,:,:,:] = segment_nucleus(raw_img_ex[1,:,:,:])
        strr = 'nucleus'
    if 'DNA' in cell_name:
        seg_rimg[1,:,:,:] = segment_nucleus(raw_img_ex[1,:,:,:])
        strr = 'nucleus'
    if 'myosin' in cell_name:
        seg_rimg[1,:,:,:] = segment_actinGFP_norot(raw_img_ex[1,:,:,:])
        strr = 'myosin'
    

    #ensure there is no structure segmentation outside the membrane segmentation
    # seg_rimg[1,:,:,:] = np.logical_and(seg_rimg[0,:,:,:], seg_rimg[1,:,:,:])



    #get intensity features
    # nucleus_int = im_temp_whole_int[1,:,:,:]
    mem_feat = get_intensity_features(raw_img_ex[0,:,:,:], seg_rimg[0,:,:,:])
    mem_keylist = [x for x in list(mem_feat) if not x.endswith('lcc')]
    str_feat = get_intensity_features(raw_img_ex[1,:,:,:], seg_rimg[1,:,:,:])
    str_keylist = [x for x in list(str_feat) if not x.endswith('lcc')]

    #crop the segmented image
    im_labeled, n_labels = skimage.measure.label(
                              seg_rimg[0,:,:,:], background=0, return_num=True)
    im_props = skimage.measure.regionprops(im_labeled)


    #get original centroids
    cent = im_props[0].centroid
    

    #SAVE SEGMENTED IMAGE
    out=seg_rimg.astype(np.uint8)
    out[out>0]=255
    
    
    # remove file if it already exists
    seg_file = savedir + cell_name + '_segmented.tiff'
    if os.path.exists(seg_file):
        os.remove(seg_file)
    OmeTiffWriter.save(out, seg_file, dimension_order = "CZYX")
    
    
   
    #SAVE THE RAW IMAGE
    raw_file = savedir + cell_name + '_raw.tiff'
    if os.path.exists(raw_file):
        os.remove(raw_file)
    OmeTiffWriter.save(raw_img_ex, raw_file, dimension_order = "CZYX")
    
    
    
    #save the info about cell
    data = {'image': image_name,
            'cell': cell_name,
            'structure': strr,
            'frame': row.frame,
            'x':(cent[2]+xmincrop)*xyres, 
            'y':(cent[1]+ymincrop)*xyres, 
            'z':(cent[0]+zmincrop)*zstep,
            'cropx (pixels)':cent[2], 
            'cropy (pixels)':cent[1], 
            'cropz (pixels)':cent[0],
            'Cell_'+mem_keylist[0]: mem_feat[mem_keylist[0]],
            'Cell_'+mem_keylist[1]: mem_feat[mem_keylist[1]],
            'Cell_'+mem_keylist[2]: mem_feat[mem_keylist[2]],
            'Cell_'+mem_keylist[3]: mem_feat[mem_keylist[3]],
            'Cell_'+mem_keylist[4]: mem_feat[mem_keylist[4]],
            'Cell_'+mem_keylist[5]: mem_feat[mem_keylist[5]],
            'Structure_'+str_keylist[0]: str_feat[str_keylist[0]],
            'Structure_'+str_keylist[1]: str_feat[str_keylist[1]],
            'Structure_'+str_keylist[2]: str_feat[str_keylist[2]],
            'Structure_'+str_keylist[3]: str_feat[str_keylist[3]],
            'Structure_'+str_keylist[4]: str_feat[str_keylist[4]],
            'Structure_'+str_keylist[5]: str_feat[str_keylist[5]],
            }
    
    return data




def segment_cells_rotafter_VV_chem(
    raw_dir,
    ind_dir,
    row,
    image_name,
    savedir,
    xyres,
    zstep,
    xmincrop, 
    ymincrop, 
    zmincrop,
    xmaxcrop, 
    ymaxcrop, 
    zmaxcrop,
):
    #get image index
    ind = pd.read_csv(ind_dir+'Chemotaxis Image Index.csv', index_col = 0)
    rand = ind[ind.Combined == image_name].Random.values[0]
    chem = ind[ind.Combined == image_name].Directed.values[0]

    if row.frame<41:
        direct = raw_dir + rand + '/'
        #read zstacks for the specific time point and channel separately
        cellstack = direct + [x for x in os.listdir(direct) if f'Reflected_t{int(row.frame)+1}' in x][0]
        structstack = direct + [x for x in os.listdir(direct) if f'Trans_t{int(row.frame)+1}' in x][0]
    else:
        direct = raw_dir + chem + '/'
        time = row.frame+1-41
        #read zstacks for the specific time point and channel separately
        cellstack = direct + [x for x in os.listdir(direct) if f'Reflected_t{int(2-(time)%2)}_l{int(math.floor((time+1)/2))}' in x][0]
        structstack = direct + [x for x in os.listdir(direct) if f'Trans_t{int(2-(time)%2)}_l{int(math.floor((time+1)/2))}' in x][0]
    

    data = segment_cells_rotafter_VV(
        cellstack,
        structstack,
        row,
        image_name,
        savedir,
        xyres,
        zstep,
        xmincrop, 
        ymincrop, 
        zmincrop,
        xmaxcrop, 
        ymaxcrop, 
        zmaxcrop,
        )
    # print('returning ' + str(data['cell']), flush=True)
    return data




def meshes_and_coeffs(
    direct,
    img_name,
    meshdir,
    xyres,
    zstep,
    center_align: bool = True,
):

    img = skimage.io.imread(direct + img_name)
    
    exceptions_list = []
    
    # use pole of inaccessibility of "center_align" is false
    center = np.array([])
    if center_align == False:
        #get distance transform
        dist = ndimage.distance_transform_edt(img[1,:,:,:])
        #get index of the maximum in the distance tranform aka pole of inaccesibility
        center = np.array(np.unravel_index(np.argmax(dist), dist.shape))
        #flip dimensions so that they go from z,y,x to x,y,z
        center = center[::-1]
        
    # get meshes and save
    (cell_coeffs, grid_rec, exceptions_list), (image_, cell_mesh, grid, transform) = shparam_mod.get_shcoeffs_shiftres(
        img_name = img_name,
        image = img[1,:,:,:],
        lmax = 16,
        xyres = xyres,
        zstep = zstep,
        exceptions_list = exceptions_list,
        sigma = 2,
        alignment_2d = False,
        center_align = center_align,
        center_point = center,
        )
    cell_coeffs['cell'] = img_name.replace('_segmented.tiff','')

        
    (nuc_coeffs, grid_rec, nuc_list), (image_, nuc_mesh, grid, transform) = shparam_mod.get_shcoeffs_shiftres(
        img_name = img_name,
        image = img[0,:,:,:],
        lmax = 16,
        xyres = xyres,
        zstep = zstep,
        exceptions_list = [],
        sigma = 2,
        alignment_2d = False,
        center_align = True,
        )
    nuc_coeffs['cell'] = img_name.replace('_segmented.tiff','')



    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(meshdir + img_name.replace('_segmented.tiff','') + '_Cell_Mesh.vtp')
    writer.SetInputData(cell_mesh)
    writer.Write()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(meshdir + img_name.replace('_segmented.tiff','') + '_Nuclear_Mesh.vtp')
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
    cz, cy, cx = np.where(img[1,:,:,:])
    nz, ny, nx = np.where(img[0,:,:,:])

    #get intensity features
    raw_img = skimage.io.imread(direct + img_name.replace('segmented','raw'))
    # nucleus_int = im_temp_whole_int[1,:,:,:]
    mem_feat = get_intensity_features(raw_img[1,:,:,:], img[1,:,:,:])
    mem_keylist = [x for x in list(mem_feat) if not x.endswith('lcc')]
    
    #Append shape metrics to dataframe
    Shape_Stats = {'cell': img_name.replace('_segmented.tiff',''),
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
    
    

    return Shape_Stats, cell_coeffs, nuc_coeffs, exceptions_list