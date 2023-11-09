# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:41:29 2022

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
            tempdata = {'cell':count, 'area':prop.area, 'extent':prop.extent}
            tempdf = tempdf.append(tempdata, ignore_index=True)
        nuc_value = tempdf[tempdf.area>6000].sort_values(by='extent', ascending=False).index[0]+1
        im_labeled[im_labeled != nuc_value] = 0

    seg = im_labeled.astype(np.uint8)
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



def caax_noise_fill(caax_ch):
    #find the intensity peak of the noise
    hist = np.histogram(caax_ch[caax_ch>200], bins=list(range(0,round(np.max(caax_ch)+100),10)), range=(0,round(np.max(caax_ch)+100)))
    peaks, properties = signal.find_peaks(hist[0],prominence=20000, width=3.25)
    #find positions in the image with values at or below noise level
    noise_positions = np.where(np.logical_and(caax_ch>properties['left_ips'][0]*10 , caax_ch<properties['right_ips'][0]*11))
    noise_sample = caax_ch[noise_positions[0],noise_positions[1],noise_positions[2]]
    noise_max = np.max(noise_sample)
    #get sample of noise from values
    r_fill = np.random.choice(noise_sample, len(caax_ch[caax_ch<np.max(noise_sample)]))
    return noise_max, r_fill

    
def nuc_noise_fill(nuc_ch):
    #find the intensity peak of the noise
    hist = np.histogram(nuc_ch[nuc_ch>100], bins=list(range(0,round(np.max(nuc_ch)),10)), range=(0,round(np.max(nuc_ch))))
    peaks, properties = signal.find_peaks(hist[0], prominence=5000,width=4)
    #find positions in the image with values at or below noise level
    noise_positions = np.where(np.logical_and(nuc_ch>properties['left_ips'][0]*10 , nuc_ch<properties['right_ips'][0]*11))
    noise_sample = nuc_ch[noise_positions[0],noise_positions[1],noise_positions[2]]
    noise_max = np.max(noise_sample)
    #get sample of noise from values
    r_fill = np.random.choice(noise_sample, len(nuc_ch[nuc_ch<np.max(noise_sample)]))
    return noise_max, r_fill
    



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
        thebox = skimage.measure.regionprops(im_labeled)[0].bbox
        #check if segmentation was poor
        check = [x == thebox[n] for n, x in enumerate(tuple([0,0,0,temp_im.shape[0], temp_im.shape[1], temp_im.shape[2]]))]
        if sum(check)>=4:
            pass
        else:
            cent = skimage.measure.regionprops(im_labeled)[0].centroid
            data = {'image': image_name,
                    'cell': row.cell,
                    'frame':row.frame,
                    'x':(cent[2]+xmincrop)*xyres, 
                    'y':(cent[1]+ymincrop)*xyres, 
                    'z':(cent[0]+zmincrop)*zstep,
                    'cropx':cent[2], 
                    'cropy':cent[1], 
                    'cropz':cent[0],
                    'z_min':int(thebox[0]+zmincrop),
                    'y_min':int(thebox[1]+ymincrop),
                    'x_min':int(thebox[2]+xmincrop),
                    'z_max':int(thebox[3]+zmincrop),
                    'y_max':int(thebox[4]+ymincrop),
                    'x_max':int(thebox[5]+xmincrop)}
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
        yderv = interpolate.splev(u,tck,der=1)
        yderv2 = interpolate.splev(u,tck,der=2)
        traj = np.vstack(yderv).T
        acc = np.vstack(yderv2).T
        
        
    for j in range(1,traj.shape[0]-1):
        
        temp_df = df.iloc[j,:]
        # row = temp.iloc[list(temp.frame == temp_df.frame)].squeeze()
        
        vec = traj[j,:]
        temp_acc = acc[j,:]
        #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        xaxis = np.array([[0,0,0],[1,0,0], [5,0,0]]).astype('float64')
        current_vec = np.stack(([0,0,0],vec), axis = 0)
        current_vec = np.concatenate((current_vec,[5*vec]), axis = 0)
        rotationthing = R.align_vectors(xaxis, current_vec)
        #below is actual rotation matrix if needed
        #rot_mat = rotationthing[0].as_matrix()
        rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)
        euler_angles = pd.DataFrame({'x':[rotthing_euler[0]], 'y':[rotthing_euler[1]], 'z': [rotthing_euler[2]]})
    
        
        #get current cropped image using updated bounding box from single cell trajectory segmentation
        xmincrop = int(xynomore(temp_df.x_min-xy_buffer, im_temp_whole))
        ymincrop = int(xynomore(temp_df.y_min-xy_buffer, im_temp_whole))
        zmincrop = int(znomore(temp_df.z_min-z_buffer, im_temp_whole))
        raw_img = im_temp_whole[int(temp_df.frame),
                    :,
                    zmincrop:int(znomore(temp_df.z_max+z_buffer, im_temp_whole,)+1),
                    ymincrop:int(xynomore(temp_df.y_max+xy_buffer, im_temp_whole,)+1),
                    xmincrop:int(xynomore(temp_df.x_max+xy_buffer, im_temp_whole,)+1)]
        
        
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
            noise_max, r_fill = caax_noise_fill(rimg[0,:,:,:])
            rimg[0,:,:,:][rimg[0,:,:,:]<noise_max] = r_fill
            
            
            #fill in the expanded parts of the image with similar noise to the rest of the image
            noise_max, r_fill = nuc_noise_fill(rimg[1,:,:,:])
            rimg[1,:,:,:][rimg[1,:,:,:]<noise_max] = r_fill
        
        if norm_rot == 'widest':
            #fill in the expanded parts of the image with similar noise to the rest of the image
            noise_max, r_fill = caax_noise_fill(rimg[0,:,:,:])
            rimg_copy = rimg.copy()
            rimg_copy[0,:,:,:][rimg_copy[0,:,:,:]<noise_max] = r_fill
            
            # get mesh to rotate for normal alignment
            temp_seg = segment_caax_final(rimg_copy[0,:,:,:])
            mesh, image_, centroid = shtools.get_mesh_from_image(image=temp_seg, sigma=2)
            
            
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
            noise_max, r_fill = caax_noise_fill(rimg[0,:,:,:])
            rimg[0,:,:,:][rimg[0,:,:,:]<noise_max] = r_fill
            
            
            #fill in the expanded parts of the image with similar noise to the rest of the image
            noise_max, r_fill = nuc_noise_fill(rimg[1,:,:,:])
            rimg[1,:,:,:][rimg[1,:,:,:]<noise_max] = r_fill
        
        #save the info about cell
        info = pd.DataFrame({'cell': temp_df.cell,
                'frame': temp_df.frame,
                'dist': temp_df.dist,
                'velocity_vec': vec,
                'acceleration_vec':temp_acc,
                'euler_angles': rotthing_euler,
                'normal_rotation_angle': widestangle
                })
        info.to_csv(savedir + image_name + '_cell_' + str(int(cell['cell'].iloc[0])) + '_frame_' + str(int(info.frame.iloc[0])) + '_cell_info.csv')
        
        
        # segment rotated image
        seg_rimg = np.zeros(rimg.shape)
        seg_rimg[0,:,:,:] = segment_caax_final(rimg[0,:,:,:])
        seg_rimg[1,:,:,:] = segment_nucleus(rimg[1,:,:,:])
        
        
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
        im_file = savedir + image_name + '_cell_' + str(int(cell['cell'].iloc[0])) + '_frame_' + str(int(info.frame.iloc[0])) + '_segmented.tiff'
        if os.path.exists(im_file):
            os.remove(im_file)
        writer = OmeTiffWriter(im_file)
        writer.save(out, dimension_order = "CZYX")
        
        
        #CROP THE RAW IMAGE
        rimg = rimg[:,
                    bound[0]-5:bound[3]+5,
                    bound[1]-5:bound[4]+5,
                    bound[2]-5:bound[5]+5]       
        #SAVE THE RAW IMAGE
        raw_file = savedir + image_name + '_cell_' + str(int(cell['cell'].iloc[0])) + '_frame_' + str(int(info.frame.iloc[0])) + '_raw.tiff'
        if os.path.exists(raw_file):
            os.remove(raw_file)
        writer = OmeTiffWriter(raw_file)
        writer.save(rimg, dimension_order = "CZYX")
        
        

            

        
    # return df.drop(['cropx', 'cropy','cropz'], axis = 1).to_dict()