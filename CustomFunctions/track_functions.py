# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 11:01:43 2021

@author: Aaron
"""

import numpy as np
import pandas as pd
import skimage.transform

from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_3d
from skimage.morphology import remove_small_objects    
from aicssegmentation.core.MO_threshold import MO

from CustomFunctions.segment_cells2short import MM_slicetostack_reader, twodholefill




def segment_caax_tracks_confocal_40x_fromsingle(imname, shape, ip, step, frame):
    #read image
    img = MM_slicetostack_reader(imname,frame,shape, range(0,shape[-3]))
    #shrink image by half
    rescaled = skimage.transform.rescale(img,0.5, preserve_range=True)
    
    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [1,5]
    gaussian_smoothing_sigma = 1
    ################################
    # intensity normalization
    struct_img = intensity_normalization(rescaled, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    # step 1: Masked-Object (MO) Thresholding
    thresh_img = MO(structure_img_smooth, local_adjust=0.92, global_thresh_method='tri', object_minArea=600)

    #do 2d hole fill
    hole_max = 500
    hole_min = 1
    thresh_img = twodholefill(thresh_img, hole_min, hole_max)
    
    minArea = 500
    seg = thresh_img > 0
    seg = remove_small_objects(seg>0, min_size=minArea, connectivity=1, in_place=False)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255
    
    
    #get objects
    im_labeled, n_labels = skimage.measure.label(seg, background=0, return_num=True)
    im_props = skimage.measure.regionprops(im_labeled)

    dictlist = []
    for count, prop in enumerate(im_props):
        thebox = np.array(prop.bbox)*2
        cent = np.array(prop.centroid)*2
        area = np.array(prop.area)*4
        convex_area = np.array(prop.convex_area)*4
        extent = prop.extent
        major_axis_length = np.array(prop.major_axis_length)*2
        minor_axis_length = np.array(prop.minor_axis_length)*2
    
        #intensity measures
        ind = np.where(im_labeled==int(count+1))
        intval = rescaled[ind]
        
        data = {'cell':count, 'frame':frame, 'z_min':thebox[0], 'y_min':thebox[1], 
                'x_min':thebox[2],'z_max':thebox[3], 'y_max':thebox[4], 'x_max':thebox[5],
               'z':cent[0]*step, 'y':cent[1]*ip, 'x': cent[2]*ip, 'z_range': img.shape[-3],
               'area':area, 'convex_area':convex_area, 'extent':extent,
               'minor_axis_length':minor_axis_length, 'major_axis_length':major_axis_length,
               'intensity_avg':intval.mean(), 'intensity_max':intval.max(), 'intensity_std':intval.std()}
        dictlist.append(data)
        
    df = pd.DataFrame.from_dict(dictlist)
            
    return df.values.tolist(), seg, frame, rescaled.shape


def segment_nuc_tracks_confocal_40x_fromsingle(imname, shape, ip, step, frame):

    #read image
    img = MM_slicetostack_reader(imname,frame,shape, range(0,shape[-3]))
    #shrink image by half
    rescaled = skimage.transform.rescale(img,0.5, preserve_range=True)

    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [1.5, 15]
    gaussian_smoothing_sigma = 1.5
    ################################
    # intensity normalization
    struct_img = intensity_normalization(rescaled, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    # step 1: Masked-Object (MO) Thresholding
    thresh_img, object_for_debug = MO(structure_img_smooth[:,:,:], global_thresh_method='tri', object_minArea=600, return_object=True)


    minArea = 170
    seg = thresh_img > 0
    seg = remove_small_objects(seg>0, min_size=minArea, connectivity=1, in_place=False)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255
    

    #get objects
    im_labeled, n_labels = skimage.measure.label(seg, background=0, return_num=True)
    im_props = skimage.measure.regionprops(im_labeled)

    
    dictlist = []
    for count, prop in enumerate(im_props):
        thebox = np.array(prop.bbox)*2
        cent = np.array(prop.centroid)*2
        area = np.array(prop.area)*4
        major_axis_length = np.array(prop.major_axis_length)*2
        minor_axis_length = np.array(prop.minor_axis_length)*2
    
        #intensity measures
        ind = np.where(im_labeled==int(count+1))
        intval = rescaled[ind]
        
        data = {'cell':count, 'frame':frame, 'z_min':thebox[0], 'y_min':thebox[1], 
                'x_min':thebox[2],'z_max':thebox[3], 'y_max':thebox[4], 'x_max':thebox[5],
               'z':cent[0]*step, 'y':cent[1]*ip, 'x': cent[2]*ip, 'z_range': img.shape[-3],
               'area':area,
               'minor_axis_length':minor_axis_length, 'major_axis_length':major_axis_length,
               'intensity_avg':intval.mean(), 'intensity_max':intval.max(), 'intensity_std':intval.std()}
        dictlist.append(data)
        
    df = pd.DataFrame.from_dict(dictlist)
            

    return df.values.tolist(), df.columns.to_list(), seg, frame, rescaled.shape







def segment_nuc_tracks_confocal_4x(struct_img0, ip, step, frame):

    df = pd.DataFrame()
    count = 0

    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [1.5,5.5]
    gaussian_smoothing_sigma = 1
    ################################
    # intensity normalization
    struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    # step 1: Masked-Object (MO) Thresholding
    thresh_img = MO(structure_img_smooth[:,:,:], global_thresh_method='tri', object_minArea=100, local_adjust= 0.992)
    
    
    seg = thresh_img.astype(np.uint8)
    seg[seg > 0] = 255

    #get objects
    im_labeled, n_labels = skimage.measure.label(seg, background=0, return_num=True)
    print(n_labels)
    im_props = skimage.measure.regionprops(im_labeled)

    
    for count, prop in enumerate(im_props):
        thebox = prop.bbox
        cent = prop.centroid
        area = prop.area
        major_axis_length = prop.major_axis_length
        minor_axis_length = prop.minor_axis_length
        
        
        data = {'cell':count, 'frame':frame, 'z_min':thebox[0], 'y_min':thebox[1], 
                'x_min':thebox[2],'z_max':thebox[3], 'y_max':thebox[4], 'x_max':thebox[5],
               'z':cent[0]*step, 'y':cent[1]*ip, 'x': cent[2]*ip, 'z_range': struct_img0.shape[-3],
               'area':area, 'minor_axis_length':minor_axis_length, 'major_axis_length':major_axis_length}
        df = df.append(data, ignore_index=True)
            

    return df.values.tolist(), seg, frame
