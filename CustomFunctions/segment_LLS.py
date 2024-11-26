# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:58:51 2024

@author: Aaron
"""

import os
import re
import math
import multiprocessing
import pandas as pd
import numpy as np
import numpy.ma as ma
from itertools import groupby
from operator import itemgetter
from aicsimageio.readers.czi_reader import CziReader
from aicsimageio.writers import OmeTiffWriter
import skimage.measure
from skimage.morphology import remove_small_objects
from aicssegmentation.core.MO_threshold import MO
from aicssegmentation.core.utils import hole_filling
from aicssegmentation.core import vessel
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_3d
from CustomFunctions import metadata_funcs
from scipy.spatial import distance
from scipy import interpolate
from CustomFunctions.persistance_activity import get_pa, DA_3D
from CustomFunctions.mp_funcs import quickcaaxseg
from CustomFunctions.MO_Threshold import MO_ma

# Function to find Angle
def angle_distance(a1, b1, c1, a2, b2, c2):
    a1,b1,c1 = [a1,b1,c1]/np.linalg.norm([a1,b1,c1])
    a2,b2,c2 = [a2,b2,c2]/np.linalg.norm([a2,b2,c2])
    d = ( a1 * a2 + b1 * b2 + c1 * c2 )
    e1 = math.sqrt( a1 * a1 + b1 * b1 + c1 * c1)
    e2 = math.sqrt( a2 * a2 + b2 * b2 + c2 * c2)
    d = d / (e1 * e2)
    A = math.degrees(math.acos(d))
    return A

def collect_results(result):
    """Uses apply_async's callback to setup up a separate Queue for each process.
    This will allow us to collect the results from different threads."""
    results.append(result)
    


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



def twodholefill(thresh, hole_min, hole_max):
    YZ = thresh.swapaxes(0,2)
    YZ_fill = hole_filling(YZ, hole_min, hole_max, fill_2d=True)
    YZrev = YZ_fill.swapaxes(2,0)
    XZ = YZrev.swapaxes(0,1)
    XZ_fill = hole_filling(XZ, hole_min, hole_max, fill_2d=True)
    XZrev = XZ_fill.swapaxes(1, 0)
    XY = hole_filling(XZrev, hole_min, hole_max, fill_2d=True)
    return XY



def segment_caax_decon(im):
    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [0]
    gaussian_smoothing_sigma = 1
    ################################
    # intensity normalization
    struct_img = intensity_normalization(im, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    #remove mega bright pixels
    structure_img_smooth = ma.masked_array(smooth, mask = smooth>np.percentile(smooth[im>100], 96))
    # step 1: Masked-Object (MO) Thresholding
    thresh_img = MO_ma(structure_img_smooth, global_thresh_method='tri', object_minArea=50000, local_adjust = 0.92)
    #filament filter
    ves = vessel.filament_2d_wrapper(structure_img_smooth, [[1.5,0.2]])
    #combine
    both = np.logical_or(ves, thresh_img)
    # set minimum area to just less that largest object
    im_labeled, n_labels = skimage.measure.label(
                              both, background=0, return_num=True)
    if n_labels > 1:
        im_props = skimage.measure.regionprops(im_labeled)
        tempdf = pd.DataFrame([])
        for count, prop in enumerate(im_props):
            area = prop.area
            tempdata = {'cell':count, 'area':area}
            tempdf = tempdf.append(tempdata, ignore_index=True)
        minArea = int(tempdf.area.max()-2)
        # create segmentation mask               
        both = remove_small_objects(im_labeled, min_size=minArea, connectivity=1, in_place=False)
    #fill holes in segmentation twice
    hole_max = 5000
    hole_min = 1
    both = twodholefill(both, hole_min, hole_max)
    both = twodholefill(both, hole_min, hole_max)
    #convert to 8bit binary
    both = both.astype(np.uint8)
    both[both > 0] = 255
    return both

def segment_nuc_decon(im,
                      mask,):
    ######### nuclear seg
    intensity_scaling_param = [0]
    gaussian_smoothing_sigma = 1
    ################################
    # intensity normalization
    struct_img = intensity_normalization(im, scaling_param=intensity_scaling_param)
    # smoothing with 3d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    # step 1: Masked-Object (MO) Thresholding
    thresh_img = MO(structure_img_smooth, global_thresh_method='tri', object_minArea=25000, local_adjust = 0.995)
    #remove small objects
    thresh_img = remove_small_objects(thresh_img, min_size=25000, connectivity=1, in_place=False)
    #fill holes in segmentation
    hole_max = 1000
    hole_min = 1
    thresh_img_fill = twodholefill(thresh_img, hole_min, hole_max)
    #restrict segmentation to membrane mask
    thresh_img_fill = np.logical_and(mask, thresh_img_fill)
    #change make it 8bit
    thresh_img_fill = thresh_img_fill.astype(np.uint8)
    thresh_img_fill[thresh_img_fill > 0] = 255
    return thresh_img_fill


def segment_actin_decon(im,
                        mask,
                        hilo: bool = False, #whether or not to do high medium and low threshold
                        ):
    if im[mask].mean() > 25:
        ######### nuclear seg
        intensity_scaling_param = [0]
        gaussian_smoothing_sigma = 1
        ################################
        # intensity normalization
        struct_img = intensity_normalization(im, scaling_param=intensity_scaling_param)
        # smoothing with 3d gaussian filter 
        structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
        #decide which and how many thresholds
        if hilo:
            allthresh = [1.5,2.0,2.5] #low, medium, and high
        else:
            allthresh = [2.0] #just medium
        seg = np.zeros((len(allthresh), im.shape[-3], im.shape[-2], im.shape[-1]))
        for i, a in enumerate(allthresh):
            # threshold in the membrane mask
            th = skimage.filters.threshold_otsu(structure_img_smooth[mask]) * a
            # print(structure_img_smooth.max(), th)
            seg[i] = structure_img_smooth > th
            #restrict segmentation to membrane mask
            seg[i] = np.logical_and(mask, seg[i])
            #change make it 8bit
            seg[i] = seg[i].astype(np.uint8)
            seg[i][seg[i] > 0] = 255
        return seg if seg.shape[0]>1 else seg[0]
    else:
        # print('No myosin signal')
        if hilo:
            seg = np.zeros((3, im.shape[-3], im.shape[-2], im.shape[-1]))
            seg = seg.astype(np.uint8)
        else:
            seg = np.zeros(im.shape)
            seg = seg.astype(np.uint8)
        return seg



def segment_myosin_decon(im,
                         mask,
                         hilo: bool = False, #whether or not to do high medium and low threshold
                         ):
    if im[mask].mean() > 30:
        ######### nuclear seg
        intensity_scaling_param = [0]
        gaussian_smoothing_sigma = 1
        ################################
        # intensity normalization
        struct_img = intensity_normalization(im, scaling_param=intensity_scaling_param)
        # smoothing with 3d gaussian filter 
        structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
        #decide which and how many thresholds
        if hilo:
            allthresh = [1.4,1.9,2.4] #low, medium, and high
        else:
            allthresh = [1.9] #just medium
        seg = np.zeros((len(allthresh), im.shape[-3], im.shape[-2], im.shape[-1]))
        for i, a in enumerate(allthresh):
            # threshold in the membrane mask
            th = skimage.filters.threshold_otsu(structure_img_smooth[mask]) * a
            # print(structure_img_smooth.max(), th)
            seg[i] = structure_img_smooth > th
            #restrict segmentation to membrane mask
            seg[i] = np.logical_and(mask, seg[i])
            #change make it 8bit
            seg[i] = seg[i].astype(np.uint8)
            seg[i][seg[i] > 0] = 255
        return seg if seg.shape[0]>1 else seg[0]
    else:
        # print('No myosin signal')
        if hilo:
            seg = np.zeros((3, im.shape[-3], im.shape[-2], im.shape[-1]))
            seg = seg.astype(np.uint8)
        else:
            seg = np.zeros(im.shape)
            seg = seg.astype(np.uint8)
        return seg
    
    
    


def getbb(im):
    ### start by segmenting the large image to get all of the "primary" cells' bounding boxes for further cropping
    rescaled = skimage.transform.rescale(im,0.25, preserve_range=True)
    ### get the shape of the rescaled to use later to leave out objects
    shape = rescaled.shape
    #mask the top pixels with signal
    mare = ma.masked_array(rescaled, mask = rescaled>np.percentile(rescaled[rescaled>100], 97))
    seg = MO_ma(mare, global_thresh_method='tri', object_minArea=200, local_adjust = 0.92)
    im_labeled, n_labels = skimage.measure.label(
                              seg, background=0, return_num=True,  )

    im_props = skimage.measure.regionprops(im_labeled)
    tempdf = pd.DataFrame([])
    for count, prop in enumerate(im_props):
        z,y,x = prop.centroid
        thebox = np.array(prop.bbox)*4
        area = prop.area * 64
        td = {'cell':count, 'z_min':thebox[0], 'y_min':thebox[1], 
                'x_min':thebox[2],'z_max':thebox[3], 'y_max':thebox[4], 'x_max':thebox[5],
               'z':z, 'y':y, 'x': x, 'z_range': seg.shape[-3], 'area':area}
        #ensure only things that aren't on the edge are chosen
        if (td['z_min']>0) and (td['y_min']>0) and (td['x_min']>0) and (td['z_max']/4<shape[-3]) and (td['y_max']/4<shape[-2]) and (td['x_max']/4<shape[-1]):
            tempdf = tempdf.append(td, ignore_index=True)
    
    if (len(tempdf)>0) and (tempdf.loc[tempdf['area'].idxmax(),'area']>50000):
        #return the largest object that isn't touching an edge    
        return tempdf.loc[tempdf['area'].idxmax()].to_dict()
    else:
        return {'cell':np.nan, 'z_min':np.nan, 'y_min':np.nan, 
                'x_min':np.nan,'z_max':np.nan, 'y_max':np.nan, 'x_max':np.nan,
               'z':np.nan, 'y':np.nan, 'x': np.nan, 'z_range': np.nan, 'area':np.nan}



def LLSseg(savedir:str,
           image_name:str,
           cropdict: dict,
           im:np.array,
           struct:str,
           xyres:float,
           zstep:float,
           decon:bool,
           orig_size:bool = False,
           orig_shape: np.array = np.zeros(1),
           xy_buffer: int = 20,
           z_buffer: int = 20,
           hilo: bool = False, #whether or not to do multiple different thresholds on the signal data
           ):
    
    newcrop = getbb(im[1,:,:,:])
    #combine crop dictionaries
    cropdict.update(newcrop)
    #actually crop the image
    #get the right cropping boundaries
    xmincrop = int(max(0, cropdict['x_min']-xy_buffer))
    ymincrop = int(max(0, cropdict['y_min']-xy_buffer))
    zmincrop = int(max(0, cropdict['z_min']-z_buffer))
    zmaxcrop = int(min(cropdict['z_max']+z_buffer, orig_shape[-3]))
    ymaxcrop = int(min(cropdict['y_max']+xy_buffer, orig_shape[-2])+1)
    xmaxcrop = int(min(cropdict['x_max']+xy_buffer, orig_shape[-1])+1)
    im = im[:,zmincrop:zmaxcrop,ymincrop:ymaxcrop,xmincrop:xmaxcrop]
    #flip channels so that membrane is first
    im = np.flip(im, axis = 0)
    
    #file name template
    cell_name = re.split(r'(?<=Subset-\d{2})', image_name)[0] + '_frame_'+ str(int(cropdict['frame']))
    shortimname = image_name.split('-Subset')[0]
    cell_number = re.findall(r"Subset-(\d+)", image_name)[0]
    
    
    ### segment the cells depending on their signals
   
    if hilo:
        bothch = np.zeros((4,im.shape[-3], im.shape[-2], im.shape[-1]))
        if decon:
            bothch[0,:,:,:] = segment_caax_decon(im[0,:,:,:])
            #make boolean mask for secondary signals
            mask = bothch[0,:,:,:].astype(bool)
            if struct == 'nucleus':
                bothch[1:,:,:,:] = segment_nuc_decon(im[1,:,:,:], mask)
                bothch = bothch[:2]
            elif struct == 'actin':
                bothch[1:,:,:,:] = segment_actin_decon(im[1,:,:,:], mask, hilo)
            elif struct == 'myosin':
                bothch[1:,:,:,:] = segment_myosin_decon(im[1,:,:,:], mask, hilo)
        else:
            pass
    
    else:
        bothch = np.zeros(im.shape)
        if decon:
            bothch[0,:,:,:] = segment_caax_decon(im[0,:,:,:])
            #make boolean mask for secondary signals
            # mask = np.invert(bothch[1,:,:,:].astype(bool))
            mask = bothch[0,:,:,:].astype(bool)
            if struct == 'nucleus':
                bothch[1,:,:,:] = segment_nuc_decon(im[1,:,:,:], mask)
            elif struct == 'actin':
                bothch[1,:,:,:] = segment_actin_decon(im[1,:,:,:], mask)
            elif struct == 'myosin':
                bothch[1,:,:,:] = segment_myosin_decon(im[1,:,:,:], mask)
        else:
            pass
    


    #get intensity features for both channels
    mem_feat = get_intensity_features(im[0,:,:,:], bothch[0,:,:,:])
    mem_keylist = [x for x in list(mem_feat) if not x.endswith('lcc')]
    str_feat = get_intensity_features(im[1,:,:,:], bothch[0,:,:,:])
    str_keylist = [x for x in list(str_feat) if not x.endswith('lcc')]

    #get final centroid
    cent = np.mean(np.nonzero(bothch[0,:,:,:]), axis = 1)
    
    #SAVE SEGMENTED IMAGE
    out=bothch.astype(np.uint8)
    out[out>0]=255
    
    
    if orig_size:
        #make empty image of the original shape
        orig_im = np.zeros(orig_shape)
        #get the right cropping boundaries
        xmincrop = int(max(0, cropdict['x_min']-xy_buffer))
        ymincrop = int(max(0, cropdict['y_min']-xy_buffer))
        zmincrop = int(max(0, cropdict['z_min']-z_buffer))
        zmaxcrop = int(min(cropdict['z_max']+z_buffer, orig_shape[-3]))
        ymaxcrop = int(min(cropdict['y_max']+xy_buffer, orig_shape[-2])+1)
        xmaxcrop = int(min(cropdict['x_max']+xy_buffer, orig_shape[-1])+1)
        #put segmented image into original empty image
        orig_im[:,
                zmincrop:zmaxcrop,
                ymincrop:ymaxcrop,
                xmincrop:xmaxcrop] = out.copy()
        # remove file if it already exists
        seg_file = savedir + cell_name + '_segmentedfull.tiff'
        if os.path.exists(seg_file):
            os.remove(seg_file)
        OmeTiffWriter.save(orig_im, seg_file, dimension_order = "CZYX")
        
        
    # remove file if it already exists
    seg_file = savedir + cell_name + '_segmented.tiff'
    if os.path.exists(seg_file):
        os.remove(seg_file)
    OmeTiffWriter.save(out, seg_file, dimension_order = "CZYX")
    
    #SAVE THE RAW IMAGE
    raw_file = savedir + cell_name + '_raw.tiff'
    if os.path.exists(raw_file):
        os.remove(raw_file)
    OmeTiffWriter.save(im, raw_file, dimension_order = "CZYX")
        
    
    ##### check that the shape isn't touching the border of the image
    edges = np.concatenate((
        out[1, 0, :, :],            # front face
        out[1, -1, :, :],           # back face
        out[1, :, 0, :],            # left face
        out[1, :, -1, :],           # right face
        out[1, :, :, 0],            # top face
        out[1, :, :, -1]            # bottom face
        ), axis=None).astype(bool)
    #if the cell is touching the edge of the image more than a little bit, don't add its data
    if np.sum(edges)>50:
        return None
    else:
        #save the info about cell
        data = {'image': shortimname,
                'cellnumber': cell_number,
                'cell': cell_name,
                'structure': struct,
                'frame': cropdict['frame'],
                'time': cropdict['time'],
                'x':(cent[2]+cropdict['x_min'])*xyres, 
                'y':(cent[1]+cropdict['y_min'])*xyres, 
                'z':(cent[0]+cropdict['z_min'])*zstep,
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
    
    

