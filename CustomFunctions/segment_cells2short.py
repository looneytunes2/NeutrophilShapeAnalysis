# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 18:28:52 2022

@author: Aaron
"""

   
import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R
from aicsimageio.writers import OmeTiffWriter

# function for core algorithm
from aicssegmentation.core.utils import hole_filling, topology_preserving_thinning
from aicssegmentation.core.vessel import filament_2d_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_3d
from skimage.morphology import remove_small_objects     # function for post-processing (size filter)
from aicssegmentation.core.MO_threshold import MO
from CustomFunctions.utils import twodholefill, get_intensity_features


import skimage.measure
import cv2

import tifffile



def MM_slicetostack_reader(direct, #directory of the image slices
                           frame, #which image frame to open
                           shape, #shape of a single frame in czyx
                           zrange, #iterable with all the z slices to include
                           ):
    #detect position in directory
    if 'Pos' in str(direct):
        pos = int(direct.split('Pos')[-1][0])
    else:
        pos = 0
    if len(shape)>3:
        ch = shape[0]
        full = np.zeros((ch, len(zrange), shape[-2], shape[-1]), dtype=np.uint16)
    else:
        ch = 1
        full = np.zeros((len(zrange), shape[-2], shape[-1]), dtype=np.uint16)
    for c in range(ch):
        for i, z in enumerate(zrange):
            full[i,:,:] = tifffile.imread(direct.joinpath(f'img_channel{c:03}_position{pos:03}_time{frame:09}_z{z:03}.tif'))
    return full




#### writes over a non-centered cell segmentation
def partial_cell_removal_caax(caax_ch, #raw data
                              im_labeled, #labeled image to use for masks
                              num, #intensity in im_labelled to use as "mask"
                              ):
    #get the positions of the noise peak and everything below that
    hist = np.histogram(caax_ch, bins=np.arange(0,1,0.002))
    noisemax = hist[1][np.argmax(hist[0])+2] #chose 1 above the peak
    noise_positions = np.where(caax_ch<=noisemax)
    noise_sample = caax_ch[noise_positions[0],noise_positions[1],noise_positions[2]]
    #dilate image a bit so that the partial cell gets more completely removed
    kern = np.ones((7,7), np.uint8)
    new = np.zeros(im_labeled.shape)
    for x in range(im_labeled.shape[0]):
        new[x,:,:] = cv2.dilate(im_labeled[x,:,:].astype(np.uint8), kern, iterations = 1)
    r_fill = np.random.choice(noise_sample, len(np.where(new ==num)[0]))
    caax_ch[np.where(new == num)] = r_fill
    return caax_ch


#### function to clip image to a min of 0 and max of 1
def int_min_max(img, #image to modify
                high, #intensity value to set to 1
                low = 0 #intensity value to set to 0
                ):
    if low == 0:
        low = img.min()
    clipped = np.clip(img, low, high)
    newimg = (clipped-low)/(high-low)
    return newimg


def segment_caax_hl60(img):
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
    thresh_img = MO(structure_img_smooth, global_thresh_method='tri', object_minArea=750, local_adjust = 0.95)
    
    # detect if there's more than one object in the thresholded image
    im_labeled, n_labels = skimage.measure.label(
                              thresh_img.astype(np.uint8), background=0, return_num=True)
    #if there's more than one object try to erase the non-focused cell and re-threshold
    if n_labels > 1:
        im_props = skimage.measure.regionprops(im_labeled)
        imcent = np.array(img.shape)/2
        distances = []
        for count, prop in enumerate(im_props):
            #append the distance between this object and the center of the image
            distances.append(distance.pdist(np.stack([imcent, np.array(prop.centroid)])))
        #get the index of the closest object to the center of the image
        realin = np.argmin(distances)
        for n in list(range(n_labels)):
            if n != realin:
                structure_img_smooth = partial_cell_removal_caax(structure_img_smooth, im_labeled, n+1)

        #remove the brightest pixels from the cell of interest
        values = structure_img_smooth[im_labeled==realin+1].flatten()
        structure_img_smooth[structure_img_smooth>np.percentile(values, 98)] = np.percentile(values, 90)
        # threshold the new modified image
        thresh_img = MO(structure_img_smooth, global_thresh_method='tri', object_minArea=750, local_adjust = 0.95)

    else:
        #remove the brightest pixels from the cell of interest
        values = structure_img_smooth[im_labeled>0].flatten()
        structure_img_smooth[structure_img_smooth>np.percentile(values, 98)] = np.percentile(values, 90)
        # threshold the new modified image
        thresh_img = MO(structure_img_smooth, global_thresh_method='tri', object_minArea=750, local_adjust = 0.95)

        
    ################################
    ## PARAMETERS for this step ##
    # f3_param = [[1, 0.3]]
    # f2_param = [[1,0.22],[2, 0.17]]
    f2_param = [[0.5,0.3]]
    ################################
    
    fil_img = filament_2d_wrapper(structure_img_smooth, f2_param)
    
    
    seg = thresh_img + fil_img
    
    # fill in the holes
    hole_max = 2500
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

    ## get image in 8-bit binary
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255

    return seg



def segment_caax_el4(img):
    
    ################################
    ## PARAMETERS for this step ##
    gaussian_smoothing_sigma = 1.5
    ################################
    # normalize by percentiles
    #find noise peak by histograme
    counts, vals = np.histogram(img, int((img.max()-img.min())/10))
    noise = vals[np.argmax(counts)]
    struct_img = int_min_max(img, np.percentile(img,99.5), noise)# np.percentile(img, 2))
    # 3d gaussian smoothening
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    
    
    ########## detect extra objects in the image and remove them
    # detect if there's more than one object in the thresholded image
    thresh_img = MO(structure_img_smooth, global_thresh_method='tri', object_minArea=750, local_adjust = 0.98)
    im_labeled, n_labels = skimage.measure.label(
                              thresh_img.astype(np.uint8), background=0, return_num=True)
    #if there's more than one object try to erase the non-focused cell and re-threshold
    if n_labels > 1:
        im_props = skimage.measure.regionprops(im_labeled)
        imcent = np.array(img.shape)/2
        distances = []
        for count, prop in enumerate(im_props):
            #append the distance between this object and the center of the image
            distances.append(distance.pdist(np.stack([imcent, np.array(prop.centroid)])))
        #get the index of the closest object to the center of the image
        realin = np.argmin(distances)
        for n in list(range(n_labels)):
            if n != realin:
                # structure_img_smooth = partial_cell_removal_caax(structure_img_smooth, im_labeled, n+1)
                struct_img = partial_cell_removal_caax(struct_img, im_labeled, n+1)
        ### finally re-smooth the image with removed cell(s)
        structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
        
        
    ############ filament filter in two orientations
    ## PARAMETERS for this step ##
    f2_param = [[0.5,0.5]]
    fil_img = filament_2d_wrapper(structure_img_smooth, f2_param)
    
    #rotate image
    rotim = np.rot90(structure_img_smooth, axes = (0,1))
    #use a different filament filter because of the different resolution
    f2_param = [[0.35,0.35]]
    rotfil_img = filament_2d_wrapper(rotim, f2_param)
    
    ### combine the filament filtered images
    seg = fil_img + np.rot90(rotfil_img, k=-1, axes = (0,1)) 
    
    #### 2D hole filling from every orientation twice
    hole_max = 3000
    hole_min = 1
    seg = twodholefill(seg, hole_min, hole_max)
    seg = twodholefill(seg, hole_min, hole_max)
    
    # set minimum area to just less that largest object
    im_labeled, n_labels = skimage.measure.label(
                              seg, background=0, return_num=True)
    
    if n_labels > 1:
        #get the label of the biggest thing
        im_props = skimage.measure.regionprops(im_labeled)
        areadict = [{'cell':count+1, 'area':prop.area} for count, prop in enumerate(im_props)]
        tempdf = pd.DataFrame(areadict)
        biggest = tempdf.sort_values('area').iloc[-1].cell
        #clear everything not with that label
        seg[im_labeled!=biggest] = 0

        
    ## get image in 8-bit binary
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255

    return seg





def seg_confocal_40x_memonly_fromslices(
    direct,
    imshape,
    row,
    savedir,
    xyres,
    zstep,
    croparr,
    whatseg = 'hl60',
):

    """
        Parameters
        ----------
        direct : str
            Directory of image slices.
        imshape : list or tuple
            3D shape of the original image stack.
        row : pd.DataFrame
            Single-row dataframe with info about cell and frame to segment. 
        xyres : float
            XY resolution of image in units/pixel.
        zstep : float
            Z resolution of image in units/pixel.
        croparr : list/array
            Array with xmincrop, xmaxcrop, ymincrop, ymaxcrop, zmincrop, zmaxcrop
            for opening the minimal image from slices.
        whatseg : str
            Defines what segmentation function to use.

        Returns
        -------
        data : dict
            Position and intensity info about the segmented cell.

        Other parameters
        ----------------

        Notes
        -----


    """
    #open the full zstack at this movie frame
    frameim = MM_slicetostack_reader(direct, int(row.frame), imshape, range(croparr[4],croparr[5]))
    #crop frame to the cell
    raw_img = frameim[
                    :,
                    croparr[2]:croparr[3],
                    croparr[0]:croparr[1]]
    
    #get cell name
    cell_name = row.cell
    
    # segment cropped image depending on subject
    if whatseg == 'hl60':
        seg_rimg = segment_caax_hl60(raw_img)
    elif whatseg == 'el4':
        seg_rimg = segment_caax_el4(raw_img)
    
    #only continue to process image if the segmentation doesn't touch the
    #image border
    if not(np.any(seg_rimg[0, :, :] > 0) or
           np.any(seg_rimg[-1, :, :] > 0) or
           np.any(seg_rimg[:, 0, :] > 0) or
           np.any(seg_rimg[:, -1, :] > 0) or
           np.any(seg_rimg[:, :, 0] > 0) or
           np.any(seg_rimg[:, :, -1] > 0)
           ):
        
        #get intensity features
        mem_feat = get_intensity_features(raw_img, seg_rimg)
        mem_keylist = [x for x in list(mem_feat) if not x.endswith('lcc')]
    
    
        #crop the segmented image
        im_labeled, n_labels = skimage.measure.label(
                                  seg_rimg, background=0, return_num=True)
        im_props = skimage.measure.regionprops(im_labeled)
        
        
        
        #get original centroids
        cent = im_props[0].centroid
    
        #SAVE SEGMENTED IMAGE
        out=seg_rimg.astype(np.uint8)
        out[out>0]=255
        
        
        # remove file if it already exists
        seg_file = savedir.joinpath(cell_name + '_segmented.tiff')
        if seg_file.exists():
            seg_file.unlink()
        OmeTiffWriter.save(out, seg_file, dimension_order = "CZYX")
        
       
        #SAVE THE RAW IMAGE
        raw_file = savedir.joinpath(cell_name + '_raw.tiff')
        if raw_file.exists():
            raw_file.unlink()
        OmeTiffWriter.save(raw_img, raw_file, dimension_order = "CZYX")
        
        
    
        
        #Append shape metrics to dataframe
        data = {'image': row.CellID.split('_cell_')[0],
                'CellID': row.CellID,
                 'cell': cell_name,
                 'structure': 'none',
                 'frame': row.frame,
                 'x':(cent[2]+croparr[0])*xyres, #centroid within the big image in microns
                 'y':(cent[1]+croparr[2])*xyres, #centroid within the big image in microns
                 'z':(cent[0]+croparr[4])*zstep,#centroid within the big image in microns
                 'xmincrop': croparr[0],
                 'ymincrop': croparr[2],
                 'zmincrop': croparr[4],
                 'xmaxcrop': croparr[1],
                 'ymaxcrop': croparr[3],
                 'zmaxcrop': croparr[5],
                'Cell_'+mem_keylist[0]: mem_feat[mem_keylist[0]],
                'Cell_'+mem_keylist[1]: mem_feat[mem_keylist[1]],
                'Cell_'+mem_keylist[2]: mem_feat[mem_keylist[2]],
                'Cell_'+mem_keylist[3]: mem_feat[mem_keylist[3]],
                'Cell_'+mem_keylist[4]: mem_feat[mem_keylist[4]],
                'Cell_'+mem_keylist[5]: mem_feat[mem_keylist[5]]
                        }
        
        return data 




    

