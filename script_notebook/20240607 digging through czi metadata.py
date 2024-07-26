# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 10:52:28 2024

@author: Aaron
"""

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy import ndimage, signal
from aicssegmentation.core.utils import hole_filling, topology_preserving_thinning
from aicssegmentation.core.vessel import filament_3d_wrapper, filament_2d_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_3d
from CustomFunctions.track_functions import MO
import skimage.measure
from skimage.morphology import remove_small_objects 
from aicsimageio.writers import OmeTiffWriter
from aicsimageio.readers.czi_reader import CziReader
import re


def partial_cell_removal_caax(caax_ch, #raw data
                              im_labeled, #labeled image to use for masks
                              num, #intensity in im_labelled to use as "mask"
                              ):
    #get the positions of the noise peak and everything below that
    hist = np.histogram(caax_ch, bins=np.arange(0,1,0.01))
    peaks, properties = signal.find_peaks(hist[0],prominence=50000)
    noise_positions = np.where(caax_ch<=hist[1][properties['right_bases'][0]])
    noise_sample = caax_ch[noise_positions[0],noise_positions[1],noise_positions[2]]
    r_fill = np.random.choice(noise_sample, len(np.where(im_labeled ==num)[0]))
    caax_ch[np.where(im_labeled == num)] = r_fill
    return caax_ch


def segment_caax_norot(img):
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
    thresh_img = MO(structure_img_smooth, global_thresh_method='tri', object_minArea=50000, local_adjust = 0.89)


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
            if n == realin:
                pass
            else:
                print('yes')
                structure_img_smooth = partial_cell_removal_caax(structure_img_smooth, im_labeled, n+1)

        # threshold the new modified image
        thresh_img, globalimg = MO(structure_img_smooth, global_thresh_method='tri', object_minArea=3000, local_adjust = 0.89, return_object = True)
        
        
        
    ################################
    ## PARAMETERS for this step ##
    # f3_param = [[1, 0.3]]
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
    thin_dist_preserve = 2
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




imdir = '//10.158.28.37/ExpansionHomesA/avlnas/LLS/2024-05-23/20240523_488_EGFP-CAAX_561_mysoin-mApple_37C_cell1-06-Subset-03_Frame128-151.czi'

imdir = 'G:/Cropped_LLS/20240524_488_EGFP-CAAX_640_actin-halotag_cell1-02-Subset-01_Frame1-60.czi'

im = CziReader(imdir)


img = im.data[0,1,:,:,:]

OmeTiffWriter.save(img, 'C:/Users/Aaron/Desktop/raw.tiff')

OmeTiffWriter.save(thresh_img.astype(np.uint8), 'C:/Users/Aaron/Desktop/thresh.tiff')







########## metadata shit
import xml.etree.ElementTree as ET
tree = ET.ElementTree(im.metadata[0])
root = tree.getroot()



for child in root[0]:
    print(child.tag, child.attrib)
    for cc in child:
        print(cc.tag)
        for c in cc:
            print(c.tag)


###### write the metadata to a text file
st = ET.tostring(root, encoding='utf8')
utf8_string = st.decode('utf8')
file_name = 'C:/Users/Aaron/Desktop/metadata.txt'
with open(file_name, "w", encoding="utf-8") as file:
    file.write(utf8_string)

[x for x in root.find('Information').find('Image')]

#get the 
adt = root.find('Information').find('Image').find('AcquisitionDateAndTime').text
dt = re.findall(r'T(.*?)\.',adt)[0]
def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)
get_sec(dt)

time_elapsed = 0
#if the subset has been cropped in time
subsetstring = root.find('CustomAttributes').find('SubsetString').text
if 'T' in subsetstring:
    first, last = [int(x) for x in re.findall(r'T\(([^)]*)\)',subsetstring)[0].split('-')]
    if first != 1:
        time_elapsed = time_elapsed + (first-1)*time_interval

######### time interval is somewhere in here:
time_interval = float(root.find('Experiment').find('ExperimentBlocks').find('AcquisitionBlock').find('SubDimensionSetups').find('TimeSeriesSetup').find('Interval').find('TimeSpan').find('Value').text)
