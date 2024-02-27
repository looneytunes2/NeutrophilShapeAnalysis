# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:38:42 2023

@author: Aaron
"""

plt.hist(hist[1],hist[0])
plt.xlim = (0,0.5)
plt.show()
image = TiffReader('D:/Aaron/Data/Galvanotaxis_Confocal_40x_30C_6s/Processed_Data/20231020_488EGFP-CAAX_4_cell_18_frame_32_raw.tiff')


img = image.data


import cv2
def partial_cell_removal_caax(caax_ch, #raw data
                              im_labeled, #labeled image to use for masks
                              num, #intensity in im_labelled to use as "mask"
                              ):
    #get the positions of the noise peak and everything below that
    hist = np.histogram(caax_ch, bins=np.arange(0,1,0.002))
    noisemax = hist[1][np.argmax(hist[0])+1]
    noise_positions = np.where(caax_ch<=noisemax)
    noise_sample = caax_ch[noise_positions[0],noise_positions[1],noise_positions[2]]
    #dilate image a bit so that the partial cell gets more completely removed
    kern = np.ones((5,5), np.uint8)
    new = np.zeros(im_labeled.shape)
    for x in range(im_labeled.shape[0]):
        new[x,:,:] = cv2.dilate(im_labeled[x,:,:].astype(np.uint8), kern, iterations = 1)
    r_fill = np.random.choice(noise_sample, len(np.where(new ==num)[0]))
    caax_ch[np.where(new == num)] = r_fill
    return caax_ch

kern = np.one((3,3), np.uint8)
new = np.zeros(im_labeled.shape)
for x in range(im_labeled.shape[0]):
    new[x,:,:] = cv2.dilate(im_labeled[x,:,:].astype(np.uint8), kern, iterations = 1)
caax_ch[np.where(new == num)] = r_fill
# dilated = cv2.dilate(im_labeled, kern, iterations = 1)


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
    thresh_img = MO(structure_img_smooth, global_thresh_method='tri', object_minArea=750, local_adjust = 0.95)
    
    # detect if there's more than one object in the thresholded image
    im_labeled, n_labels = skimage.measure.label(
                              thresh_img.astype(np.uint8), background=0, return_num=True)
    #if there's more than one object try to erase the non-focused cell and re-threshold
    if n_labels > 1:
        print('yesy')
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
                print(n+1)
                structure_img_smooth = partial_cell_removal_caax(structure_img_smooth, im_labeled, n+1)

        #remove the brightest pixels from the cell of interest
        values = structure_img_smooth[im_labeled==realin].flatten()
        structure_img_smooth[structure_img_smooth>np.percentile(values, 99.9)] = np.percentile(values, 90)
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
    seg = hole_filling(seg, hole_min, hole_max, fill_2d=True) 
    
    
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



OmeTiffWriter.save(caax_ch, 'C:/Users/Aaron/Desktop/gfy.ome.tiff')
