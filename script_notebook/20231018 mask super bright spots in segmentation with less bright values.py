# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:35:36 2023

@author: Aaron
"""

img = TiffReader('D:/Aaron/Data/Galvanotaxis_Longtracks/Processed_Data/20230914_488EGFP-CAAX_3mA_Galv_1_cell_11_frame_17_raw.tiff').data


raw_img[0,:,:,:].copy()


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
    thresh_img = MO(structure_img_smooth, global_thresh_method='tri', object_minArea=3000, local_adjust = 0.89)


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
        
        #remove the brightest pixels from the cell of interest
        values = structure_img_smooth[im_labeled==realin].flatten()
        structure_img_smooth[structure_img_smooth>np.percentile(values, 97)] = np.percentile(values, 90)
        # threshold the new modified image
        thresh_img, globalimg = MO(structure_img_smooth, global_thresh_method='tri', object_minArea=3000, local_adjust = 0.89, return_object = True)

    else:
        #remove the brightest pixels from the cell of interest
        values = structure_img_smooth[im_labeled>0].flatten()
        structure_img_smooth[structure_img_smooth>np.percentile(values, 97)] = np.percentile(values, 90)
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


    OmeTiffWriter.save(seg,'C:/Users/Aaron/Documents/Python Scripts/temp/seg.ome.tiff')     


















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
            if n == realin:
                pass
            else:
                print('yes')
                structure_img_smooth = partial_cell_removal_caax(structure_img_smooth, im_labeled, n+1)
        
        #remove the brightest pixels from the cell of interest
        values = structure_img_smooth[im_labeled==realin].flatten()
        structure_img_smooth[structure_img_smooth>np.percentile(values, 97)] = np.percentile(values, 90)
        # threshold the new modified image
        thresh_img = MO(structure_img_smooth, global_thresh_method='tri', object_minArea=750, local_adjust = 0.95)

    else:
        #remove the brightest pixels from the cell of interest
        values = structure_img_smooth[im_labeled>0].flatten()
        structure_img_smooth[structure_img_smooth>np.percentile(values, 97)] = np.percentile(values, 90)
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
    hole_max = 750
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


    OmeTiffWriter.save(seg,'C:/Users/Aaron/Documents/Python Scripts/temp/seg.ome.tiff')