# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 18:49:22 2023

@author: Aaron
"""


def caax_noise_fill(caax_ch):
    #find the intensity peak of the noise
    hist = np.histogram(caax_ch, bins=list(range(0,round(np.max(caax_ch)),5)), range=(0,round(np.max(caax_ch))))
    peaks, properties = signal.find_peaks(hist[0],prominence=50000)
    #find positions in the image with values at or below noise level
    noise_positions = np.where(np.logical_and(caax_ch>properties['left_ips'][0]*10 , caax_ch<properties['right_ips'][0]*11))
    noise_sample = caax_ch[noise_positions[0],noise_positions[1],noise_positions[2]]
    noise_max = np.max(noise_sample)
    #get sample of noise from values
    r_fill = np.random.choice(noise_sample, len(caax_ch[caax_ch<np.max(noise_sample)]))
    return noise_max, r_fill


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


img = raw_img[0,:,:,:].copy()




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
thresh_img, globalimg = MO(structure_img_smooth[:,:,:], global_thresh_method='tri', object_minArea=3000, local_adjust = 0.89, return_object = True)

# detect if there's more than one object in the thresholded image
im_labeled, n_labels = skimage.measure.label(
                          thresh_img.astype(np.uint8), background=0, return_num=True)
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


OmeTiffWriter.save(thresh_img.astype(np.uint8), 'C:/Users/Aaron/Documents/Python Scripts/temp/tresh.ome.tiff')














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
    thresh_img, globalimg = MO(structure_img_smooth[:,:,:], global_thresh_method='tri', object_minArea=3000, local_adjust = 0.89, return_object = True)

    # detect if there's more than one object in the thresholded image
    im_labeled, n_labels = skimage.measure.label(
                              thresh_img.astype(np.uint8), background=0, return_num=True)
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