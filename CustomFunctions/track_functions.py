# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 11:01:43 2021

@author: Aaron
"""


import numpy as np
import pandas as pd
# A whole bunch of skimage stuff
import skimage.filters
import skimage.filters.rank
import skimage.io
import skimage.morphology
import skimage.transform


from aicssegmentation.core.utils import hole_filling
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_slice_by_slice, image_smoothing_gaussian_3d
from skimage.morphology import remove_small_objects    
from aicssegmentation.core.MO_threshold import MO


from aicsimageio.readers.tiff_reader import TiffReader
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader
from CustomFunctions.segment_cells2short import MM_slicetostack_reader



def twodholefill(thresh, hole_min, hole_max):
    YZ = thresh.swapaxes(0,2)
    YZ_fill = hole_filling(YZ, hole_min, hole_max, fill_2d=True)
    XZ = thresh.swapaxes(0,1)
    XZ_fill = hole_filling(XZ, hole_min, hole_max, fill_2d=True)
    both_fill = np.logical_or(XZ_fill.swapaxes(1, 0), YZ_fill.swapaxes(2,0))
    filled = hole_filling(both_fill, hole_min, hole_max, fill_2d=True)
    return filled




### nuclear thresholding and tracking from Nathan
def bebi103_thresh(im, selem, white_true=True, k_range=(0.5, 1.5),
                   min_size=100):
    """
    Author Justin Bois, @Caltech.
    Threshold image.  Morphological mean filter is
    applied using selem.
    """
    # Determine comparison operator
    if white_true:
        compare = np.greater
        sign = -1
    else:
        compare = np.less
        sign = 1

    # Do the mean filter
    im_mean = skimage.filters.rank.mean(im, selem)

    # Compute number of pixels in binary image as a function of k
    k = np.linspace(k_range[0], k_range[1], 100)
    n_pix = np.empty_like(k)
    for i in range(len(k)):
        n_pix[i] = compare(im, k[i] * im_mean).sum()

    # Compute rough second derivative
    dn_pix_dk2 = np.diff(np.diff(n_pix))

    # Find index of maximal second derivative
    max_ind = np.argmax(sign * dn_pix_dk2)

    # Use this index to set k
    k_opt = k[max_ind - sign * 2]

    # Threshold with this k
    im_bw = compare(im, k_opt * im_mean)

    # Remove all the small objects
    im_bw = skimage.morphology.remove_small_objects(im_bw, min_size=min_size)

    return im_bw, k_opt


def segment_cells(t, num_z, filelist_fl, im_id, threshold, ip, step):
    print(t)
    df = pd.DataFrame()
    count = 0

    im_labeled_temp = np.zeros((1024,1024))
    for z in np.arange(0,num_z):
        # print(fname, ' : ', t, ' z: ', z)
        im_fname_405 = im_id[0] + str(t+im_id[4]).zfill(im_id[1])  + im_id[2] + str(z+im_id[4]).zfill(im_id[3]) + '.tif'
        
        #  basic threshold to remove background
        im_temp = skimage.io.imread(im_fname_405)
        im_temp[im_temp <= threshold] = 0
        
        # Make the structuring element 50 pixel radius disk
        selem = skimage.morphology.disk(50)

        # Threshhold based on mean filter
        im_bw, k = bebi103_thresh(im_temp, selem, white_true=True, min_size=50)
        # Label binary image; backward kwarg says value in im_bw to consider backgr.
        im_labeled, n_labels = skimage.measure.label(
                                  im_bw, background=0, return_num=True)

        # Get properties
        im_props = skimage.measure.regionprops(im_labeled)

        for i, prop in enumerate(im_props):
            if im_labeled_temp[int(prop.centroid[0]),int(prop.centroid[1])] == 0:
                x = (prop.centroid[1])*ip
                y = (prop.centroid[0])*ip

                # find z by a weighted average of signal intensity across stacks and area that likely has entire nuclei
                zframes = num_z - z

                z_sum = np.zeros(13)
                for k in range(0,np.min([zframes,13])):
                    im_fname_405_ = im_id[0] + str(t+im_id[4]).zfill(im_id[1])  + im_id[2] + str(z+im_id[4] + k).zfill(im_id[3]) + '.tif'
                    im = np.zeros([1024+30,1024+30])
                    im[15:-15, 15:-15] = skimage.io.imread(im_fname_405_)
                    im[im <= threshold] = 0
                    z_sum[k] = im[(15+int(prop.centroid[0])-15):(15+int(prop.centroid[0])+15), (15+int(prop.centroid[1])-15):(15+int(prop.centroid[1])+15)].sum().sum()
                z_max = (np.arange(13)*z_sum).sum()/z_sum.sum()
                if np.min([zframes,13]) == 1:
                        z_max = 1.0
                z_pos = step*(float(z) + z_max)

                # append data to df
                data = {'cell':count, 'frame':t, 'x':x, 'y':y, 'z':z_pos}
                df = df.append(data, ignore_index=True)

                count += 1
        # make temp make to use for comparing identified objects in next time point
        im_labeled_temp = im_labeled.copy()


    return df.values.tolist()




def segment_cells_wholeim(t, im_temp_whole, threshold, ip, step):
    print(t)
    df = pd.DataFrame()
    count = 0


    im_temp_nuc = im_temp_whole[:,0,:,:,:]
    
    
    im_labeled_temp = np.zeros(im_temp_nuc.shape[-2:])

    for z in np.arange(0,im_temp_nuc.shape[-3]):
        
        #  basic threshold to remove background
        im_temp = im_temp_nuc[t,z,:,:]

        im_temp[im_temp <= threshold] = 0
        
        # Make the structuring element 50 pixel radius disk
        selem = skimage.morphology.disk(50)

        # Threshhold based on mean filter
        im_bw, k = bebi103_thresh(im_temp, selem, white_true=True, min_size=50)
        # Label binary image; backward kwarg says value in im_bw to consider backgr.
        im_labeled, n_labels = skimage.measure.label(
                                  im_bw, background=0, return_num=True)

        # Get properties
        im_props = skimage.measure.regionprops(im_labeled)

        for i, prop in enumerate(im_props):
            if im_labeled_temp[int(prop.centroid[0]),int(prop.centroid[1])] == 0:
                x = (prop.centroid[1])*ip
                y = (prop.centroid[0])*ip

                # find z by a weighted average of signal intensity across stacks and area that likely has entire nuclei
                zframes = im_temp_nuc.shape[-3] - z

                z_sum = np.zeros(13)
                for k in range(0,np.min([zframes,13])):
                    im = np.zeros(np.array((im_temp_nuc.shape[-2:]))+30)
                    im[15:-15, 15:-15] = im_temp_nuc[t,z,:,:]
                    im[im <= threshold] = 0
                    z_sum[k] = im[(15+int(prop.centroid[0])-15):(15+int(prop.centroid[0])+15), (15+int(prop.centroid[1])-15):(15+int(prop.centroid[1])+15)].sum().sum()
                z_max = (np.arange(13)*z_sum).sum()/z_sum.sum()
                if np.min([zframes,13]) == 1:
                        z_max = 1.0
                z_pos = step*(float(z) + z_max)

                # append data to df
                data = {'cell':count, 'frame':t, 'x':x, 'y':y, 'z':z_pos}
                df = df.append(data, ignore_index=True)

                count += 1
        # make temp make to use for comparing identified objects in next time point
        im_labeled_temp = im_labeled.copy()


    return df.values.tolist()



def tracking_track(df):
    '''
    Uses the positional information to track nuclei across time points.  The
    approach matches nuclei/cells from one frame to the next by essentially
    minimizing the total displacement across the set of nuclei. For example,
    the nuclei that is at position (x,y,z) in time t+1 that is closest to (x,y,z)
    at time t is most likely the same nuclei cell. Cell 'identity' from t to t+ 1
    is determined by first matching the closest cells (e.g. a dead cell that doesn't move 
    will be matched first and removed from further consideration).  
    
    This approach to tracking requires either the density
    to be sparse enough or acquisition time to be small enough that cells will not cross paths.

    Parameters
    ----------
    df: pandas DataFrame
        DataFrame with columns of 'cell', 'frame', 'x', 'y', and 'z' positions.

    Returns
    -------
    df: pandas DataFrame
        DataFrame with columns corresponding to the reordered and matched cell index
        (i.e. cell 1 across all time points should correspond to same cell/nuclei),
        time point, x position, y position, and  z position.
    '''

    df_track = df[df.frame==0].sort_values(by=['cell'])
    num_cells = len(df_track)

    for t in np.arange(1,len(df.frame.unique())):
        disp_arr = []

        num_cells = len(df_track[df_track.frame==t-1].cell.unique())
        df_tminus = df_track[df_track.frame==t-1].sort_values(by=['cell'])

        for cell in df[df.frame==t].cell.unique():
#             print(t, cell)
            disp_arr_temp = np.zeros(3+num_cells)
#             print(df[(df.cell==cell) & (df.frame==t)].x)
            disp_arr_temp[0] = df[(df.cell==cell) & (df.frame==t)].x
            disp_arr_temp[1] = df[(df.cell==cell) & (df.frame==t)].y
            disp_arr_temp[2] = df[(df.cell==cell) & (df.frame==t)].z
            disp_arr_temp[3:] = np.sqrt((df_tminus['x'] - disp_arr_temp[0])**2 + \
                           (df_tminus['y'] - disp_arr_temp[1])**2 + \
                           (df_tminus['z'] -  disp_arr_temp[2])**2).values

            disp_arr = np.append(disp_arr,disp_arr_temp)

        # reshape array to correct size (columns: i, x, y, number of cells considered; rows: number of items considered)
        disp_arr = disp_arr.reshape(int(len(disp_arr)/(3+len(df_tminus))),3+len(df_tminus))

        # note that I should sort such that I assign closest objects first! Lets try.
        disp_arr_sorted = np.min(disp_arr[:,3:].copy(),axis=0)
        disp_arr_sorted_ind = np.argsort(disp_arr_sorted)

        for cell in disp_arr_sorted_ind:
            # look for an objects that are close to each other between this and prior time point
            if  disp_arr[:,3+cell].min() <= 30.0:
                disp_ind = np.where(disp_arr[:,3+cell] == disp_arr[:,3+cell].min())[0][0]

                # Here I could consider checking the intensity values for +/- a couple z values
                # in actual image and pick z with highest intensity value.
                x_pos = disp_arr[disp_ind,0]
                y_pos = disp_arr[disp_ind,1]
                z_pos = disp_arr[disp_ind,2]

                data = df[(df.frame==t) & (df.cell==disp_ind)]
                data = data.replace({'cell': disp_ind}, cell)
                df_track = df_track.append(data, ignore_index=True)

                # 'remove' object/cell that has been assigned from the current
                # array of objects, by making it infinitely far away
                disp_arr[disp_ind,3:] = np.inf

        # for any cell from the previous time point which wasn't assigned, assume
        # it was lost (i.e. went out of frame)
        for cell in np.arange(0,num_cells):
            if cell not in df_track[df_track.frame==t].cell.unique():

                data = df_track[(df_track.frame==t-1) & (df_track.cell==cell)].copy()
                # print(data)
                data = data.replace({'frame': t-1, 'cell': cell, 'x':data.at[data.index[0],'x'], 'y':data.at[data.index[0],'y'], 'z':data.at[data.index[0],'z']}, 
                                     {'frame': t, 'cell': cell, 'x':np.inf, 'y':np.inf, 'z':np.inf})
                #data = data.replace({'frame': {t-1:t}, 'x':{data.at[data.index[0],'x']:np.inf}, 'y':{data.at[data.index[0],'y']:np.inf}, 'z':{data.at[data.index[0],'z']:np.inf}})

                df_track = df_track.append(data, ignore_index=True)

        count = 0
        for prop_ind in np.arange(0,len(disp_arr[:,0])):
            if 30.01 <= disp_arr[prop_ind,3:].min() <= 10000.0: # upper bound due to my use of np.inf
                x_pos = disp_arr[prop_ind,0]
                y_pos = disp_arr[prop_ind,1]
                z_pos = disp_arr[prop_ind,2]

                data = df[(df.frame==t) & (df.cell==prop_ind)]
                
                data = data.replace({'cell': prop_ind}, num_cells + count)
                df_track = df_track.append(data, ignore_index=True)
                count += 1

    return df_track


def tracking_label(df, analysis_date, framerate, date, exp,
        scope, obj, trial, fmlp, efield,
        media = ' ', misc = ' '):
    '''
    Append additional experimental information to Pandas DataFrame.

    Parameters
    ----------
    df: pandas DataFrame
        DataFrame with columns of positional information
    analysis_date: str
        Record of what date analysis was run.
    framerate: int
        Framerate of image aquisition in seconds
    date: int
        Date of experimental work.
    celltype: str
        Details of cell line/ fish used
    scope: str
        Microscope used to collect data.
    obj: str
        Objective used.
    efield: str
        Electric field details.
    media: str
        Media information (that cells are in during imaging.)
    misc: str
        Any additional notes  worth keeping record of.
    Returns
    -------
    df: pandas DataFrame
        DataFrame with additional information appended.
    '''

    df_temp  = pd.DataFrame()

    for t in np.arange(0,len(df.frame.unique())):
        for cell in df[df.frame==t].cell.unique():
            data = {'cell':cell, 'frame':t, 'framerate':framerate,
                    'date':date, 'experiment_detail':exp, 
                    'scope':scope, 'magnification':obj, 'trial':trial, 
                    'media': media, 'misc': misc, 'analysis_date':analysis_date}
            df_temp = df_temp.append(data, ignore_index=True)

    # append the details to the main DataFrame
    return pd.merge(df, df_temp, on=['cell','frame'])

#FILE_NAME = 'C:/Users/Aaron/Documents/PythonScripts/Data/20210406/20200406_Hoechst_CAAXJF647_30C_15s_1/20200406_Hoechst_CAAXJF647_30C_15s_1_MMStack_Pos0.ome.tif'
#im_temp_whole = skimage.io.imread(FILE_NAME)



def segment_cells_wholeimtwo(t, im_temp_whole, threshold, ip, step):
    print(t)
    df = pd.DataFrame()
    count = 0

    im_temp_nuc = im_temp_whole[t,0,:,:,:]

    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [1, 15]
    gaussian_smoothing_sigma = 1
    ################################
    # intensity normalization
    struct_img = intensity_normalization(im_temp_nuc, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter slice by slice 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)

    
    # step 1: Masked-Object (MO) Thresholding
    thresh_img, object_for_debug = MO(structure_img_smooth[:,:,:], global_thresh_method='tri', object_minArea=1200, local_adjust = 1.2, return_object=True)

    # fill in the holes
    hole_max = 500
    hole_min = 2
    seg = hole_filling(thresh_img, hole_min, hole_max)
    
    ################################
    ## PARAMETERS for this step ##
    minArea = 35
    ################################
    #combine the two segmentations
    # create segmentation mask               
    seg = seg > 0

    seg = remove_small_objects(seg>0, min_size=minArea, connectivity=1, in_place=False)

    
    im_labeled_temp = np.zeros(im_temp_nuc.shape[-2:])

    for z in np.arange(0,im_temp_nuc.shape[-3]):
        
        #  basic threshold to remove background
        im_bw = seg[z,:,:]


        # Label binary image; backward kwarg says value in im_bw to consider backgr.
        im_labeled, n_labels = skimage.measure.label(
                                  im_bw, background=0, return_num=True)

        # Get properties
        im_props = skimage.measure.regionprops(im_labeled)

        for i, prop in enumerate(im_props):
            if im_labeled_temp[int(prop.centroid[0]),int(prop.centroid[1])] == 0:
                x = (prop.centroid[1])*ip
                y = (prop.centroid[0])*ip

                # find z by a weighted average of signal intensity across stacks and area that likely has entire nuclei
                zframes = im_temp_nuc.shape[-3] - z

                z_sum = np.zeros(13)
                for k in range(0,np.min([zframes,13])):
                    im = np.zeros(np.array((im_temp_nuc.shape[-2:]))+30)
                    im[15:-15, 15:-15] = im_temp_nuc[z,:,:]
                    z_sum[k] = im[(15+int(prop.centroid[0])-15):(15+int(prop.centroid[0])+15), (15+int(prop.centroid[1])-15):(15+int(prop.centroid[1])+15)].sum().sum()
                z_max = (np.arange(13)*z_sum).sum()/z_sum.sum()
                if np.min([zframes,13]) == 1:
                        z_max = 1.0
                z_pos = step*(float(z) + z_max)

                # append data to df
                data = {'cell':count, 'frame':t, 'x':x, 'y':y, 'z':z_pos}
                df = df.append(data, ignore_index=True)

                count += 1
        # make temp make to use for comparing identified objects in next time point
        im_labeled_temp = im_labeled.copy()


    return df.values.tolist()




def segment_caax_tracks(struct_img0, ip, step, channel, frame):
    df = pd.DataFrame()
    count = 0

    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [1, 10]
    gaussian_smoothing_sigma = 2
    ################################
    # intensity normalization
    struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_slice_by_slice(struct_img, sigma=gaussian_smoothing_sigma)
    # step 1: Masked-Object (MO) Thresholding
    thresh_img, object_for_debug = MO(structure_img_smooth[:,:,:], global_thresh_method='tri', object_minArea=1200, return_object=True)
    # #get objects
    # im_labeled, n_labels = skimage.measure.label(thresh_img, background=0, return_num=True)
    # print(n_labels)

    # # fill in the holes
    # hole_max = 2000
    # hole_min = 1
    # thresh_img = hole_filling(thresh_img, hole_min, hole_max)

    #do 2d hole fill
    hole_max = 1500
    hole_min = 1
    thresh_img = twodholefill(thresh_img, hole_min, hole_max)

    minArea = 3000
    seg = thresh_img > 0
    seg = remove_small_objects(seg>0, min_size=minArea, connectivity=1, in_place=False)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255

    #get objects
    im_labeled, n_labels = skimage.measure.label(seg, background=0, return_num=True)
    print(n_labels)
    im_props = skimage.measure.regionprops(im_labeled)

    
    for count, prop in enumerate(im_props):
        thebox = prop.bbox
        cent = prop.centroid
        solidity = prop.solidity
        area = prop.area
        convex_area = prop.convex_area
        extent = prop.extent
        major_axis_length = prop.major_axis_length
        minor_axis_length = prop.minor_axis_length
        
        
        data = {'cell':count, 'frame':frame, 'z_min':thebox[0], 'y_min':thebox[1], 
                'x_min':thebox[2],'z_max':thebox[3], 'y_max':thebox[4], 'x_max':thebox[5],
               'z':cent[0]*step, 'y':cent[1]*ip, 'x': cent[2]*ip, 'z_range': struct_img0.shape[-3],
               'solidity':solidity, 'area':area, 'convex_area':convex_area, 'extent':extent,
               'minor_axis_length':minor_axis_length, 'major_axis_length':major_axis_length}
        df = df.append(data, ignore_index=True)
            

    return df.values.tolist(), seg, frame


def segment_caax_tracks_iSIM(struct_img0, ip, step, frame):

    df = pd.DataFrame()
    count = 0

    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [1.5, 5.5]
    gaussian_smoothing_sigma = 2
    ################################
    # intensity normalization
    struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    # step 1: Masked-Object (MO) Thresholding
    thresh_img, object_for_debug = MO(structure_img_smooth[:,:,:], global_thresh_method='tri', object_minArea=4000, return_object=True)

    # fill in the holes
    hole_max = 15000
    hole_min = 1
    thresh_img = hole_filling(thresh_img, hole_min, hole_max)

    minArea = 4000
    seg = thresh_img > 0
    seg = remove_small_objects(seg, min_size=minArea, connectivity=1, in_place=False)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255

    #get objects
    im_labeled, n_labels = skimage.measure.label(seg, background=0, return_num=True)
    print(n_labels)
    im_props = skimage.measure.regionprops(im_labeled)

    
    for count, prop in enumerate(im_props):
        thebox = prop.bbox
        cent = prop.centroid
        solidity = prop.solidity
        area = prop.area
        convex_area = prop.convex_area
        extent = prop.extent
        major_axis_length = prop.major_axis_length
        minor_axis_length = prop.minor_axis_length
        
        
        data = {'cell':count, 'frame':frame, 'z_min':thebox[0], 'y_min':thebox[1], 
                'x_min':thebox[2],'z_max':thebox[3], 'y_max':thebox[4], 'x_max':thebox[5],
               'z':cent[0]*step, 'y':cent[1]*ip, 'x': cent[2]*ip, 'z_range': struct_img0.shape[-3],
               'solidity':solidity, 'area':area, 'convex_area':convex_area, 'extent':extent,
               'minor_axis_length':minor_axis_length, 'major_axis_length':major_axis_length}
        df = df.append(data, ignore_index=True)
            

    return df.values.tolist(), seg, frame



def segment_caax_tracks_iSIM_visiview(imname, ip, step, frame):
    img = TiffReader(imname).data
    df = pd.DataFrame()

    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [1.5, 5.5]
    gaussian_smoothing_sigma = 2
    ################################
    # intensity normalization
    struct_img = intensity_normalization(img, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    # step 1: Masked-Object (MO) Thresholding
    thresh_img, object_for_debug = MO(structure_img_smooth, global_thresh_method='tri', object_minArea=4000, return_object=True)

    # fill in the holes
    hole_max = 15000
    hole_min = 1
    thresh_img = hole_filling(thresh_img, hole_min, hole_max)

    minArea = 4000
    seg = thresh_img > 0
    seg = remove_small_objects(seg, min_size=minArea, connectivity=1, in_place=False)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255


    #get objects
    im_labeled, n_labels = skimage.measure.label(seg, background=0, return_num=True)
    im_props = skimage.measure.regionprops(im_labeled)

    
    for count, prop in enumerate(im_props):
        thebox = prop.bbox
        cent = prop.centroid
        area = prop.area
        convex_area = prop.convex_area
        extent = prop.extent
        major_axis_length = prop.major_axis_length
        minor_axis_length = prop.minor_axis_length
        
        
        data = {'cell':count, 'frame':frame, 'z_min':thebox[0], 'y_min':thebox[1], 
                'x_min':thebox[2],'z_max':thebox[3], 'y_max':thebox[4], 'x_max':thebox[5],
               'z':cent[0]*step, 'y':cent[1]*ip, 'x': cent[2]*ip, 'z_range': img.shape[-3],
               'area':area, 'convex_area':convex_area, 'extent':extent,
               'minor_axis_length':minor_axis_length, 'major_axis_length':major_axis_length}
        df = df.append(data, ignore_index=True)
            

    return df.values.tolist(), seg, frame, img.shape


def segment_caax_tracks_iSIM_visiview_halfsize(imname, ip, step, frame):
    img = TiffReader(imname).data

    rescaled = skimage.transform.rescale(img,0.5, preserve_range=True)
    
    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [1, 5]
    # intensity_scaling_param = [1200]
    gaussian_smoothing_sigma = 1
    ################################
    # intensity normalization
    struct_img = intensity_normalization(rescaled, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    # step 1: Masked-Object (MO) Thresholding
    thresh_img = MO(structure_img_smooth, local_adjust=0.92, global_thresh_method='tri', object_minArea=2000)


    # fill in the holes
    hole_max = 5000
    hole_min = 1
    thresh_img = hole_filling(thresh_img, hole_min, hole_max)

    minArea = 100
    seg = thresh_img > 0
    seg = remove_small_objects(seg, min_size=minArea, connectivity=1)
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



def segment_caax_tracks_iSIM_20x(struct_img0, ip, step, channel, frame):

    df = pd.DataFrame()
    count = 0

    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [1.5, 5.5]
    gaussian_smoothing_sigma = 2
    ################################
    # intensity normalization
    struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    # step 1: Masked-Object (MO) Thresholding
    thresh_img, object_for_debug = MO(structure_img_smooth[:,:,:], global_thresh_method='tri', object_minArea=3000, return_object=True)

    # fill in the holes
    hole_max = 3000
    hole_min = 1
    thresh_img = hole_filling(thresh_img, hole_min, hole_max)

    minArea = 5000
    seg = thresh_img > 0
    seg = remove_small_objects(seg, min_size=minArea, connectivity=1, in_place=False)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255

    #get objects
    im_labeled, n_labels = skimage.measure.label(seg, background=0, return_num=True)
    print(n_labels)
    im_props = skimage.measure.regionprops(im_labeled)

    
    for count, prop in enumerate(im_props):
        thebox = prop.bbox
        cent = prop.centroid
        solidity = prop.solidity
        area = prop.area
        convex_area = prop.convex_area
        extent = prop.extent
        major_axis_length = prop.major_axis_length
        minor_axis_length = prop.minor_axis_length
        
        
        data = {'cell':count, 'frame':frame, 'z_min':thebox[0], 'y_min':thebox[1], 
                'x_min':thebox[2],'z_max':thebox[3], 'y_max':thebox[4], 'x_max':thebox[5],
               'z':cent[0]*step, 'y':cent[1]*ip, 'x': cent[2]*ip, 'z_range': struct_img0.shape[-3],
               'solidity':solidity, 'area':area, 'convex_area':convex_area, 'extent':extent,
               'minor_axis_length':minor_axis_length, 'major_axis_length':major_axis_length}
        df = df.append(data, ignore_index=True)
            

    return df.values.tolist(), seg, frame

def segment_nuc_tracks_iSIM(struct_img0, ip, step, channel, frame):

    df = pd.DataFrame()
    count = 0

    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [1.5, 5.5]
    gaussian_smoothing_sigma = 2
    ################################
    # intensity normalization
    struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    # step 1: Masked-Object (MO) Thresholding
    thresh_img, object_for_debug = MO(structure_img_smooth[:,:,:], global_thresh_method='tri', object_minArea=2000, return_object=True)

    # fill in the holes
    hole_max = 15000
    hole_min = 1
    thresh_img = hole_filling(thresh_img, hole_min, hole_max)

    minArea = 3000
    seg = thresh_img > 0
    seg = remove_small_objects(seg, min_size=minArea, connectivity=1, in_place=False)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255

    #get objects
    im_labeled, n_labels = skimage.measure.label(seg, background=0, return_num=True)
    print(n_labels)
    im_props = skimage.measure.regionprops(im_labeled)

    
    for count, prop in enumerate(im_props):
        thebox = prop.bbox
        cent = prop.centroid
        solidity = prop.solidity
        area = prop.area
        convex_area = prop.convex_area
        extent = prop.extent
        major_axis_length = prop.major_axis_length
        minor_axis_length = prop.minor_axis_length
        
        
        data = {'cell':count, 'frame':frame, 'z_min':thebox[0], 'y_min':thebox[1], 
                'x_min':thebox[2],'z_max':thebox[3], 'y_max':thebox[4], 'x_max':thebox[5],
               'z':cent[0]*step, 'y':cent[1]*ip, 'x': cent[2]*ip, 'z_range': struct_img0.shape[-3],
               'solidity':solidity, 'area':area, 'convex_area':convex_area, 'extent':extent,
               'minor_axis_length':minor_axis_length, 'major_axis_length':major_axis_length}
        df = df.append(data, ignore_index=True)
            

    return df.values.tolist(), seg, frame


def segment_nuc_tracks_iSIM_iXON_20x(struct_img0, ip, step, frame):

    df = pd.DataFrame()
    count = 0

    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [1.5, 5.5]
    gaussian_smoothing_sigma = 2.5
    ################################
    # intensity normalization
    struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    # step 1: Masked-Object (MO) Thresholding
    thresh_img, object_for_debug = MO(structure_img_smooth[:,:,:], global_thresh_method='tri', object_minArea=500, return_object=True)

    # fill in the holes
    hole_max = 500
    hole_min = 1
    thresh_img = hole_filling(thresh_img, hole_min, hole_max)

    minArea = 500
    seg = thresh_img > 0
    seg = remove_small_objects(seg, min_size=minArea, connectivity=1, in_place=False)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255

    #get objects
    im_labeled, n_labels = skimage.measure.label(seg, background=0, return_num=True)
    print(n_labels)
    im_props = skimage.measure.regionprops(im_labeled)

    
    for count, prop in enumerate(im_props):
        thebox = prop.bbox
        cent = prop.centroid
        solidity = prop.solidity
        area = prop.area
        convex_area = prop.convex_area
        extent = prop.extent
        major_axis_length = prop.major_axis_length
        minor_axis_length = prop.minor_axis_length
        
        
        data = {'cell':count, 'frame':frame, 'z_min':thebox[0], 'y_min':thebox[1], 
                'x_min':thebox[2],'z_max':thebox[3], 'y_max':thebox[4], 'x_max':thebox[5],
               'z':cent[0]*step, 'y':cent[1]*ip, 'x': cent[2]*ip, 'z_range': struct_img0.shape[-3],
               'solidity':solidity, 'area':area, 'convex_area':convex_area, 'extent':extent,
               'minor_axis_length':minor_axis_length, 'major_axis_length':major_axis_length}
        df = df.append(data, ignore_index=True)
            

    return df.values.tolist(), seg, frame



def segment_caax_tracks_confocal_40x(struct_img0, ip, step, frame):
    
    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [0]
    gaussian_smoothing_sigma = 1.5
    ################################
    # intensity normalization
    struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_slice_by_slice(struct_img, sigma=gaussian_smoothing_sigma)
    # step 1: Masked-Object (MO) Thresholding
    thresh_img = MO(structure_img_smooth, global_thresh_method='tri', object_minArea=1200, local_adjust=0.99)
    
    
    #do 2d hole fill
    hole_max = 1000
    hole_min = 1
    thresh_img = twodholefill(thresh_img, hole_min, hole_max)
    
    minArea = 2000
    seg = thresh_img > 0
    seg = remove_small_objects(seg>0, min_size=minArea, connectivity=1, in_place=False)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255
    
    #get objects
    im_labeled, n_labels = skimage.measure.label(seg, background=0, return_num=True)
    print(n_labels)
    im_props = skimage.measure.regionprops(im_labeled)
    
    df = pd.DataFrame()
    for count, prop in enumerate(im_props):
        thebox = prop.bbox
        cent = prop.centroid
        area = prop.area
        convex_area = prop.convex_area
        extent = prop.extent
        major_axis_length = prop.major_axis_length
        minor_axis_length = prop.minor_axis_length
        
        
        data = {'cell':count, 'frame':frame, 'z_min':thebox[0], 'y_min':thebox[1], 
                'x_min':thebox[2],'z_max':thebox[3], 'y_max':thebox[4], 'x_max':thebox[5],
               'z':cent[0]*step, 'y':cent[1]*ip, 'x': cent[2]*ip, 'z_range': struct_img0.shape[-3],
                'area':area, 'convex_area':convex_area, 'extent':extent,
               'minor_axis_length':minor_axis_length, 'major_axis_length':major_axis_length}
        df = df.append(data, ignore_index=True)
            
    
    return df.values.tolist(), seg, frame



def segment_caax_tracks_confocal_40x_fromsingle(imname, shape, ip, step, frame):
    #read image
    img = MM_slicetostack_reader(imname,frame,shape, range(0,shape[-3]))
    #shrink image by half
    rescaled = skimage.transform.rescale(img,0.5, preserve_range=True)
    
    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [1,5]
    # intensity_scaling_param = [1200]
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
