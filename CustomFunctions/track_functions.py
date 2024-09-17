# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 11:01:43 2021

@author: Aaron
"""


import numpy as np
import pandas as pd
# A whole bunch of skimage stuff
import skimage
import skimage.feature
import skimage.filters
import skimage.filters.rank
import skimage.io
import skimage.exposure
import skimage.morphology
import skimage.restoration
import skimage.segmentation
import skimage.transform

from typing import List
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects, ball, dilation
from skimage.filters import threshold_triangle, threshold_otsu
from skimage.measure import label
from aicsimageio.readers.tiff_reader import TiffReader
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader
from CustomFunctions.segment_cells2short import MM_slicetostack_reader


def hole_filling(
    bw: np.ndarray, hole_min: int, hole_max: int, fill_2d: bool = True
) -> np.ndarray:
    """Fill holes in 2D/3D segmentation

    Parameters:
    -------------
    bw: np.ndarray
        a binary 2D/3D image.
    hole_min: int
        the minimum size of the holes to be filled
    hole_max: int
        the maximum size of the holes to be filled
    fill_2d: bool
        if fill_2d=True, a 3D image will be filled slice by slice.
        If you think of a hollow tube alone z direction, the inside
        is not a hole under 3D topology, but the inside on each slice
        is indeed a hole under 2D topology.

    Return:
        a binary image after hole filling
    """
    bw = bw > 0
    if len(bw.shape) == 2:
        background_lab = label(~bw, connectivity=1)
        fill_out = np.copy(background_lab)
        component_sizes = np.bincount(background_lab.ravel())
        too_big = component_sizes > hole_max
        too_big_mask = too_big[background_lab]
        fill_out[too_big_mask] = 0
        too_small = component_sizes < hole_min
        too_small_mask = too_small[background_lab]
        fill_out[too_small_mask] = 0
    elif len(bw.shape) == 3:
        if fill_2d:
            fill_out = np.zeros_like(bw)
            for zz in range(bw.shape[0]):
                background_lab = label(~bw[zz, :, :], connectivity=1)
                out = np.copy(background_lab)
                component_sizes = np.bincount(background_lab.ravel())
                too_big = component_sizes > hole_max
                too_big_mask = too_big[background_lab]
                out[too_big_mask] = 0
                too_small = component_sizes < hole_min
                too_small_mask = too_small[background_lab]
                out[too_small_mask] = 0
                fill_out[zz, :, :] = out
        else:
            background_lab = label(~bw, connectivity=1)
            fill_out = np.copy(background_lab)
            component_sizes = np.bincount(background_lab.ravel())
            too_big = component_sizes > hole_max
            too_big_mask = too_big[background_lab]
            fill_out[too_big_mask] = 0
            too_small = component_sizes < hole_min
            too_small_mask = too_small[background_lab]
            fill_out[too_small_mask] = 0
    else:
        print("error in image shape")
        return

    return np.logical_or(bw, fill_out)


def twodholefill(thresh, hole_min, hole_max):
    YZ = thresh.swapaxes(0,2)
    YZ_fill = hole_filling(YZ, hole_min, hole_max, fill_2d=True)
    XZ = thresh.swapaxes(0,1)
    XZ_fill = hole_filling(XZ, hole_min, hole_max, fill_2d=True)
    both_fill = np.logical_or(XZ_fill.swapaxes(1, 0), YZ_fill.swapaxes(2,0))
    filled = hole_filling(both_fill, hole_min, hole_max, fill_2d=True)
    return filled


def intensity_normalization(struct_img: np.ndarray, scaling_param: List):
    """Normalize the intensity of input image so that the value range is from 0 to 1.

    Parameters:
    ------------
    img: np.ndarray
        a 3d image
    scaling_param: List
        a list with only one value 0, i.e. [0]: Min-Max normlaizaiton,
            the max intensity of img will be mapped to 1 and min will
            be mapped to 0
        a list with a single positive integer v, e.g. [5000]: Min-Max normalization,
            but first any original intensity value > v will be considered as outlier
            and reset of min intensity of img. After the max will be mapped to 1
            and min will be mapped to 0
        a list with two float values [a, b], e.g. [1.5, 10.5]: Auto-contrast
            normalizaiton. First, mean and standard deviaion (std) of the original
            intensity in img are calculated. Next, the intensity is truncated into
            range [mean - a * std, mean + b * std], and then recaled to [0, 1]
        a list with four float values [a, b, c, d], e.g. [0.5, 15.5, 200, 4000]:
            Auto-contrast normalization. Similat to above case, but only intensity value
            between c and d will be used to calculated mean and std.
    """
    assert len(scaling_param) > 0

    if len(scaling_param) == 1:
        if scaling_param[0] < 1:
            print(
                "intensity normalization: min-max normalization with NO absolute"
                + "intensity upper bound"
            )
        else:
            print(f"intensity norm: min-max norm with upper bound {scaling_param[0]}")
            struct_img[struct_img > scaling_param[0]] = struct_img.min()
        strech_min = struct_img.min()
        strech_max = struct_img.max()
        struct_img = (struct_img - strech_min + 1e-8) / (strech_max - strech_min + 1e-8)
    elif len(scaling_param) == 2:
        m, s = norm.fit(struct_img.flat)
        strech_min = max(m - scaling_param[0] * s, struct_img.min())
        strech_max = min(m + scaling_param[1] * s, struct_img.max())
        struct_img[struct_img > strech_max] = strech_max
        struct_img[struct_img < strech_min] = strech_min
        struct_img = (struct_img - strech_min + 1e-8) / (strech_max - strech_min + 1e-8)
    elif len(scaling_param) == 4:
        img_valid = struct_img[
            np.logical_and(struct_img > scaling_param[2], struct_img < scaling_param[3])
        ]
        m, s = norm.fit(img_valid.flat)
        strech_min = max(scaling_param[2] - scaling_param[0] * s, struct_img.min())
        strech_max = min(scaling_param[3] + scaling_param[1] * s, struct_img.max())
        struct_img[struct_img > strech_max] = strech_max
        struct_img[struct_img < strech_min] = strech_min
        struct_img = (struct_img - strech_min + 1e-8) / (strech_max - strech_min + 1e-8)

    # print('intensity normalization completes')
    return struct_img


def image_smoothing_gaussian_3d(struct_img, sigma, truncate_range=3.0):
    """
    wrapper for 3D Guassian smoothing
    """

    structure_img_smooth = gaussian_filter(
        struct_img, sigma=sigma, mode="nearest", truncate=truncate_range
    )

    return structure_img_smooth


def image_smoothing_gaussian_slice_by_slice(struct_img, sigma, truncate_range=3.0):
    """
    wrapper for applying 2D Guassian smoothing slice by slice on a 3D image
    """
    structure_img_smooth = np.zeros_like(struct_img)
    for zz in range(struct_img.shape[0]):
        structure_img_smooth[zz, :, :] = gaussian_filter(
            struct_img[zz, :, :], sigma=sigma, mode="nearest", truncate=truncate_range
        )

    return structure_img_smooth


def edge_preserving_smoothing_3d(
    struct_img: np.ndarray,
    numberOfIterations: int = 10,
    conductance: float = 1.2,
    timeStep: float = 0.0625,
    spacing: List = [1, 1, 1],
):
    """perform edge preserving smoothing on a 3D image

    Parameters:
    -------------
    struct_img: np.ndarray
        the image to be smoothed
    numberOfInterations: int
        how many smoothing iterations to perform. More iterations give more
        smoothing effect. Default is 10.
    timeStep: float
         the time step to be used for each iteration, important for numberical
         stability. Default is 0.0625 for 3D images. Do not suggest to change.
    spacing: List
        the spacing of voxels in three dimensions. Default is [1, 1, 1]

    Reference:
    -------------
    https://itk.org/Doxygen/html/classitk_1_1GradientAnisotropicDiffusionImageFilter.html
    """
    import itk

    itk_img = itk.GetImageFromArray(struct_img.astype(np.float32))

    # set spacing
    itk_img.SetSpacing(spacing)

    gradientAnisotropicDiffusionFilter = (
        itk.GradientAnisotropicDiffusionImageFilter.New(itk_img)
    )
    gradientAnisotropicDiffusionFilter.SetNumberOfIterations(numberOfIterations)
    gradientAnisotropicDiffusionFilter.SetTimeStep(timeStep)
    gradientAnisotropicDiffusionFilter.SetConductanceParameter(conductance)
    gradientAnisotropicDiffusionFilter.Update()

    itk_img_smooth = gradientAnisotropicDiffusionFilter.GetOutput()

    img_smooth_ag = itk.GetArrayFromImage(itk_img_smooth)

    return img_smooth_ag


def suggest_normalization_param(structure_img0):
    """
    suggest scaling parameter assuming the image is a representative example
    of this cell structure
    """
    m, s = norm.fit(structure_img0.flat)
    print(f"mean intensity of the stack: {m}")
    print(f"the standard deviation of intensity of the stack: {s}")

    p99 = np.percentile(structure_img0, 99.99)
    print(f"0.9999 percentile of the stack intensity is: {p99}")

    pmin = structure_img0.min()
    print(f"minimum intensity of the stack: {pmin}")

    pmax = structure_img0.max()
    print(f"maximum intensity of the stack: {pmax}")

    up_ratio = 0
    for up_i in np.arange(0.5, 1000, 0.5):
        if m + s * up_i > p99:
            if m + s * up_i > pmax:
                print(f"suggested upper range is {up_i-0.5}, which is {m+s*(up_i-0.5)}")
                up_ratio = up_i - 0.5
            else:
                print(f"suggested upper range is {up_i}, which is {m+s*up_i}")
                up_ratio = up_i
            break

    low_ratio = 0
    for low_i in np.arange(0.5, 1000, 0.5):
        if m - s * low_i < pmin:
            print(f"suggested lower range is {low_i-0.5}, which is {m-s*(low_i-0.5)}")
            low_ratio = low_i - 0.5
            break

    print(f"So, suggested parameter for normalization is [{low_ratio}, {up_ratio}]")
    print(
        "To further enhance the contrast: You may increase the first value "
        + "(may loss some dim parts), or decrease the second value"
        + "(may loss some texture in super bright regions)"
    )
    print(
        "To slightly reduce the contrast: You may decrease the first value, or "
        + "increase the second value"
    )





def MO(
    structure_img_smooth: np.ndarray,
    global_thresh_method: str,
    object_minArea: int,
    extra_criteria: bool = False,
    local_adjust: float = 0.98,
    return_object: bool = False,
) -> np.ndarray:
    """
    Implementation of "Masked Object Thresholding" algorithm. Specifically, the
    algorithm is a hybrid thresholding method combining two levels of thresholds.
    The steps are [1] a global threshold is calculated, [2] extract each individual
    connected componet after applying the global threshold, [3] remove small objects,
    [4] within each remaining object, a local Otsu threshold is calculated and applied
    with an optional local threshold adjustment ratio (to make the segmentation more
    and less conservative). An extra check can be used in step [4], which requires the
    local Otsu threshold larger than 1/3 of global Otsu threhsold and otherwise this
    connected component is discarded.

    Parameters:
    --------------
    structure_img_smooth: np.ndarray
        the image (should have already been smoothed) to apply the method on
    global_thresh_method: str
        which method to use for calculating global threshold. Options include:
        "triangle" (or "tri"), "median" (or "med"), and "ave_tri_med" (or "ave").
        "ave" refers the average of "triangle" threshold and "mean" threshold.
    object_minArea: int
        the size filter for excluding small object before applying local threshold
    extra_criteria: bool
        whether to use the extra check when doing local thresholding, default is False
    local_adjust: float
        a ratio to apply on local threshold, default is 0.98
    return_object: bool
        whether to return the global thresholding results in order to obtain the
        individual objects the local thresholding is made on

    Return:
    --------------
    a binary nd array of the segmentation result
    """

    if global_thresh_method == "tri" or global_thresh_method == "triangle":
        th_low_level = threshold_triangle(structure_img_smooth)
    elif global_thresh_method == "med" or global_thresh_method == "median":
        th_low_level = np.percentile(structure_img_smooth, 50)
    elif global_thresh_method == "ave" or global_thresh_method == "ave_tri_med":
        global_tri = threshold_triangle(structure_img_smooth)
        global_median = np.percentile(structure_img_smooth, 50)
        th_low_level = (global_tri + global_median) / 2

    bw_low_level = structure_img_smooth > th_low_level
    bw_low_level = remove_small_objects(
        bw_low_level, min_size=object_minArea, connectivity=1
    )
    bw_low_level = dilation(bw_low_level, selem=ball(2))

    bw_high_level = np.zeros_like(bw_low_level)
    lab_low, num_obj = label(bw_low_level, return_num=True, connectivity=1)
    if extra_criteria:
        local_cutoff = 0.333 * threshold_otsu(structure_img_smooth)
        for idx in range(num_obj):
            single_obj = lab_low == (idx + 1)
            local_otsu = threshold_otsu(structure_img_smooth[single_obj > 0])
            if local_otsu > local_cutoff:
                bw_high_level[
                    np.logical_and(
                        structure_img_smooth > local_otsu * local_adjust, single_obj
                    )
                ] = 1
    else:
        for idx in range(num_obj):
            single_obj = lab_low == (idx + 1)
            local_otsu = threshold_otsu(structure_img_smooth[single_obj > 0])
            bw_high_level[
                np.logical_and(
                    structure_img_smooth > local_otsu * local_adjust, single_obj
                )
            ] = 1

    if return_object:
        return bw_high_level > 0, bw_low_level
    else:
        return bw_high_level > 0





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
    disp_arr = []

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
    disp_arr = []

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
    disp_arr = []

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


def segment_nuc_tracks_confocal_40x(struct_img0, ip, step, channel, frame):

    df = pd.DataFrame()
    count = 0

    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [1.5, 15]
    gaussian_smoothing_sigma = 1.5
    ################################
    # intensity normalization
    struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    # step 1: Masked-Object (MO) Thresholding
    thresh_img, object_for_debug = MO(structure_img_smooth[:,:,:], global_thresh_method='tri', object_minArea=1200, return_object=True)


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


def segment_nuc_tracks(imname, ip, step, channel, frame, ):
    
    
    df = pd.DataFrame()
    im = OmeTiffReader(imname)
    struct_img0 = im.data[frame,channel,:,:,:]
    

    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [1, 15]
    gaussian_smoothing_sigma = 1.5
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
    
    # fill in the holes
    hole_max = 500
    hole_min = 2
    thresh_img_hole = hole_filling(thresh_img, hole_min, hole_max)
    

    minArea = 100
    seg = thresh_img_hole > 0
    # seg = remove_small_objects(seg>0, min_size=minArea, connectivity=1, in_place=False)
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