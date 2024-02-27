# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter




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
    bw_low_level = dilation(bw_low_level, footprint=ball(2))

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




imname = 'C:/Users/Aaron/Data/Raw/Galvanotaxis/20230329/20230329_488GFP-CAAX_640SiR-DNA_10minrandom_20mingalv1/20230329_488GFP-CAAX_640SiR-DNA_10minrandom_20mingalv1_w2GFP-Cy5-UVEpi-Reflected_t11.ome.tif'
ip = 0.1613
step =  0.5
frame = 0



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

    
    data = {'cell':count, 'frame':frame, 'z_min':thebox[0], 'y_min':thebox[1], 
            'x_min':thebox[2],'z_max':thebox[3], 'y_max':thebox[4], 'x_max':thebox[5],
           'z':cent[0]*step, 'y':cent[1]*ip, 'x': cent[2]*ip, 'z_range': img.shape[-3],
           'area':area, 'convex_area':convex_area, 'extent':extent,
           'minor_axis_length':minor_axis_length, 'major_axis_length':major_axis_length}
    dictlist.append(data)
df = pd.DataFrame.from_dict(dictlist)
    

OmeTiffWriter.save(seg, 'C:/Users/Aaron/Data/temp/seg.ome.tif', dim_order='ZYX')



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
    seg = remove_small_objects(seg, min_size=minArea, connectivity=1, in_place=False)
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
    
        
        data = {'cell':count, 'frame':frame, 'z_min':thebox[0], 'y_min':thebox[1], 
                'x_min':thebox[2],'z_max':thebox[3], 'y_max':thebox[4], 'x_max':thebox[5],
               'z':cent[0]*step, 'y':cent[1]*ip, 'x': cent[2]*ip, 'z_range': img.shape[-3],
               'area':area, 'convex_area':convex_area, 'extent':extent,
               'minor_axis_length':minor_axis_length, 'major_axis_length':major_axis_length}
        dictlist.append(data)
        
    df = pd.DataFrame.from_dict(dictlist)
            

    return df.values.tolist(), seg, frame, rescaled.shape