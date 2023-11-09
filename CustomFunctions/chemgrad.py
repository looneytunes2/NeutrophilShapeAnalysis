# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:48:56 2023

@author: Aaron
"""

import numpy as np
from aicsimageio.readers.tiff_reader import TiffReader
from skimage.filters import gaussian
from skimage.transform import rescale, resize


def process_gradient(time,
                    im, #directory and name of the image to open
                    flat,
                     xyres,
                     zstep,
                    ):
    
    
    
    #open processed flat
    flatim = TiffReader(flat).data
    #open gradient image
    img = TiffReader(im).data
    #expand z of gradient image to match z res with xy
    size = (round(img.shape[-3]*zstep/xyres), img.shape[-1])
    img_ex = np.zeros((size[0], img.shape[-2], img.shape[-1]))
    for i in range(img.shape[-2]):
        img_ex[:,i,:] = resize(img[:,i,:], size, 1)

    ######## find the vectors that define the gradient
    #smoothen image
    smooth = gaussian(img_ex, sigma = 50)
    #correct the gradient with the flat
    corrected = smooth * np.mean(flatim.data)/flatim.data
    #normalize image
    norm = corrected-corrected.min()
    norm = norm/norm.max()
    # #downscale image
    small = rescale(norm, 0.0625)
    #get derivative ie vectors of gradient
    dz, dy, dx = np.gradient(small)
    vec_arr = np.stack([dz, dy, dx])

    return time, corrected, vec_arr, img_ex.shape, small.shape