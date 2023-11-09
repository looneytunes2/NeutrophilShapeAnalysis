# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 14:56:20 2023

@author: Aaron
"""


import numpy as np
from scipy import ndimage



#function to pad an image to match a given shape
def match_shape(a, shape):
    t_,c_,z_, y_, x_ = shape
    t, c, z, y, x = a.shape
    t_pad = (t_-t)
    c_pad = (c_-c)
    z_pad = (z_-z)
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((t_pad//2, t_pad//2 + t_pad%2), 
                    (c_pad//2, c_pad//2 + c_pad%2), 
                    (z_pad//2, z_pad//2 + z_pad%2), 
                    (y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant')


def rotate_tiff(
        im,
        EulerX,
        EulerZ,
        NormalRotationAngle,
        smooth,):
    #potentially smoothen the image before rotation
    if smooth:
        for x in range(im.shape[-4]):
            im[0,x,:,:,:] = ndimage.gaussian_filter(im[0,x,:,:,:], sigma=1, mode="nearest")
    
    #expand image before rotating to make sure nothing gets cut off
    shapemax = np.max(np.array([im.shape]))
    exam = shapemax + round(np.sqrt(shapemax)) + 3
    rimg = match_shape(im,list(im.shape[:2])+[exam]*3)


    #rotate image around x-axis
    rimg = ndimage.rotate(rimg, -EulerX, axes=(-3,-2))
    #rotate image around z-axis
    rimg = ndimage.rotate(rimg, -EulerZ, axes=(-2,-1))
    #rotate image around x-axis so that the widest part of the cell is pointing towards the -y direction
    rimg = ndimage.rotate(rimg, NormalRotationAngle, axes=(-3,-2))
    
    return rimg


#version of rotate tiff for multiprocessing, where result order is important
def rotate_tiff_mp(
        position, #the relative position of this image in a list of images
        im,
        EulerX,
        EulerZ,
        NormalRotationAngle,
        smooth: bool = False,):
    return position, rotate_tiff(im,EulerX,EulerZ,NormalRotationAngle,smooth,)

