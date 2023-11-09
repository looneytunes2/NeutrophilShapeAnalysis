# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 14:40:36 2021

@author: Aaron
"""

from skimage import transform

def xyexpand(img, size, ordr, z):
    ex = transform.resize(img, size, order=ordr, preserve_range=True)
    return [z, ex]

def xzexpand(img, size, ordr, z):
    ex = transform.resize(img, size, order=ordr, preserve_range=True)
    return [z, ex]