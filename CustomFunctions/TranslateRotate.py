# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:06:45 2021

@author: Aaron
"""

from skimage import transform


def TR(im, ch, i):
    rot_img = transform.rotate(im, -0.6, preserve_range=True)
    translato = transform.AffineTransform(
            translation=(6.105612607169708,-53.36166650068579)
            )
    timg1 = transform.warp(rot_img, translato.inverse)

    return [[i, timg1]]