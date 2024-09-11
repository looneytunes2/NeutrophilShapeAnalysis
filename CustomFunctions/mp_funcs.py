# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:54:13 2024

@author: Aaron
"""

from aicssegmentation.core.MO_threshold import MO


def quickcaaxseg(im):
    return MO(im,global_thresh_method = 'tri', object_minArea = 50000)