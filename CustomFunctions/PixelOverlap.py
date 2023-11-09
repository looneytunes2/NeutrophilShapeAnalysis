# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 17:52:32 2021

@author: Aaron
"""

import mathutils
import numpy as np

def PixelOverlap(obj1,obj2,alltheobjects,binarycoords,):

    # Get their world matrix
    mat1 = obj1.matrix_world
    mat2 = obj2.matrix_world

    # Get the geometry in world coordinates
    vert1 = [mat1 @ v.co for v in obj1.data.vertices] 
    poly1 = [p.vertices for p in obj1.data.polygons]

    vert2 = [mat2 @ v.co for v in obj2.data.vertices] 
    poly2 = [p.vertices for p in obj2.data.polygons]

    # Create the BVH trees
    bvh1 = mathutils.bvhtree.BVHTree.FromPolygons( vert1, poly1 )
    bvh2 = mathutils.bvhtree.BVHTree.FromPolygons( vert2, poly2 )

    # Test if overlap
    if bvh1.overlap( bvh2 ):
#         print( "Overlap" )
        #get vertex of pixel that's representative of 3D position
        numpyvert = np.array(vert2[:])
        currentcoord = np.array([[np.min(numpyvert[:,0]),
                                np.min(numpyvert[:,1]),
                                np.min(numpyvert[:,2])]])
        binarycoords = np.append(binarycoords, currentcoord, axis = 0)
        #add pixel object to list
        # alltheobjects.append(obj2)
        obj2.select_set(True)
    else:
#         print( "No overlap" )
        #add pixel object to list
        # alltheobjects.append(obj2)
        obj2.select_set(True)