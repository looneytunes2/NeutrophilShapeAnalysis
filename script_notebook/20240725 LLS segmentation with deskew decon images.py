# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:34:24 2024

@author: Aaron
"""

import os
import numpy as np
from aicssegmentation.core.MO_threshold import MO
from aicsimageio.czi_reader import CziReader

im = '//10.158.28.37/ExpansionHomesA/avlnas/LLS/2024-07-11/20240711_488_EGFP-CAAX_640_actin-halotag_01perDMSO_cell2-02-Subset-01-DeskewedDecon5.czi'


czi = CziReader(im)


big = czi.data

single = czi.data[0,:,:,:,:]


caax = single[1,:,:,:]


OmeTiffWriter.save(caax,'C:/Users/Aaron/NeutrophilShapeAnalysis/script_notebook/notebook_data/20240711_488_EGFP-CAAX_640_actin-halotag_01perDMSO_cell2-02-Subset-01-DeskewedDecon5_frame1.ome.tiff')


smooth = 

thresh = MO(caax,global_thresh_method = 'tri', object_minArea = 50000)
seg = thresh.astype(np.uint8)
seg[seg > 0] = 255
OmeTiffWriter.save(seg,'C:/Users/Aaron/Desktop/20240711_488_EGFP-CAAX_640_actin-halotag_01perDMSO_cell2-02-Subset-01-DeskewedDecon5_frame1.ome.tiff')



im_labeled, n_labels = skimage.measure.label(
                          seg, background=0, return_num=True,  )
if n_labels > 1:
    im_props = skimage.measure.regionprops(im_labeled)
    tempdf = pd.DataFrame([])
    for count, prop in enumerate(im_props):
        area = prop.area
        af = prop.filled_area
        tempdata = {'cell':count, 'area':area, 'area_fill':af}
        tempdf = tempdf.append(tempdata, ignore_index=True)
    minArea = int(tempdf.area.max()-2)
    
    seg = remove_small_objects(seg>0, min_size=minArea, connectivity=1, in_place=False)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255
else:
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255    
    
    
shape = list(big.shape)
shape.remove(2)
bigseg = np.zeros(shape)
minArea = 50000
tempdata = []
for b in range(big.shape[0]):
    thresh = MO(big[b,1,:,:,:],global_thresh_method = 'tri', object_minArea = 50000)
    seg = remove_small_objects(thresh>0, min_size=minArea, connectivity=1, in_place=False)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255
    bigseg[b,:,:,:] = seg
    im_labeled, n_labels = skimage.measure.label(
                              seg, background=0, return_num=True,  )
    im_props = skimage.measure.regionprops(im_labeled)
    for count, prop in enumerate(im_props):
        z,y,x = prop.centroid
        
        thebox = prop.bbox
        area = prop.area
        convex_area = prop.convex_area
        major_axis_length = prop.major_axis_length
        minor_axis_length = prop.minor_axis_length
        
        
        tempdata.append({'cell':count, 'frame':b, 'z_min':thebox[0], 'y_min':thebox[1], 
                'x_min':thebox[2],'z_max':thebox[3], 'y_max':thebox[4], 'x_max':thebox[5],
               'z':cent[0]*step, 'y':cent[1]*ip, 'x': cent[2]*ip, 'z_range': struct_img0.shape[-3],
                'area':area, 'convex_area':convex_area, 'minor_axis_length':minor_axis_length, 'major_axis_length':major_axis_length})
        
    print('segmented '+str(b))
df = pd.DataFrame(tempdata)    
    
    
from CustomFunctions.track_functions import tracking_track
    
newdf = tracking_track(df)

celllist = newdf.cell.unique()
#remove all the things that aren't in frame the whole time
for c in newdf.cell.unique():
    curcell = newdf[newdf.cell == c]
    if len(curcell) != big.shape[0]:
        celllist.remove(c)
if len(celllist)>1:
    
    
    
    

[big.shape[-1]]*len(curcell)
disp_arr_temp[3:] = np.sqrt((curcell['x'] - disp_arr_temp[0])**2 +
                            (curcell['y'] - disp_arr_temp[1])**2 +
                            (curcell['z'] - disp_arr_temp[2])**2).values








tracks = track_objects_in_4d_video(bigseg,max_distance = 50)
    
video_4d = bigseg.copy()




    
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist

def track_objects_in_4d_video(video_4d, max_distance=10):
    """
    Track 3D objects moving through time in a 4D video.

    Parameters:
    - video_4d: numpy array of shape (t, z, y, x) representing the 4D video.
    - max_distance: maximum allowed distance between centroids to consider objects the same across frames.

    Returns:
    - tracks: list of dictionaries containing object IDs and their positions over time.
    """
    # Initialize a list to store tracking information
    tracks = []
    
    # Iterate through each time frame
    for t in range(video_4d.shape[0]):
        # Label 3D objects in the current frame
        labeled_frame = label(video_4d[t])
        regions = regionprops(labeled_frame)
        
        # Extract centroids of the labeled objects
        centroids = np.array([region.centroid for region in regions])
        
        if t == 0:
            # Initialize tracks with the first frame's objects
            for i, centroid in enumerate(centroids):
                tracks.append({'id': i, 'positions': [centroid]})
        else:
            # Match current frame's centroids to previous frame's centroids
            previous_centroids = np.array([track['positions'][-1] for track in tracks if track['positions'][-1] is not None])
            if previous_centroids.size == 0:
                previous_centroids = np.empty((0, 3))  # Ensure previous_centroids is 2-dimensional
            
            distances = cdist(previous_centroids, centroids) if previous_centroids.size > 0 and centroids.size > 0 else np.empty((0, 0))
            for i, track in enumerate(tracks):
                if previous_centroids.size > 0 and centroids.size > 0:
                    min_dist_idx = np.argmin(distances[i])
                    if distances[i, min_dist_idx] <= max_distance:
                        track['positions'].append(centroids[min_dist_idx])
                    else:
                        track['positions'].append(None)
                else:
                    track['positions'].append(None)
            
            # Handle new objects
            if centroids.size > 0:
                matched_indices = np.argmin(distances, axis=0) if distances.size > 0 else []
                for j in range(centroids.shape[0]):
                    if distances.size == 0 or np.min(distances[:, j]) > max_distance:
                        new_id = len(tracks)
                        tracks.append({'id': new_id, 'positions': [None] * t + [centroids[j]]})
        print('finished ' + str(t))
    
    return tracks