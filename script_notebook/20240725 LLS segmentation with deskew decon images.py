# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:34:24 2024

@author: Aaron
"""

import os
import numpy as np
from aicssegmentation.core.MO_threshold import MO
from aicsimageio.readers.czi_reader import CziReader
from skimage.morphology import remove_small_objects
import skimage
import pandas as pd
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from aicssegmentation.core.pre_processing_utils import edge_preserving_smoothing_3d, intensity_normalization, image_smoothing_gaussian_slice_by_slice, image_smoothing_gaussian_3d
from aicssegmentation.core.hessian import compute_3d_hessian_matrix
from aicssegmentation.core import vessel
from aicssegmentation.core.utils import hole_filling
from scipy.ndimage import laplace
im = 'G:/Deskewed_Decon_LLS/20240527_488_EGFP-CAAX_640_SPY650-DNA_cell2-01-Subset-02-Deskewed.czi'


czi = CziReader(im)


big = czi.data

single = czi.data[0,:,:,:,:]


caax = single[1,:,:,:]


OmeTiffWriter.save(caax,'C:/Users/Aaron/NeutrophilShapeAnalysis/script_notebook/notebook_data/20240711_488_EGFP-CAAX_640_actin-halotag_01perDMSO_cell2-02-Subset-01-DeskewedDecon5_frame1.ome.tiff')

 

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
    
    
    
from CustomFunctions import metadata_funcs

scale = metadata_funcs.getscale(czi)
xyres = scale[0]
zstep = scale[-1]
imshape = list(big.shape)
imshape.remove(2)
bigseg = np.zeros(imshape)
minArea = 50000
tempdata = []
for b in range(imshape[0]):
    thresh = MO(big[b,1,:,:,:],global_thresh_method = 'tri', object_minArea = 50000)
    seg = remove_small_objects(thresh>0, min_size=minArea, connectivity=1, in_place=False)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255

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
        
        im_labeled, n_labels = skimage.measure.label(
                                  seg, background=0, return_num=True,  )
    else:
        seg = seg.astype(np.uint8)
        seg[seg > 0] = 255
    
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
               'z':z*zstep, 'y':y*xyres, 'x': x*xyres, 'z_range': seg.shape[-3],
                'area':area, 'convex_area':convex_area, 'minor_axis_length':minor_axis_length, 'major_axis_length':major_axis_length})
    bigseg[b,:,:,:] = seg
    print('segmented '+str(b))
df = pd.DataFrame(tempdata)    
    
OmeTiffWriter.save(bigseg, 'C:/Users/Aaron/Desktop/seg.ome.tiff')



if __name__ ==  '__main__':
        # use multiprocessing to perform segmentation and x,y,z determination
        pool = multiprocessing.Pool(processes=60)
        results = []
        for t, row in cell.iterrows():
            
            tdir = raw_dir +u.split('_')[0]+'/' +u+'/Default/'
            
            xmincrop = int(max(0, row.x_min-xy_buffer))
            ymincrop = int(max(0, row.y_min-xy_buffer))
            zmincrop = int(max(0, row.z_min-z_buffer))

            zmaxcrop = int(min(row.z_max+z_buffer, imshape[-3]))
            ymaxcrop = int(min(row.y_max+xy_buffer, imshape[-2])+1)
            xmaxcrop = int(min(row.x_max+xy_buffer, imshape[-1])+1)
            
            tempim = big[int(row.frame),zmincrop:zmaxcrop,ymincrop:ymaxcrop,xmincrop:xmaxcrop].copy()
            
            pool.apply_async(seg_confocal_40x_memonly_fromslices, args = (
                tdir,
                tempim,
                imshape,
                row,
                u,
                savedir,
                xyres,
                zstep,
                ),             
                callback = collect_results)

        pool.close()
        pool.join()

        print(f'Done segmenting {u} cell {cell.cell.iloc[0]}')


row = df.iloc[0]
xy_buffer = 20
z_buffer = 20
xmincrop = int(max(0, row.x_min-xy_buffer))
ymincrop = int(max(0, row.y_min-xy_buffer))
zmincrop = int(max(0, row.z_min-z_buffer))

zmaxcrop = int(min(row.z_max+z_buffer, imshape[-3]))
ymaxcrop = int(min(row.y_max+xy_buffer, imshape[-2])+1)
xmaxcrop = int(min(row.x_max+xy_buffer, imshape[-1])+1)

tempim = big[int(row.frame),:,zmincrop:zmaxcrop,ymincrop:ymaxcrop,xmincrop:xmaxcrop].copy()



####### segment image
segment_caax_decon():

OmeTiffWriter.save(tempim, 'C:/Users/Aaron/orig.ome.tiff')
################################
## PARAMETERS for this step ##
intensity_scaling_param = [0]
gaussian_smoothing_sigma = 1
################################
# intensity normalization
struct_img = intensity_normalization(tempim[1,:,:,:], scaling_param=intensity_scaling_param)
# smoothing with 2d gaussian filter 
structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
OmeTiffWriter.save(structure_img_smooth, 'C:/Users/Aaron/Desktop/smooth.ome.tiff')

# step 1: Masked-Object (MO) Thresholding
thresh_img = MO(structure_img_smooth, global_thresh_method='tri', object_minArea=50000)#, local_adjust = 0.97)

ves = vessel.filament_2d_wrapper(structure_img_smooth, [[1.5,0.2]])
thresh_img = thresh_img.astype(np.uint8)
thresh_img[thresh_img > 0] = 255
OmeTiffWriter.save(thresh_img, 'C:/Users/Aaron/Desktop/thresh.ome.tiff')
ves = ves.astype(np.uint8)
ves[ves > 0] = 255
OmeTiffWriter.save(ves, 'C:/Users/Aaron/Desktop/vessel.ome.tiff')

both = ves.copy() + thresh_img.copy()
both = remove_small_objects(both>0, min_size=50000, connectivity=1, in_place=False)

both = both.astype(np.uint8)
both[both > 0] = 255
OmeTiffWriter.save(both, 'C:/Users/Aaron/Desktop/both.ome.tiff')


hess = compute_3d_hessian_matrix(struct_img)
OmeTiffWriter.save(hess, 'C:/Users/Aaron/Desktop/hess.ome.tiff')



tempim = big[int(row.frame),:,zmincrop:zmaxcrop,ymincrop:ymaxcrop,xmincrop:xmaxcrop].copy()

mask = np.invert(thresh_img.astype(bool))

######### nuclear seg
intensity_scaling_param = [0]
gaussian_smoothing_sigma = 1
################################
# intensity normalization
struct_img = intensity_normalization(tempim[0,:,:,:], scaling_param=intensity_scaling_param)
# smoothing with 2d gaussian filter 
structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
OmeTiffWriter.save(structure_img_smooth, 'C:/Users/Aaron/Desktop/nucsmooth.ome.tiff')

maa = ma.array(structure_img_smooth,mask = mask)
# step 1: Masked-Object (MO) Thresholding
nucthresh_img = MO(maa, global_thresh_method='tri', object_minArea=25000)#, local_adjust = 0.95)
nucthresh_img = nucthresh_img.astype(np.uint8)
nucthresh_img[nucthresh_img > 0] = 255
OmeTiffWriter.save(nucthresh_img, 'C:/Users/Aaron/Desktop/nucthresh.ome.tiff')


hess = compute_3d_hessian_matrix(struct_img)
OmeTiffWriter.save(hess, 'C:/Users/Aaron/Desktop/hess.ome.tiff')


lap = laplace(structure_img_smooth)
lapthresh = MO(lap, global_thresh_method='tri', object_minArea=25000)#, local_adjust = 0.99)
lapthresh = lapthresh.astype(np.uint8)
lapthresh[lapthresh > 0] = 255
OmeTiffWriter.save(lapthresh, 'C:/Users/Aaron/Desktop/lapthresh.ome.tiff')

####### get shape stats


####### save image 
## optional save as full size of original




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










# detect if there's more than one object in the thresholded image
im_labeled, n_labels = skimage.measure.label(
                          thresh_img.astype(np.uint8), background=0, return_num=True)
#if there's more than one object try to erase the non-focused cell and re-threshold
if n_labels > 1:
    im_props = skimage.measure.regionprops(im_labeled)
    imcent = np.array(img.shape)/2
    distances = []
    for count, prop in enumerate(im_props):
        #append the distance between this object and the center of the image
        distances.append(distance.pdist(np.stack([imcent, np.array(prop.centroid)])))
    #get the index of the closest object to the center of the image
    realin = np.argmin(distances)
    for n in list(range(n_labels)):
        if n == realin:
            pass
        else:
            structure_img_smooth = partial_cell_removal_caax(structure_img_smooth, im_labeled, n+1)

    #remove the brightest pixels from the cell of interest
    values = structure_img_smooth[im_labeled==realin+1].flatten()
    structure_img_smooth[structure_img_smooth>np.percentile(values, 98)] = np.percentile(values, 90)
    # threshold the new modified image
    thresh_img = MO(structure_img_smooth, global_thresh_method='tri', object_minArea=750, local_adjust = 0.95)

else:
    #remove the brightest pixels from the cell of interest
    values = structure_img_smooth[im_labeled>0].flatten()
    structure_img_smooth[structure_img_smooth>np.percentile(values, 98)] = np.percentile(values, 90)
    # threshold the new modified image
    # step 1: Masked-Object (MO) Thresholding
    thresh_img = MO(structure_img_smooth, global_thresh_method='tri', object_minArea=50000, local_adjust = 0.999)

    thresh_img = thresh_img.astype(np.uint8)
    thresh_img[thresh_img > 0] = 255
    OmeTiffWriter.save(thresh_img, 'C:/Users/Aaron/Desktop/thresh.ome.tiff')
    
    
    
    
    
######################## test runs ###############################
croppeddir = 'G:/Deskewed_Decon_LLS/'
savedir = 'G:/Processed_LLS/'
if not os.path.exists(savedir):
    os.makedirs(savedir)


allcellinfo = []

cell_list = set([re.findall(r'(.*?_cell\d)', x)[0] for x in os.listdir(croppeddir)])
for i in cell_list:
    #get all of the images from a particular cell I was following
    curimlist = [x for x in os.listdir(croppeddir) if i in x]
    #find the total number of cells I cropped while following the cell of interest
    cellnums = set([re.findall(r'Subset-(\d+)',x)[0] for x in curimlist])
    for s in cellnums:
        #get all the images of a given cell
        curcell = [x for x in curimlist if f'Subset-{s}' in x]
        #sort the current cell to be in chronological order
        curcell.sort(key=lambda x: float(re.findall(r'(\d+)-Subset', x)[0]))
        time_elapsed = 0
        for n, c in enumerate(curcell):
            celldir = croppeddir + c
            #open the image
            czi = CziReader(celldir)
            #absolute timepoint of first image
            if n == 0:
                timezero = metadata_funcs.adjustedstarttime(czi)
            #get time interval and number of frames and start time
            ti = metadata_funcs.gettimeinterval(czi)
            fn = metadata_funcs.framesinsubset(czi)
            ast = metadata_funcs.adjustedstarttime(czi)
            
            #get all the times at the current frame since the cell was initially observed
            frametimes = [int(ast - timezero + (f*ti)) for f in range(fn)]
            
            #segment the cells and return the position info
            info = segandinfo_LLS(savedir,
                               celldir,
                               times = frametimes,
                               interval = ti,
                               decon = True,
                               orig_size = True,
                               )

            allcellinfo.append(info)
            
            
allcellinfo = pd.DataFrame(allcellinfo)
allcellinfo.to_csv(datadir + 'All_Cell_Tracking_Info.csv')


    
