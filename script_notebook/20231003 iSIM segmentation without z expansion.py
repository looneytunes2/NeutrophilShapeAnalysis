# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:45:36 2023

@author: Aaron
"""

import numpy as np
import os
import pandas as pd
import time
import math

# package for io 
from aicsimageio.writers import OmeTiffWriter
from aicsimageio.readers import tiff_reader, OmeTiffReader

from CustomFunctions.track_functions import segment_caax_tracks_confocal_40x
import itertools
from skimage import measure as skmeasure
import datetime


from skimage.morphology import disk
from skimage.filters import median, gaussian


# load in some stuff to speed up processing
# (following https://sedeh.github.io/python-pandas-multiprocessing-workaround.html)
import multiprocessing

def collect_results(result):
    """Uses apply_async's callback to setup up a separate Queue for each process.
    This will allow us to collect the results from different threads."""
    results.append(result)
    

def mygrouper(n, iterable):
    args = [iter(iterable)] * n
    return ([e for e in t if e != None] for t in itertools.zip_longest(*args))
    


def get_intensity_features(img, seg):
    features = {}
    input_seg = seg.copy()
    input_seg = (input_seg>0).astype(np.uint8)
    input_seg_lcc = skmeasure.label(input_seg)
    for mask, suffix in zip([input_seg, input_seg_lcc], ['', '_lcc']):
        values = img[mask>0].flatten()
        if values.size:
            features[f'intensity_mean{suffix}'] = values.mean()
            features[f'intensity_std{suffix}'] = values.std()
            features[f'intensity_1pct{suffix}'] = np.percentile(values, 1)
            features[f'intensity_99pct{suffix}'] = np.percentile(values, 99)
            features[f'intensity_max{suffix}'] = values.max()
            features[f'intensity_min{suffix}'] = values.min()
        else:
            features[f'intensity_mean{suffix}'] = np.nan
            features[f'intensity_std{suffix}'] = np.nan
            features[f'intensity_1pct{suffix}'] = np.nan
            features[f'intensity_99pct{suffix}'] = np.nan
            features[f'intensity_max{suffix}'] = np.nan
            features[f'intensity_min{suffix}'] = np.nan
    return features



def dist_f(a1, b1, c1, a2, b2, c2):

    d = ( a1 * a2 + b1 * b2 + c1 * c2 )
    e1 = math.sqrt( a1 * a1 + b1 * b1 + c1 * c1)
    e2 = math.sqrt( a2 * a2 + b2 * b2 + c2 * c2)
    d = d / (e1 * e2)
    A = math.degrees(math.acos(d))
    return A

# Function to find Angle
def angle_distance(a1, b1, c1, a2, b2, c2):
    a1,b1,c1 = [a1,b1,c1]/np.linalg.norm([a1,b1,c1])
    a2,b2,c2 = [a2,b2,c2]/np.linalg.norm([a2,b2,c2])
    d = ( a1 * a2 + b1 * b2 + c1 * c2 )
    e1 = math.sqrt( a1 * a1 + b1 * b1 + c1 * c1)
    e2 = math.sqrt( a2 * a2 + b2 * b2 + c2 * c2)
    d = d / (e1 * e2)
    A = math.degrees(math.acos(d))
    return A



############### SEGMENT AND SAVE CELLS ################################

############## alignment by chemical gradient ###################

from scipy.spatial import KDTree, distance
from itertools import groupby
from operator import itemgetter
from scipy import interpolate
import re
from aicsimageio.readers.bioformats_reader import BioformatsReader

#import the cell segmentation and rotation function
# import sys
# # insert at 1, 0 is the script path (or '' in REPL)
# import_path = 'C:/Users/Aaron/Documents/PythonScripts/CustomFunctions'
# sys.path.insert(1, import_path)
from CustomFunctions.segment_cells_iSIM import segment_cells_rotafter_VV

from CustomFunctions.persistance_activity import get_pa


# path to folder(s)
folder_fl = 'D:/Aaron/Data/Galvanotaxis/Tracking_Images/'
filelist_fl = [f for f in os.listdir(folder_fl) if '.' not in f]
savedir = 'D:/Aaron/Data/Galvanotaxis/Processed_Data/'
raw_dir = '//10.158.28.37/ExpansionHomesA/avlnas/HL60 Galv/'

#parameters for segmentation
xy_buffer = 25 #pixels
z_buffer = 7 #pixels
sq_size = 250 #pixels
xyres = 0.1613 #um / pixel
zstep = 0.5 # um
interval = 15
intthresh = 120 #for the half shrunken images, determined by manually crossreferencing

u = filelist_fl[10]


# for u in filelist_fl:

################## align trackmate data with region props data ################
rpcsv = [x for x in os.listdir(folder_fl + u) if x.endswith("region_props.csv")][0]
rp = pd.read_csv(folder_fl + u + '/' + rpcsv, index_col = 0)
tmcsv = [x for x in os.listdir(folder_fl + u) if x.endswith("TrackMateLog.csv")][0]
tm = pd.read_csv(folder_fl + u+ '/' + tmcsv)
#fix trackmate columns to get names right and units in microns
tm['x'] = tm.POSITION_X*xyres
tm['y'] = tm.POSITION_Y*xyres
tm['z'] = tm.POSITION_Z*zstep
#make kdtree and query with trackmate log
kd = KDTree(rp[['frame','x','y','z']].to_numpy())
dd, ii = kd.query(tm[['FRAME','x','y','z']])
df_track = pd.concat([tm.drop(columns=['POSITION_X','POSITION_Y','POSITION_Z']), 
                    rp.iloc[ii].drop(columns=['frame','x','y','z', 'cell']).reset_index(drop=True)], axis=1)
df_track = df_track.rename(columns={'TRACK_ID':'cell', 'FRAME':'frame'})
#sort by cell and frame
df_track = df_track.sort_values(['cell','frame'])



# #read in image and tracking data
image_name = u
direct = folder_fl+u+'/'
test = [x for x in os.listdir(direct+'/') if '.tif' in x][0]
# currentim = folder_fl+filelist_fl[u]+'/'+ filelist_fl[u] + '_MMStack_Pos0.ome.tif'
currentim = direct+'/'+ test
im_temp_whole = tiff_reader.TiffReader(currentim)
#double shape because it was halved
imshape = [2*x for x in im_temp_whole.shape[1:]]


#############find distance travelled##################
longdistmatrix = distance.pdist(df_track[['x','y','z']])
shortdistmatrix = distance.squareform(longdistmatrix)
shortdistmatrix = np.array(shortdistmatrix)
dist = pd.Series([], dtype = 'float64')
for count, i in enumerate(shortdistmatrix):
    if count == 0:
        temp = pd.Series([0])
        dist = dist.append(temp, ignore_index=True)
    else:
        temp = pd.Series(shortdistmatrix[count,count-1])
        dist = dist.append(temp, ignore_index=True)
df_track = df_track.reset_index(drop = True)
df_track['dist'] = dist
#first rows that have super long distances from previous cell, so set them to 0
df_track.loc[df_track.groupby('cell').head(1).index,'dist'] = 0

############ replace unrealistic jumps in distance ##############
for x in df_track[df_track.dist>4].index.values:
    df_track['dist'][x] = df_track.dist.mean()

############## find euclidean distance #############
euclid = pd.DataFrame([])
for i, cell in df_track.groupby('cell'):
    FL = cell.iloc[[0,-1]]
    euc_dist = distance.pdist(FL[['x','y','z']])
    euclid = euclid.append({'cell':cell.cell.iloc[0], 'euc_dist':euc_dist[0]}, ignore_index = True)
cellsmorethan = euclid.loc[euclid['euc_dist']>5, 'cell']
df_track = df_track[df_track.cell.isin(cellsmorethan)]

#     ########remove "slow"/dead cells############
#     #sum distances
#     df_track_distsums = df_track.groupby('cell').sum()
#     df_track_distsums = df_track_distsums.add_suffix('_sum').reset_index()

#     #grab only cells with sums above a threshold distance
#     cellsmorethan = df_track_distsums.loc[df_track_distsums['dist_sum']>5, 'cell']
#     df_track = df_track[df_track.cell.isin(cellsmorethan)]


##########remove small things that are likely dead cells or parts of cells###########
df_track = df_track[df_track['area'] > 15000 ]

######### remove cells with low caax intensity ###########
df_track = df_track[df_track['intensity_avg'] > intthresh ]
#reset index after dropping all the rows
df_track = df_track.reset_index(drop = True)


########remove edge cells############
#only grab rows that aren't zero in z_min
df_track = df_track.loc[df_track['x_min'] >8 ]
df_track = df_track.loc[df_track['y_min'] >8 ]
df_track = df_track.loc[df_track['z_min'] !=0 ]
#remove rows where z_max matches z_range
df_track = df_track.loc[df_track['x_max'] < int(imshape[-1]-4)]
df_track = df_track.loc[df_track['y_max'] < int(imshape[-2]-4)]
df_track = df_track.loc[df_track['z_max'] != (df_track['z_range']-1)]
#reset index after dropping all the rows
df_track = df_track.reset_index(drop = True)





######## remove cells that touch ###########    
to_remove = []
for i, cell in df_track.groupby('cell'):
    if i>0:
        changes = abs(cell['convex_area'].pct_change())
        largerthan = changes[changes>0.75]
        if largerthan.empty == False:
            to_remove.extend(largerthan.index.to_list())


        # changes = cell['convex_area'].diff()
        # largerthan = changes[changes>cell['convex_area']*0.333]
        # smallerthan = changes[changes<cell['convex_area']*-0.333]
        # print(largerthan.index,smallerthan.index)
        # #remove all frames of a cell after it contacts another cell
        # if largerthan.empty == False:
        #     for n in largerthan.index:
        #         to_remove.append(list(range(n, max(cell.index)+1)))
        # #remove all frames of a cell before it splits from another cell
        # if smallerthan.empty == False:
        #     for n in smallerthan.index:
        #         to_remove.append(list(range(cell.index[0], n-1)))
# #remove duplicate indicies
# to_remove = [j for x in to_remove for j in x]
# to_remove = list(set(to_remove))
#drop touching or splitting cells
df_track = df_track.drop(to_remove)



if df_track.empty == False:
    for i, cells in df_track.groupby('cell'):
        cells = cells.reset_index(drop = True)
        runs = list()
        #######https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
        for k, g in groupby(enumerate(cells['frame']), lambda ix: ix[0] - ix[1]):
            currentrun = list(map(itemgetter(1), g))
            list.append(runs, currentrun)
        for r in runs:
            r = np.array(r, dtype=int)
            #skip runs less than 3 frames long
            if len(r)<3:
                pass
            else:
                cell = cells.iloc[[cells[cells.frame==y].index[0] for y in r]]
                #segment the cell channel and get centroid
                df = pd.DataFrame()
                if __name__ ==  '__main__':
                    # use multiprocessing to perform segmentation and x,y,z determination
                    pool = multiprocessing.Pool(processes=60)
                    results = []
                    for t, row in cell.iterrows():
                        
                        
                        t = 1
                        row = cell.iloc[t]
                        
                        
                        tdir = raw_dir +u.split('_')[0]+'/'
                        cellstack = tdir + [x for x in os.listdir(tdir) if 'Reflected' in x and u in x and bool(int(re.search(r'_t(\d+)', x).group(1))==int(row.frame+1))][0]
                        structstack = tdir + [x for x in os.listdir(tdir) if 'Trans' in x and u in x and bool(int(re.search(r'_t(\d+)', x).group(1))==int(row.frame+1))][0]

                        xmincrop = int(max(0, row.x_min-xy_buffer))
                        ymincrop = int(max(0, row.y_min-xy_buffer))
                        zmincrop = int(max(0, row.z_min-z_buffer))

                        zmaxcrop = int(min(row.z_max+z_buffer, imshape[-3])+1)
                        ymaxcrop = int(min(row.y_max+xy_buffer, imshape[-2])+1)
                        xmaxcrop = int(min(row.x_max+xy_buffer, imshape[-1])+1)

                        segment_cells_rotafter_VV(
                            celldir,
                            structdir,
                            row,
                            image_name,
                            savedir,
                            xyres,
                            zstep,
                            xmincrop, 
                            ymincrop, 
                            zmincrop,
                            xmaxcrop, 
                            ymaxcrop, 
                            zmaxcrop,
                            ),             
                            callback = collect_results)

    
    
    
    

def segment_cells_rotafter_VV(
    cellstack,
    structstack,
    row,
    image_name,
    savedir,
    xyres,
    zstep,
    xmincrop, 
    ymincrop, 
    zmincrop,
    xmaxcrop, 
    ymaxcrop, 
    zmaxcrop,
):

    """
        Parameters
        ----------
        image : ndarray
            Input image. 
        zstep : float
            Z step of the image
        xyres : float
            microns/pixel resolution
        Returns
        -------

        Other parameters
        ----------------

        Notes
        -----


    """

    celld = TiffReader(cellstack)
    struct = TiffReader(structstack)
    
    #construct cropped raw image
    raw_img = np.stack((celld.data[zmincrop:zmaxcrop,
                            ymincrop:ymaxcrop,
                            xmincrop:xmaxcrop]
                        ,struct.data[zmincrop:zmaxcrop,
                            ymincrop:ymaxcrop,
                            xmincrop:xmaxcrop]))
    
    #file name template
    cell_name = image_name + f'_cell_{int(row.cell)}_frame_{int(row.frame)+1}'


    
    # segment cropped image
    seg_rimg = np.zeros(raw_img.shape)
    seg_rimg[0,:,:,:] = segment_caax_norot(raw_img_ex[0,:,:,:])
    if 'actin' in cell_name:
        seg_rimg[1,:,:,:] = segment_actinGFP_norot(raw_img_ex[1,:,:,:])
        strr = 'actin'
    if 'Hoechst' in cell_name:
        seg_rimg[1,:,:,:] = segment_nucleus(raw_img_ex[1,:,:,:])
        strr = 'nucleus'
    if 'DNA' in cell_name:
        seg_rimg[1,:,:,:] = segment_nucleus(raw_img_ex[1,:,:,:])
        strr = 'nucleus'
    if 'myosin' in cell_name:
        seg_rimg[1,:,:,:] = segment_actinGFP_norot(raw_img_ex[1,:,:,:])
        strr = 'myosin'
    

    #ensure there is no structure segmentation outside the membrane segmentation
    # seg_rimg[1,:,:,:] = np.logical_and(seg_rimg[0,:,:,:], seg_rimg[1,:,:,:])



    #get intensity features
    # nucleus_int = im_temp_whole_int[1,:,:,:]
    mem_feat = get_intensity_features(raw_img_ex[0,:,:,:], seg_rimg[0,:,:,:])
    mem_keylist = [x for x in list(mem_feat) if not x.endswith('lcc')]
    str_feat = get_intensity_features(raw_img_ex[1,:,:,:], seg_rimg[1,:,:,:])
    str_keylist = [x for x in list(str_feat) if not x.endswith('lcc')]

    #crop the segmented image
    im_labeled, n_labels = skimage.measure.label(
                              seg_rimg[0,:,:,:], background=0, return_num=True)
    im_props = skimage.measure.regionprops(im_labeled)


    #get original centroids
    cent = im_props[0].centroid
    

    #SAVE SEGMENTED IMAGE
    out=seg_rimg.astype(np.uint8)
    out[out>0]=255
    
    
    # remove file if it already exists
    seg_file = savedir + cell_name + '_segmented.tiff'
    if os.path.exists(seg_file):
        os.remove(seg_file)
    OmeTiffWriter.save(out, seg_file, dimension_order = "CZYX")
    
    
   
    #SAVE THE RAW IMAGE
    raw_file = savedir + cell_name + '_raw.tiff'
    if os.path.exists(raw_file):
        os.remove(raw_file)
    OmeTiffWriter.save(raw_img, raw_file, dimension_order = "CZYX")
    
    
    
    #save the info about cell
    data = {'image': image_name,
            'cell': cell_name,
            'structure': strr,
            'frame': row.frame,
            'x':(cent[2]+xmincrop)*xyres, 
            'y':(cent[1]+ymincrop)*xyres, 
            'z':(cent[0]+zmincrop)*zstep,
            'cropx (pixels)':cent[2], 
            'cropy (pixels)':cent[1], 
            'cropz (pixels)':cent[0],
            'Cell_'+mem_keylist[0]: mem_feat[mem_keylist[0]],
            'Cell_'+mem_keylist[1]: mem_feat[mem_keylist[1]],
            'Cell_'+mem_keylist[2]: mem_feat[mem_keylist[2]],
            'Cell_'+mem_keylist[3]: mem_feat[mem_keylist[3]],
            'Cell_'+mem_keylist[4]: mem_feat[mem_keylist[4]],
            'Cell_'+mem_keylist[5]: mem_feat[mem_keylist[5]],
            'Structure_'+str_keylist[0]: str_feat[str_keylist[0]],
            'Structure_'+str_keylist[1]: str_feat[str_keylist[1]],
            'Structure_'+str_keylist[2]: str_feat[str_keylist[2]],
            'Structure_'+str_keylist[3]: str_feat[str_keylist[3]],
            'Structure_'+str_keylist[4]: str_feat[str_keylist[4]],
            'Structure_'+str_keylist[5]: str_feat[str_keylist[5]],
            }
    
    return data



img = raw_img[0,:,:,:].copy()

def segment_caax_norot(img):
    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [0]
    gaussian_smoothing_sigma = 2
    ################################
    # intensity normalization
    struct_img = intensity_normalization(img, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)

    # step 1: Masked-Object (MO) Thresholding
    thresh_img, object_for_debug = MO(structure_img_smooth[:,:,:], global_thresh_method='tri', object_minArea=3000, local_adjust = 0.85, return_object=True)

    # # fill in the holes
    # hole_max = 15000
    # hole_min = 1
    # thresh_img_fill = hole_filling(thresh_img, hole_min, hole_max)


    # structure_img_smooth = edge_preserving_smoothing_3d(struct_img)
    ################################
    ## PARAMETERS for this step ##
    # f3_param = [[1, 0.3]]
    # f2_param = [[1,0.22],[2, 0.17]]
    f2_param = [[1,0.3]]
    ################################

    fil_img = filament_2d_wrapper(structure_img_smooth, f2_param)


    seg = thresh_img + fil_img

    # fill in the holes
    hole_max = 8000
    hole_min = 1
    seg = hole_filling(seg, hole_min, hole_max) 


    # Step 2: Perform topology-preserving thinning
    thin_dist_preserve = 2
    thin_dist = 1
    seg = topology_preserving_thinning(seg, thin_dist_preserve, thin_dist)



    # set minimum area to just less that largest object
    im_labeled, n_labels = skimage.measure.label(
                              seg, background=0, return_num=True)
    if n_labels > 1:
        im_props = skimage.measure.regionprops(im_labeled)
        tempdf = pd.DataFrame([])
        for count, prop in enumerate(im_props):
            area = prop.area
            tempdata = {'cell':count, 'area':area}
            tempdf = tempdf.append(tempdata, ignore_index=True)
        minArea = int(tempdf.area.max()-2)
        # create segmentation mask               
        seg = remove_small_objects(im_labeled, min_size=minArea, connectivity=1, in_place=False)
        seg = seg.astype(np.uint8)
        seg[seg > 0] = 255
    else:
        seg = seg.astype(np.uint8)
        seg[seg > 0] = 255



OmeTiffWriter.save(seg, 'C:/Users/Aaron/Documents/Python Scripts/temp/'+cell_name+'.ome.tiff',dim_order='ZYX')


img = raw_img[1,:,:,:].copy()

def segment_nucleus(img):
    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [0]
    gaussian_smoothing_sigma = 2
    ################################
    # intensity normalization
    struct_img = intensity_normalization(img, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter slice by slice 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    
    # step 1: Masked-Object (MO) Thresholding
    thresh_img = MO(structure_img_smooth[:,:,:], global_thresh_method='tri', object_minArea=1200, local_adjust = 1.25)
    
    # fill in the holes
    hole_max = 4000
    hole_min = 2
    seg = hole_filling(thresh_img, hole_min, hole_max)
    

    ################################
    ## PARAMETERS for this step ##
    # set minimum area to just less that largest object
    im_labeled, n_labels = skimage.measure.label(
                              seg, background=0, return_num=True)
    if n_labels > 1:
        im_props = skimage.measure.regionprops(im_labeled)
        tempdf = pd.DataFrame([])
        for count, prop in enumerate(im_props):
            area = prop.area
            tempdata = {'cell':count, 'area':area}
            tempdf = tempdf.append(tempdata, ignore_index=True)
        minArea = tempdf.sort_values(by='area', ascending=False).loc[0,['area']][0]-1
    else:
        minArea = 6000
    ################################
    #combine the two segmentations
    # create segmentation mask               
    seg = remove_small_objects(seg>0, min_size=minArea, connectivity=1, in_place=False)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255
    
    
OmeTiffWriter.save(seg, 'C:/Users/Aaron/Documents/Python Scripts/temp/'+cell_name+'nuc.ome.tiff',dim_order='ZYX')



img = raw_img[1,:,:,:].copy()

def segment_actinGFP_norot(img):
    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [0]
    gaussian_smoothing_sigma = 1.5
    ################################
    # intensity normalization
    struct_img = intensity_normalization(img, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    
    # step 1: Masked-Object (MO) Thresholding
    thresh_img = MO(structure_img_smooth, global_thresh_method='tri', local_adjust=1.12 ,object_minArea=20)
    
    
    seg = thresh_img.astype(np.uint8)
    seg[seg > 0] = 255
    
    OmeTiffWriter.save(seg, 'C:/Users/Aaron/Documents/Python Scripts/temp/'+cell_name+'nuc.ome.tiff',dim_order='ZYX')
