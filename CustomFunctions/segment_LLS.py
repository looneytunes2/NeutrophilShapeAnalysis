# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:58:51 2024

@author: Aaron
"""

import os
import re
import math
import multiprocessing
import pandas as pd
import numpy as np
import numpy.ma as ma
from itertools import groupby
from operator import itemgetter
from aicsimageio.readers.czi_reader import CziReader
from aicsimageio.writers import OmeTiffWriter
import skimage.measure
from skimage.morphology import remove_small_objects
from aicssegmentation.core.MO_threshold import MO
from aicssegmentation.core.utils import hole_filling
from aicssegmentation.core import vessel
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_3d
from CustomFunctions import metadata_funcs
from scipy.spatial import distance
from scipy import interpolate
from CustomFunctions.persistance_activity import get_pa, DA_3D
from CustomFunctions.mp_funcs import quickcaaxseg
from CustomFunctions.MO_Threshold import MO_ma

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

def collect_results(result):
    """Uses apply_async's callback to setup up a separate Queue for each process.
    This will allow us to collect the results from different threads."""
    results.append(result)
    


def get_intensity_features(img, seg):
    features = {}
    input_seg = seg.copy()
    input_seg = (input_seg>0).astype(np.uint8)
    input_seg_lcc = skimage.measure.label(input_seg)
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



def twodholefill(thresh, hole_min, hole_max):
    YZ = thresh.swapaxes(0,2)
    YZ_fill = hole_filling(YZ, hole_min, hole_max, fill_2d=True)
    YZrev = YZ_fill.swapaxes(2,0)
    XZ = YZrev.swapaxes(0,1)
    XZ_fill = hole_filling(XZ, hole_min, hole_max, fill_2d=True)
    XZrev = XZ_fill.swapaxes(1, 0)
    XY = hole_filling(XZrev, hole_min, hole_max, fill_2d=True)
    return XY



def segment_caax_decon(im):
    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [0]
    gaussian_smoothing_sigma = 1
    ################################
    # intensity normalization
    struct_img = intensity_normalization(im, scaling_param=intensity_scaling_param)
    # smoothing with 2d gaussian filter 
    smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    #remove mega bright pixels
    structure_img_smooth = ma.masked_array(smooth, mask = smooth>np.percentile(smooth[im>100], 98))
    # step 1: Masked-Object (MO) Thresholding
    thresh_img = MO_ma(structure_img_smooth, global_thresh_method='tri', object_minArea=50000, local_adjust = 0.95)
    #remove small objects
    thresh_img = remove_small_objects(thresh_img, min_size=50000, connectivity=1, in_place=False)
    # detect if there's more than one object in the thresholded image
    im_labeled, n_labels = skimage.measure.label(
                              thresh_img.astype(np.uint8), background=0, return_num=True)
    #if there's more than one object try to erase the non-focused cell and re-threshold
    if n_labels > 1:
        im_props = skimage.measure.regionprops(im_labeled)
        imcent = np.array(im.shape)/2
        distances = []
        for count, prop in enumerate(im_props):
            #append the distance between this object and the center of the image
            distances.append(distance.pdist(np.stack([imcent, np.array(prop.centroid)])))
        #select only the object closest to the image center
        thresh_img = im_labeled == int(np.argmin(distances)+1)
        

    #filament filter
    ves = vessel.filament_2d_wrapper(structure_img_smooth, [[1.5,0.2]])
    #combine
    both = np.logical_or(ves, thresh_img)
    # set minimum area to just less that largest object
    im_labeled, n_labels = skimage.measure.label(
                              both, background=0, return_num=True)
    if n_labels > 1:
        im_props = skimage.measure.regionprops(im_labeled)
        tempdf = pd.DataFrame([])
        for count, prop in enumerate(im_props):
            area = prop.area
            tempdata = {'cell':count, 'area':area}
            tempdf = tempdf.append(tempdata, ignore_index=True)
        minArea = int(tempdf.area.max()-2)
        # create segmentation mask               
        both = remove_small_objects(im_labeled, min_size=minArea, connectivity=1, in_place=False)
    #fill holes in segmentation
    hole_max = 5000
    hole_min = 1
    both = twodholefill(both, hole_min, hole_max)
    #convert to 8bit binary
    both = both.astype(np.uint8)
    both[both > 0] = 255
    return both

def segment_nuc_decon(im,
                      ):
    ######### nuclear seg
    intensity_scaling_param = [0]
    gaussian_smoothing_sigma = 1
    ################################
    # intensity normalization
    struct_img = intensity_normalization(im, scaling_param=intensity_scaling_param)
    # smoothing with 3d gaussian filter 
    structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    # step 1: Masked-Object (MO) Thresholding
    thresh_img = MO(structure_img_smooth, global_thresh_method='tri', object_minArea=25000, local_adjust = 0.995)
    #remove small objects
    thresh_img = remove_small_objects(thresh_img, min_size=25000, connectivity=1, in_place=False)
    #fill holes in segmentation
    hole_max = 1000
    hole_min = 1
    thresh_img_fill = twodholefill(thresh_img, hole_min, hole_max)
    #change make it 8bit
    thresh_img_fill = thresh_img_fill.astype(np.uint8)
    thresh_img_fill[thresh_img_fill > 0] = 255
    return thresh_img_fill


def LLSseg(savedir:str,
           image_name:str,
           cropdict: dict,
           im:np.array,
           struct:str,
           xyres:float,
           zstep:float,
           decon:bool,
           orig_size:bool = False,
           orig_shape: np.array = np.zeros(1),
           xy_buffer: int = 20,
           z_buffer: int = 20,
           ):
    #file name template
    cell_name = re.split(r'(?<=Subset-\d{2})', image_name)[0] + '_frame_'+ str(int(cropdict['frame']))
    shortimname = image_name.split('-Subset')[0]
    cell_number = re.findall(r"Subset-(\d+)", image_name)[0]
    
    
    ### segment the cells depending on their signals
    bothch = np.zeros(im.shape)
    if decon:
        bothch[1,:,:,:] = segment_caax_decon(im[1,:,:,:])
        #make boolean mask for secondary signals
        mask = np.invert(bothch[1,:,:,:].astype(bool))
        if struct == 'nucleus':
            bothch[0,:,:,:] = segment_nuc_decon(im[0,:,:,:])
        elif struct == 'actin':
            pass
            # segment_actin_decon(im[0,:,:,:], mask)
        elif struct == 'myosin':
            pass
            # segment_myosin_decon(im[0,:,:,:], mask)
    else:
        pass



    #get intensity features for both channels
    mem_feat = get_intensity_features(im[1,:,:,:], bothch[1,:,:,:])
    mem_keylist = [x for x in list(mem_feat) if not x.endswith('lcc')]
    str_feat = get_intensity_features(im[0,:,:,:], bothch[0,:,:,:])
    str_keylist = [x for x in list(str_feat) if not x.endswith('lcc')]

    #get final centroid
    cent = np.mean(np.nonzero(bothch[1,:,:,:]), axis = 1)
    
    #SAVE SEGMENTED IMAGE
    out=bothch.astype(np.uint8)
    out[out>0]=255
    
    
    if orig_size:
        #make empty image of the original shape
        orig_im = np.zeros(orig_shape)
        #get the right cropping boundaries
        xmincrop = int(max(0, cropdict['x_min']-xy_buffer))
        ymincrop = int(max(0, cropdict['y_min']-xy_buffer))
        zmincrop = int(max(0, cropdict['z_min']-z_buffer))
        zmaxcrop = int(min(cropdict['z_max']+z_buffer, orig_shape[-3]))
        ymaxcrop = int(min(cropdict['y_max']+xy_buffer, orig_shape[-2])+1)
        xmaxcrop = int(min(cropdict['x_max']+xy_buffer, orig_shape[-1])+1)
        #put segmented image into original empty image
        orig_im[:,
                zmincrop:zmaxcrop,
                ymincrop:ymaxcrop,
                xmincrop:xmaxcrop] = out.copy()
        # remove file if it already exists
        seg_file = savedir + cell_name + '_segmentedfull.tiff'
        if os.path.exists(seg_file):
            os.remove(seg_file)
        OmeTiffWriter.save(orig_im, seg_file, dimension_order = "CZYX")
        
        
    # remove file if it already exists
    seg_file = savedir + cell_name + '_segmented.tiff'
    if os.path.exists(seg_file):
        os.remove(seg_file)
    OmeTiffWriter.save(out, seg_file, dimension_order = "CZYX")
    
    #SAVE THE RAW IMAGE
    raw_file = savedir + cell_name + '_raw.tiff'
    if os.path.exists(raw_file):
        os.remove(raw_file)
    OmeTiffWriter.save(im, raw_file, dimension_order = "CZYX")
        
    
    
    #save the info about cell
    data = {'image': shortimname,
            'cellnumber': cell_number,
            'cell': cell_name,
            'structure': struct,
            'frame': cropdict['frame'],
            'time': cropdict['time'],
            'x':(cent[2]+cropdict['x_min'])*xyres, 
            'y':(cent[1]+cropdict['y_min'])*xyres, 
            'z':(cent[0]+cropdict['z_min'])*zstep,
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
    
    
def getbb(im):
    ### start by segmenting the large image to get all of the "primary" cells' bounding boxes for further cropping
    rescaled = skimage.transform.rescale(im,0.25, preserve_range=True)
    #mask the top pixels with signal
    mare = ma.masked_array(rescaled, mask = rescaled>np.percentile(rescaled[rescaled>100], 98))
    seg = MO_ma(mare, global_thresh_method='tri', object_minArea=200, local_adjust = 0.95)
    im_labeled, n_labels = skimage.measure.label(
                              seg, background=0, return_num=True,  )

    im_props = skimage.measure.regionprops(im_labeled)
    tempdf = pd.DataFrame([])
    for count, prop in enumerate(im_props):
        z,y,x = prop.centroid
        thebox = np.array(prop.bbox)*4
        area = prop.area * 64
        td = {'cell':count, 'z_min':thebox[0], 'y_min':thebox[1], 
                'x_min':thebox[2],'z_max':thebox[3], 'y_max':thebox[4], 'x_max':thebox[5],
               'z':z, 'y':y, 'x': x, 'z_range': seg.shape[-3], 'area':area}
        #ensure only things that aren't on the edge are chosen
        if (td['z_min']>1) and (td['y_min']>1) and (td['x_min']>1):
            tempdf = tempdf.append(td, ignore_index=True)
    
    if (len(tempdf)>0) and (tempdf.loc[tempdf['area'].idxmax(),'area']>50000):
        #return the largest object that isn't touching an edge    
        return tempdf.loc[tempdf['area'].idxmax()].to_dict()
    else:
        return {'cell':np.nan, 'z_min':np.nan, 'y_min':np.nan, 
                'x_min':np.nan,'z_max':np.nan, 'y_max':np.nan, 'x_max':np.nan,
               'z':np.nan, 'y':np.nan, 'x': np.nan, 'z_range': np.nan, 'area':np.nan}


def segandinfo_LLS(savedir,
                   celldir,
                   times,
                   interval,
                   decon,
                   orig_size: bool = False,
                   ):
    #get the file name
    image_name = os.path.basename(celldir).split('.')[0]
    
    #read the image
    czi = CziReader(celldir)
    imdata = czi.data
    
    #choose structure name based on file name
    if 'actin' in image_name:
        struct = 'actin'
    elif ('Hoechst' in image_name) or ('DNA' in image_name):
        struct = 'nucleus'
    elif 'myosin' in image_name:
        struct = 'myosin'
    else:
        struct = ''
    
    #get the pixel size from the metadata
    scale = metadata_funcs.getscale(czi)
    xyres = scale[0]
    zstep = scale[-1]
    #set image shape
    imshape = czi.shape
    #get the actual frame numbers from the original video
    first, last = metadata_funcs.frame_range_in_subset(czi)
    framelist = list(range(first-1, last))
    
    ### start by segmenting the large image to get all of the "primary" cells' bounding boxes for further cropping
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=60)
        imlist = [imdata[b,1,:,:,:] for b in range(imshape[0])]
        allseg = pool.map(quickcaaxseg, imlist)
        pool.close()
        pool.join()
    #get bounding boxes
    tempdata = []
    for f, seg in enumerate(allseg):
        im_labeled, n_labels = skimage.measure.label(
                                  seg, background=0, return_num=True,  )
        if n_labels > 1:
            im_props = skimage.measure.regionprops(im_labeled)
            tempdf = pd.DataFrame([])
            for count, prop in enumerate(im_props):
                area = prop.area
                af = prop.filled_area
                td = {'cell':count, 'area':area, 'area_fill':af}
                tempdf = tempdf.append(td, ignore_index=True)
                
            #select the part of the image with the largest area
            seg = im_labeled == int(tempdf.loc[tempdf['area'].idxmax()].cell+1)
            
            im_labeled, n_labels = skimage.measure.label(
                                      seg, background=0, return_num=True,  )

        #get the bounding boxes from all of the frames
        im_props = skimage.measure.regionprops(im_labeled)
        for count, prop in enumerate(im_props):
            print(f)
            z,y,x = prop.centroid
            thebox = prop.bbox
            area = prop.area
            convex_area = prop.convex_area
            tempdata.append({'cell':count, 'frame':f, 'actual_frame': framelist[f], 'z_min':thebox[0], 'y_min':thebox[1], 
                    'x_min':thebox[2],'z_max':thebox[3], 'y_max':thebox[4], 'x_max':thebox[5],
                   'z':z, 'y':y, 'x': x, 'z_range': seg.shape[-3],
                    'area':area, 'convex_area':convex_area})
    celldf = pd.DataFrame(tempdata)    
    
    #add actual times that were previously calculated from metadata
    celldf['time'] = times
    
    if __name__ ==  '__main__':
        # use multiprocessing to perform segmentation and x,y,z determination
        pool = multiprocessing.Pool(processes=60)
        results = []
        xy_buffer = 20
        z_buffer = 20
        for t, row in celldf.iterrows():
            
            xmincrop = int(max(0, row.x_min-xy_buffer))
            ymincrop = int(max(0, row.y_min-xy_buffer))
            zmincrop = int(max(0, row.z_min-z_buffer))

            zmaxcrop = int(min(row.z_max+z_buffer, imshape[-3]))
            ymaxcrop = int(min(row.y_max+xy_buffer, imshape[-2])+1)
            xmaxcrop = int(min(row.x_max+xy_buffer, imshape[-1])+1)
            
            tempim = imdata[int(row.frame),:,zmincrop:zmaxcrop,ymincrop:ymaxcrop,xmincrop:xmaxcrop].copy()
            

            
            pool.apply_async(LLSseg, args = (
                savedir,
                image_name,
                row.to_dict(),
                tempim,
                struct,
                xyres,
                zstep,
                decon,
                orig_size,
                imshape[-4:],
                ),             
                callback = collect_results)

        pool.close()
        pool.join()

    
        #list that will store all later trajectory info
        allcellinfo = []
        
        
        #deal with any frames that messed up
        if any([x == None for x in results]):
            ind = results.index(None)
            if len(results[:ind])<3:
                print(image_name + ' did not have enough segmented frames in movie')
            else:
                for r in ind:
                    del results[r]
        

        #aggregate the dataframe
        df = pd.DataFrame()
        for d in results:
            df = df.append(pd.DataFrame(d, columns = d.keys(), index=[0]))
        df = df.sort_values(by = 'frame').reset_index(drop=True)


        #make sure there are no gaps due to failed segmentations
        if any(df.frame.diff()>1):
            dft = df.reset_index(drop = True)
            runs = list()
            #######https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
            for k, g in groupby(enumerate(dft['frame']), lambda ix: ix[0] - ix[1]):
                currentrun = list(map(itemgetter(1), g))
                list.append(runs, currentrun)
            whichframes = np.array(max(runs, key=len), dtype=int)
            pullrows = dft[dft.frame.isin(whichframes)]
            df = pullrows.copy().reset_index(drop=True)
    
        #make sure there are at least three frames
        if len(df)<3:
            print(image_name + ' did not have enough segmented frames in movie')
        else:

            #set the k order for interpolation to the max possible
            if len(df)<6:
                kay = len(df)-1
            else:
                kay = 5
    
            pos = df[['x','y','z']]
            if bool(pos[pos.duplicated()].index.tolist()):
                ######### FIND CELL TRAJECTORY AND EULER ANGLES ################
                # if there is duplicate positions
                dups = pos[pos.duplicated()].index.tolist()
                pos_drop = pos.drop(dups, axis = 0)
                if pos_drop.shape[0]<3:
                    traj = np.zeros([1,len(pos),3])
                    trajsmo = pos.to_numpy().copy()
                else:
                    #get trajectories without the duplicates
                    tck, u = interpolate.splprep(pos_drop.to_numpy().T, k=kay, s=5)
                    yderv = interpolate.splev(u,tck,der=1)
                    traj = np.vstack(yderv).T
                    #get smoothened trajectory
                    ysmo = interpolate.splev(u,tck,der=0)
                    trajsmo = np.vstack(ysmo).T
                    #re-insert duplicate row that was dropped
                    for d, dd in enumerate(dups):
                        traj = np.insert(traj, dd, traj[dd-1,:], axis=0)
                        trajsmo = np.insert(trajsmo, dd, trajsmo[dd-1,:], axis=0)
    
            else:
                ######### FIND CELL TRAJECTORY AND EULER ANGLES ################
                #no duplicate positions
                #interpolate and get tangent at midpoint
                tck, b = interpolate.splprep(pos.to_numpy().T, k=kay, s=5)
                yderv = interpolate.splev(b,tck,der=1)
                traj = np.vstack(yderv).T
                #get smoothened trajectory
                ysmo = interpolate.splev(b,tck,der=0)
                trajsmo = np.vstack(ysmo).T
    
            ###add smoothened trajectory positions 
            #change x y z names in the dataframe
            df.rename(columns={"x": "x_raw", "y": "y_raw", "z": "z_raw"}, inplace = True)
            #add smoothened positions
            df['x'] = trajsmo[:,0]
            df['y'] = trajsmo[:,1]
            df['z'] = trajsmo[:,2]
    
            ############## Bayesian persistence and activity #################
            persistence, activity, speed = get_pa(df, interval)
            df['persistence'] = np.concatenate([np.array([np.nan]*2), persistence])
            df['activity'] = np.concatenate([np.array([np.nan]*2), activity])
            df['speed'] = np.concatenate([np.array([np.nan]), speed])
            df['avg_persistence'] = np.array([persistence.mean()]*(len(persistence)+2))
            df['avg_activity'] = np.array([activity.mean()]*(len(activity)+2))
            df['avg_speed'] = np.array([speed.mean()]*(len(speed)+1))
    
            #add directional autocorrelations
            df['directional_autocorrelation'] = DA_3D(df[['x','y','z']].to_numpy())
    
            #get the trajectory and the previous trajectory for each frame and 
            #save as an individual dataframe for each cell and frame
            for v, row in df.iterrows():
                if v==0:
                    row['Prev_Trajectory_X'] = np.nan
                    row['Prev_Trajectory_Y'] = np.nan
                    row['Prev_Trajectory_Z'] = np.nan
                    row['Trajectory_X'] = traj[v,0]
                    row['Trajectory_Y'] = traj[v,1]
                    row['Trajectory_Z'] = traj[v,2]
                    row['Turn_Angle'] = np.nan
    #                             pd.DataFrame(row.to_dict(),index=[0]).to_csv(savedir + row.cell + '_cell_info.csv')
                    allcellinfo.append(row)
                if v>0:
                    row['Prev_Trajectory_X'] = traj[v-1,0]
                    row['Prev_Trajectory_Y'] = traj[v-1,1]
                    row['Prev_Trajectory_Z'] = traj[v-1,2]
                    row['Trajectory_X'] = traj[v,0]
                    row['Trajectory_Y'] = traj[v,1]
                    row['Trajectory_Z'] = traj[v,2]
                    row['Turn_Angle'] = angle_distance(traj[v-1,0], traj[v-1,1], traj[v-1,2], traj[v,0], traj[v,1], traj[v,2])
    #                             pd.DataFrame(row.to_dict(),index=[0]).to_csv(savedir + row.cell + '_cell_info.csv')
                    allcellinfo.append(row)

    
    
    return pd.DataFrame(allcellinfo)



