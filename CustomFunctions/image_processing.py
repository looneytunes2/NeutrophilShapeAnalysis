# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:39:57 2025

@author: Aaron
"""

import numpy as np
import os
import pandas as pd
import math
import re
import multiprocessing
from itertools import groupby
from operator import itemgetter
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from aicsimageio.readers.czi_reader import CziReader
from scipy.spatial import KDTree, distance
from scipy import interpolate
from CustomFunctions.segment_cells2short import seg_confocal_40x_memonly_fromslices
from CustomFunctions.persistance_activity import get_pa, DA_3D
from CustomFunctions import shparam_mod, metadata_funcs, segment_LLS
from CustomFunctions.track_functions import segment_caax_tracks_confocal_40x_fromsingle
from CustomFunctions.file_management import multicsv


def collect_results(result):
    """Uses apply_async's callback to setup up a separate Queue for each process.
    This will allow us to collect the results from different threads."""
    results.append(result)

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




def segment_whole_images(
        raw_dir, #parent directory for the folders from different imaging days
        foldlist, #the dates on the folders from different imaging days
        trackdir, #where to save the segmented tracking images
        xyres,
        zstep,
        fullimshape, #shape of the full movie in TZYX format
        ):

    
    for f in foldlist:
        ims = [o for o in os.listdir(raw_dir+f+'/')]
        for i in ims:
    
            #make the trackdir if it doesn't exist
            if not os.path.exists(trackdir+i+'/'):
                os.makedirs(trackdir+i+'/')
            #directory of all the slices for this movie
            imdir = raw_dir+f+'/'+i+'/Default/'
    
    
            ##### automatically detect image size
            #         #sort the list of images and get the last one
            #         last = sorted([o for o in os.listdir(imdir) if o.endswith('tif')])[-1]
            #         #get stats about acquisition
            #         cc,pos,maxtime,maxslices = [x for x in re.findall('\d*', last) if len(x)>1]
            #         maxtime = int(maxtime)+1
            #         maxslices = int(maxslices)+1
            #         #open that image to get x,y size
            #         shape = tiff_reader.TiffReader(imdir+last).shape
    
            results = []
            # use multiprocessing to perform segmentation and x,y,z determination
            pool = multiprocessing.Pool(processes=60)
            for t in range(fullimshape[0]):
                pool.apply_async(segment_caax_tracks_confocal_40x_fromsingle, args=(imdir,
                                                                                    fullimshape[-3:],
                                                                                    xyres,
                                                                                    zstep,
                                                                                    t, ), 
                                 callback=collect_results)
            pool.close()
            pool.join()

    
            #organize the semented frames into a segmented stack
            segmented_img = np.zeros((fullimshape[0],
                                     results[0][3][-3],
                                     results[0][3][-2],
                                     results[0][3][-1]))
            for r in results:
                fr = r[2]
                segmented_img[fr,:,:,:] = r[1]    
    
            #covert to more compact data type
            segmented_img = segmented_img.astype(np.uint8)
    
            #save the segmented image
            OmeTiffWriter.save(segmented_img, trackdir+i+'/'+i+'_segmented.ome.tiff', dim_order = "TZYX", overwrite_file=True)
    
    
            #save the skimage region props
            df = pd.DataFrame()
            for d in results:
                df = df.append(pd.DataFrame(d[0], columns = ['cell', 
                             'frame', 'z_min', 'y_min', 
                            'x_min','z_max', 'y_max', 'x_max',
                           'z', 'y', 'x', 'z_range',
                           'area', 'convex_area', 'extent',
                           'minor_axis_length', 'major_axis_length',
                            'intensity_avg', 'intensity_max', 'intensity_std']))
            df = df.sort_values(by = ['frame','cell'])
            df.to_csv(trackdir+i+'/'+i+'_region_props.csv')
    
            print(f'Finished processing {i}')



############### SEGMENT AND SAVE CELLS ################################
def segment_and_crop(
        mindir,
        raw_dir, #directory with original images (saved as individual slices)
        xyres, #xy resolution of images
        zstep, #z resolution of images
        xy_buffer, #amount to buffer cropped images in xy
        z_buffer, #amount to buffer cropped images in z
        stackshape, #shape of one z stack in pixels (z,y,x) format
        ):

    folder_fl = mindir + 'Tracking_Images/'
    filelist_fl = [f for f in os.listdir(folder_fl)]
    savedir = mindir + 'processed_images/'
    posdir = mindir + 'position_info/'
    #make the savedir if it doesn't exist
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    if not os.path.exists(posdir):
        os.makedirs(posdir)
    

    for u in filelist_fl:
    
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
        #     #first rows that have super long distances from previous cell, so set them to 0
        #     df_track.loc[df_track.groupby('cell').head(1).index,'dist'] = 0
    
        ############ replace unrealistic jumps in distance ##############
        for x in df_track[df_track.dist>4].index.values:
            df_track.loc[x,'dist'] = df_track.dist.mean()
    
        ############## find euclidean distance #############
        euclid = pd.DataFrame([])
        for i, cell in df_track.groupby('cell'):
            FL = cell.iloc[[0,-1]]
            euc_dist = distance.pdist(FL[['x','y','z']])
            euclid = euclid.append({'cell':cell.cell.iloc[0], 'euc_dist':euc_dist[0]}, ignore_index = True)
        cellsmorethan = euclid.loc[euclid['euc_dist']>10, 'cell']
        df_track = df_track[df_track.cell.isin(cellsmorethan)]
    
        #     ########remove "slow"/dead cells############
        #     #sum distances
        #     df_track_distsums = df_track.groupby('cell').sum()
        #     df_track_distsums = df_track_distsums.add_suffix('_sum').reset_index()
    
        #     #grab only cells with sums above a threshold distance
        #     cellsmorethan = df_track_distsums.loc[df_track_distsums['dist_sum']>5, 'cell']
        #     df_track = df_track[df_track.cell.isin(cellsmorethan)]
    
    
        ########remove edge cells############
        #only grab rows that aren't zero in z_min
        df_track = df_track.loc[df_track['x_min'] !=0 ]
        df_track = df_track.loc[df_track['y_min'] !=0 ]
        df_track = df_track.loc[df_track['z_min'] !=0 ]
        #remove rows where z_max matches z_range
        df_track = df_track.loc[df_track['x_max'] < stackshape[-1]]
        df_track = df_track.loc[df_track['y_max'] < stackshape[-2]]
        df_track = df_track.loc[df_track['z_max'] != (df_track['z_range'])]
    
    
        ##########remove small things that are likely dead cells or parts of cells###########
        df_track = df_track[df_track['area'] > 4000 ]
        #reset index after dropping all the rows
        df_track = df_track.reset_index(drop = True)
    
    
        # ######## remove cells that touch ###########    
        # to_remove = []
        # for i, cell in df_track.groupby('cell'):
        #     if i>0:
        #         changes = abs(cell['convex_area'].pct_change())
        #         largerthan = changes[changes>0.75]
        #         if largerthan.empty == False:
        #             to_remove.extend(largerthan.index.to_list())
    
    
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
        # df_track = df_track.drop(to_remove)
    
    
    
        if df_track.empty == False:
            for i, cells in df_track.groupby('cell'):
                cell = cells.reset_index(drop = True)
                # use multiprocessing to perform segmentation and x,y,z determination
                pool = multiprocessing.Pool(processes=60)
                results = []
                for t, row in cell.iterrows():

                    tdir = raw_dir +u.split('_')[0]+'/' +u+'/Default/'

                    xmincrop = int(max(0, row.x_min-xy_buffer))
                    ymincrop = int(max(0, row.y_min-xy_buffer))
                    zmincrop = int(max(0, row.z_min-z_buffer))

                    zmaxcrop = int(min(row.z_max+z_buffer, stackshape[-3]))
                    ymaxcrop = int(min(row.y_max+xy_buffer, stackshape[-2])+1)
                    xmaxcrop = int(min(row.x_max+xy_buffer, stackshape[-1])+1)
                    
                    #croparray
                    croparr = np.array([xmincrop,xmaxcrop,ymincrop,ymaxcrop,zmincrop,zmaxcrop])
                    #run the segmentation function
                    result = pool.apply_async(seg_confocal_40x_memonly_fromslices, args = (
                        tdir,
                        stackshape,
                        row,
                        u,
                        savedir,
                        xyres,
                        zstep,
                        croparr, 
                        ))
                    results.append(result)
                pool.close()
                pool.join()

                print(f'Done segmenting {u} cell {cell.cell.iloc[0]}')
                
                #get results
                results = [r.get() for r in results]
                #make sure there's no None results from failed segmentations
                results = [x for x in results if x!=None]
                #aggregate the dataframe
                df = pd.DataFrame(results).sort_values(by = 'frame').reset_index(drop=True)
                #add cell ID before saving
                df['CellID'] = [df.cell.iloc[0].split('_frame')[0]]*len(df)
                df.to_csv(posdir+df.cell.iloc[0].split('_frame')[0]+'_cellpos.csv')



########## GET TRAJECTORIES FROM POSITION INFO 
def get_smooth_trajectories(
        mindir,
        time_interval,
        ):
    
    #define directory stuff
    datadir = mindir + 'Data_and_Figs/'
    csvdir = mindir + 'processed_data/'
    posdir = mindir + 'position_info/'
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    if not os.path.exists(csvdir):
        os.makedirs(csvdir)
    
        
    #combine all of the cell csvs into one dataframe
    fileslist = [x for x in os.listdir(posdir) if x.endswith('.csv')]
    csvlist = [posdir+i for i in fileslist]
    with multiprocessing.Pool(processes=60) as pool:
        celllist = pool.map(multicsv, csvlist)
    cellinfo = pd.concat(celllist).reset_index(drop=True)
    #add time to the confocal data
    if 'time' not in cellinfo.columns.to_list():
        cellinfo['time'] = cellinfo['frame'].values * time_interval
    
    for i, df in cellinfo.groupby('CellID'):
    
        #first get dataframe in time order
        df = df.sort_values(by = 'time').reset_index(drop=True)
    
        #make sure there are no gaps due to failed segmentations
        if any(abs(df.time.diff())>time_interval):
            diff = df.time.diff()
            difflist = [0]
            difflist.extend(diff[diff>time_interval].index.to_list())
            runs = []
            for x in range(len(difflist)-1):
                runs.append(list(range(difflist[x], difflist[x+1])))
        else:
            runs = [df.index.to_list()]
    
        #save the df in case it gets broken up later    
        brokendf = df.copy()
    
        for r in runs:
            if len(r)>2:
                df = brokendf.iloc[r].reset_index(drop=True)
                #set the k order for interpolation to the max possible
                if len(df)<6:
                    kay = len(df)-1
                else:
                    kay = 5

    
                #do speed and trajectory stuff
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
                    tck, b = interpolate.splprep(pos.to_numpy().T, k=kay, s=120)
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
                persistence, activity, speed = get_pa(df, time_interval)
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
                        pd.DataFrame(row.to_dict(),index=[0]).to_csv(csvdir + row.cell + '_cell_info.csv')
    
                    if v>0:
                        row['Prev_Trajectory_X'] = traj[v-1,0]
                        row['Prev_Trajectory_Y'] = traj[v-1,1]
                        row['Prev_Trajectory_Z'] = traj[v-1,2]
                        row['Trajectory_X'] = traj[v,0]
                        row['Trajectory_Y'] = traj[v,1]
                        row['Trajectory_Z'] = traj[v,2]
                        if all(traj[v-1,:] == traj[v,:]):
                            row['Turn_Angle'] = 0
                        else:
                            row['Turn_Angle'] = angle_distance(traj[v-1,0], traj[v-1,1], traj[v-1,2], traj[v,0], traj[v,1], traj[v,2])
                        pd.DataFrame(row.to_dict(),index=[0]).to_csv(csvdir + row.cell + '_cell_info.csv')
    
        print(f'Finished tracking cell {i}')
    




############ FIND WIDTH ROTATIONS THAT DEPEND ON PREVIOUS FRAMES TO LIMIT ROTATION FLIPPING ################
def get_normal_rotations(
        mindir, 
        xyres,
        zstep,
        align_method = 'trajectory',
        sigma = 0,
        ):
    
    
    imdir = mindir + 'processed_images/'
    datadir = mindir + 'Data_and_Figs/'
    csvdir = mindir + 'processed_data/'
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    ### get the list of unique cells that we have trajectory info for
    imlist = []
    for o in os.listdir(csvdir):
        cellid = o.split('_frame')[0]
        if cellid not in imlist:
            imlist.append(cellid)

    ### loop through the unique cells and open the segmented images to rotate
    ### each mesh until you find the rotation angle for the widest axis perpendicular
    ### to the trajectory
    trajinfolist = os.listdir(csvdir)
    segimlist = [x for x in os.listdir(imdir) if 'segmented' in x]
    allresults = []
    for i in imlist:
        cellframelist = [u.split('_cell_info')[0] for u in trajinfolist if u.split('_frame')[0] == i]
        cellseglist = [j for j in segimlist if j.split('_segmented')[0] in cellframelist]
        results = []
        pool = multiprocessing.Pool(processes=60)
        for y in cellseglist:
            #get path to segmented image
            impath = imdir + y
            #put in the pool
            result = pool.apply_async(shparam_mod.find_normal_width_peaks, args = (
                impath,
                csvdir, 
                xyres,
                zstep,
                sigma,
                align_method,
                ))
            results.append(result)
        pool.close()
        pool.join()
        
        #get results
        results = [r.get() for r in results]
        results.sort(key=lambda x: float(re.findall('(?<=frame_)\d*', x[0])[0]))
        tempframe = pd.DataFrame(results, columns = ['cell','Width_Peaks'])
        tempframe['frame'] = [float(re.findall('(?<=frame_)\d*', x[0])[0]) for x in results]
        
        runs = list()
        #######https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
        for k, g in groupby(enumerate(tempframe['frame']), lambda ix: ix[0] - ix[1]):
            currentrun = list(map(itemgetter(1), g))
            list.append(runs, currentrun)
    
        
        #find the minima in each frame that are closest to the minimum chosen in the last frame
        #aka the one that results in the least amount of consecutive rotation
        fullminlist = []
        for xx in runs:
            runframe = tempframe[tempframe.frame.isin(xx)]
            wplist = runframe.Width_Peaks.to_list()
            seeds = []
            allallmins = []
            #for all the starting peaks find the least different rotations through time
            for s in wplist[0]:
                allmins = [s]
                for wp in wplist[1:]:
                    if bool(len(wp) == 0):
                        allmins.append(allmins[-1])
                    else:
                        allmins.append(wp[np.argmin(abs(wp-(allmins[-1])))])
                allallmins.append(allmins)
                seeds.append(np.sum(abs(np.diff(allmins))))
            #add rotations of current run to the list
            fullminlist.extend(allallmins[np.argmin(seeds)])  
        
        
        #add all mins to tempframe
        tempframe['Closest_minimums'] = fullminlist
        
        allresults.append(tempframe)
        
        print('Finished '+ i)
    
    #save the shape metrics dataframe
    bigdf = pd.concat(allresults)
    bigdf.to_csv(datadir + 'Closest_Width_Peaks.csv')




def seg_to_mesh(
        mindir, #base directory with all of the different data folders and files
        xyres, #xy resolution
        zstep, # z resolution
        align_method = 'trajectory', #how to align the cells
        norm_rot = 'provided', #how to perform the normal rotation
        l_order = 10, # L order for SH coefficients
        nisos = [1,63], # list of shells to calculate in PILRs
        pilr_method = 'none', #how to calculate PILRs
        sigma = 0, #how much to smoothen image before turning to mesh
        ):

    #make dirs if it doesn't exist
    datadir = mindir + 'Data_and_Figs/'
    csvdir = mindir + 'processed_data/'
    imdir = mindir + 'processed_images/'
    #make dirs if it doesn't exist
    meshf = mindir+'Meshes/'  
    if not os.path.exists(meshf):
        os.makedirs(meshf)
    pilrf = mindir+'PILRs/'
    if not os.path.exists(pilrf):
        os.makedirs(pilrf)

    
    if norm_rot == 'provided':
        widthpeaks = pd.read_csv(datadir + 'Closest_Width_Peaks.csv', index_col = 0)
        
    #get all segmented images that were analyzed
    datalist = [x.split('_cell_info.csv')[0] for x in os.listdir(csvdir)]
    imlist = [x for x in os.listdir(imdir) if x.endswith('segmented.tiff') and x.split('_segmented.tiff')[0] in datalist]
    
    errorlist = []
    start = 0
    stop = 300
    allresults = []
    while start<len(imlist):
        print(f'Finished {start}, starting {start}-{stop}')
        results = []
        pool = multiprocessing.Pool(processes=60)
        for i in imlist[start:stop]:
            
            #choose structure name based on file name
            if 'actin' in i:
                str_name = 'actin'
            elif ('Hoechst' in i) or ('DNA' in i):
                str_name = 'nucleus'
            elif 'myosin' in i:
                str_name = 'myosin'
            else:
                str_name = ''
            
            #assign the normal rotation value for that particular cell
            if (norm_rot == 'provided') or (type(norm_rot) == float):
#                 try:
                norm_rot = float(widthpeaks[widthpeaks.cell == i.split('_segment')[0]]['Closest_minimums'].values[0])
#                 #exception for if 
#                 except:
#                     norm_rot = 'widest weighted'
                    
            #put in the pool
            result = pool.apply_async(shparam_mod.shcoeffs_and_PILR_nonuc, args = (
                i,
                mindir,
                xyres,
                zstep,
                str_name,
                errorlist,
                norm_rot,
                l_order,
                nisos,
                pilr_method,
                sigma,
                align_method,
                ))
            results.append(result)
        pool.close()
        pool.join()
        #get results and append to the larger results list
        results = [r.get() for r in results]
        allresults.extend(results)
        
        start = stop + 1
        stop = stop + 1000
        if stop>len(imlist):
            stop = len(imlist)
    
    errorlist = []
    bigdf = pd.DataFrame()
    
    for r in allresults:
        
    
        Shape_Stats = pd.DataFrame([r[0].values()],
                                      columns = list(r[0].keys()))
        cell_coeffs = pd.DataFrame([r[1].values()],
                                   columns = list(r[1].keys()))
    
        bigdf = bigdf.append(pd.concat([Shape_Stats,cell_coeffs], axis=1))
    
        errorlist.extend(r[2])
    
    
    #save the shape metrics dataframe
    bigdf = bigdf.set_index('cell')
    bigdf.to_csv(datadir + 'Shape_Metrics.csv')
    
    #save list of cells that don't have centroid in shape
    pd.Series(errorlist).to_csv(datadir + 'ListToExclude.csv')





################ SEGMENT AND TRACK CELLS FROM MANUALLY CROPPED LLS MOVIES #############
def segment_and_crop_LLS_manual(
        cellstr,#the name of the unique cell being cropped and segmented across multiple videos
        savedir, #direcetory to save images
        posdir, #directory to save position data
        croppeddir, #directory where all of the cropped LLS images live
        decon = True, #are these images deconvolved?
        orig_size = False, #should we save the images at their original size?
        xy_buffer = 25, #crop buffer in x-y
        z_buffer = 25, #crrop buffer in z
        hilo = True, #whether or not to do multiple thresholds for segmenting secondary signals
        ):
    #get all of the images from a particular cell I was following
    curimlist = [x for x in os.listdir(croppeddir) if cellstr in x]
    #find the total number of cells I cropped while following the cell of interest
    cellnums = list(set([re.findall(r'Subset-(\d+)',x)[0] for x in curimlist]))
    cellnums.sort()
    for s in cellnums:
        #list to put all dataframes from all subsets
        dflist = []
        #get all the images of a given cell
        curcell = [x for x in curimlist if f'Subset-{s}' in x]
        #sort the current cell to be in chronological order
        curcell.sort(key=lambda x: float(re.findall(r'(\d+)-Subset', x)[0]))
        for n, c in enumerate(curcell):
            celldir = croppeddir + c
            #open the image
            czi = CziReader(celldir)
            imdata = czi.data
            #absolute timepoint of first image
            if n == 0:
                timezero = metadata_funcs.adjustedstarttime(czi)
            #get time interval and number of frames and start time
            ti = metadata_funcs.gettimeinterval(czi)
            fn = metadata_funcs.framesinsubset(czi)
            ast = metadata_funcs.adjustedstarttime(czi)

            #get all the times at the current frame since the cell was initially observed
            times = [int(ast - timezero + (f*ti)) for f in range(fn)]

            #segment the cells and return the position info
            #get the file name
            image_name = os.path.basename(celldir).split('.')[0]

            #choose structure name based on file name
            if 'actin' in image_name:
                struct = 'actin'
            elif ('Hoechst' in image_name) or ('DNA' in image_name):
                struct = 'nucleus'
            elif 'mysoin' in image_name:
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

        
            #get the crops for each frame based on coarse thresholding
            celldf = segment_LLS.getbb_movie(imdata[:,1,:,:,:])
            celldf['actual_frame'] = framelist
            celldf['frame'] = list(range(len(celldf)))
            #add actual times that were previously calculated from metadata
            celldf['time'] = times
            #drop any na frames that weren't able to find bounding boxes
            celldf = celldf.dropna().reset_index(drop=True)
            
            # use multiprocessing to perform segmentation and x,y,z determination
            pool = multiprocessing.Pool(processes=60)
            results = []
            for t, row in celldf.iterrows():
                
                
                #segment the cropped images
                result = pool.apply_async(segment_LLS.LLSseg, args = (
                    savedir,
                    image_name,
                    row.to_dict(),
                    imdata[int(row.frame),:,:,:,:],
                    struct,
                    xyres,
                    zstep,
                    decon,
                    orig_size,
                    imshape[-4:],
                    xy_buffer,
                    z_buffer,
                    hilo,
                    ))
                results.append(result)
            pool.close()
            pool.join()

            #print progress
            print('Finished segmenting cropped images of '+c)
            
            #get results
            results = [r.get() for r in results]
            #deal with any frames that messed up
            bef = len(results)
            results = [l for l in results if l is not None]
            af = len(results)
            if af<bef:
                print(str(bef-af)+' frames dropped from ' + image_name)
            if af>0:
                #aggregate the dataframe
                df = pd.DataFrame()
                for d in results:
                    df = df.append(pd.DataFrame(d, columns = d.keys(), index=[0]))
                df = df.sort_values(by = 'frame').reset_index(drop=True)
                dflist.append(df)
            else:
                print(image_name + ' did not have enough segmented frames in movie')
        #combine all of the subset dataframes and save
        fulldf = pd.concat(dflist).reset_index(drop=True)
        fulldf['CellID'] = [cellstr+f'_{s}']*len(fulldf)
        fulldf.to_csv(posdir + cellstr + f'_{s}_cellpos.csv')


