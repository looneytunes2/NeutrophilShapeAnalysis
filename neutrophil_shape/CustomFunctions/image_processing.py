# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:39:57 2025

@author: Aaron
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path
import multiprocessing
import tifffile
from aicspylibczi import CziFile
from scipy.spatial import KDTree, distance
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
from .segment_cells2short import seg_confocal_40x_memonly_fromslices
from .persistance_activity import get_pa, DA_3D
from . import shparam_mod, metadata_funcs, segment_LLS
from .shtools_mod import read_polydata
from .track_functions import segment_caax_tracks_confocal_40x_fromsingle
# from .PILRagg import read_pilr_regions
from .utils import get_consecutive_timepoints, angle3D, align_vec_to_xaxis_euler
from neutrophil_shape.config.models import Config
from tqdm import tqdm



def segment_whole_images(
        raw_dir: Path,  # parent directory for the folders from different imaging days
        foldlist: list,  # the dates on the folders from different imaging days
        imdir: Path,  # where to save the segmented tracking images
        config: Config,  # Config class
):
    #define tracking image subd directory
    trackdir = imdir / 'Tracking_Images'

    for f in foldlist:
        ims = [o for o in raw_dir.joinpath(f).glob('*') if o.is_dir()]
        # create the actual list of image directories including if there is
        # multiple positions
        imagedirs = [x for i in ims for x in i.glob('*') if x.is_dir()]
        for imdir in imagedirs:
            # define the name of the acquisition based on whether there are
            # multiple positions
            imagename = imdir.parent.name
            # make the trackdir if it doesn't exist
            if not trackdir.joinpath(imagename).exists():
                trackdir.joinpath(imagename).mkdir(parents=True)

            # automatically detec image shape based on slice names
            shapestring = sorted(imdir.glob('*.tif'))[-1].name
            shapetime = int(re.findall(r'(?<=time)\d+', shapestring)[0])
            shapez = int(re.findall(r'(?<=_z)\d+', shapestring)[0])
            # combine and add 1 because of zero index
            fullimshape = [int(shapetime+1),
                           int(shapez+1)] + config.confocal.stackshape[-2:]

            results = []
            # use multiprocessing to perform segmentation and x,y,z determination
            pool = multiprocessing.Pool(processes=60)
            for t in range(fullimshape[0]):
                result = pool.apply_async(segment_caax_tracks_confocal_40x_fromsingle, args=(
                    imdir,
                    fullimshape[-3:],
                    config.confocal.xyres,
                    config.confocal.zstep,
                    t, ))
                results.append(result)

            pool.close()
            pool.join()
            results = [r.get() for r in results]

            # organize the semented frames into a segmented stack
            segmented_img = np.zeros((fullimshape[0],
                                     results[0][3][-3],
                                     results[0][3][-2],
                                     results[0][3][-1]))
            for r in results:
                fr = r[2]
                segmented_img[fr, :, :, :] = r[1]

            # covert to more compact data type
            segmented_img = segmented_img.astype(np.uint8)

            # save the segmented image
            tifffile.imwrite(trackdir.joinpath(
                                   imagename, imagename+'_segmented.ome.tiff'),
                            segmented_img)
            
            # save the skimage region props
            df = pd.DataFrame()
            for d in results:
                df = df.append(pd.DataFrame(d[0], columns=['cell',
                                                           'frame', 'z_min', 'y_min',
                                                           'x_min', 'z_max', 'y_max', 'x_max',
                                                           'z', 'y', 'x', 'z_range',
                                                           'area', 'convex_area', 'extent',
                                                           'minor_axis_length', 'major_axis_length',
                                                           'intensity_avg', 'intensity_max', 'intensity_std']))
            df = df.sort_values(by=['frame', 'cell'])
            df.to_csv(trackdir.joinpath(
                imagename, imagename+'_region_props.csv'))

            print(f'Finished processing {imagename}')


############### SEGMENT AND SAVE CELLS ################################
def segment_and_crop_confocal(
        raw_dir,  # directory with original images (saved as individual slices)
        imdir,  # directory to access tracking data and where processed data will be saved
        config, # Config class
):

    folder_fl = imdir.joinpath('Tracking_Images')
    filelist_fl = [f for f in folder_fl.glob('*') if f.is_dir()]
    procimdir = imdir.joinpath('processed_images')
    posdir = imdir.joinpath('position_info')
    meshdir = imdir.joinpath('meshes')
    # make the savedir if it doesn't exist
    if not procimdir.exists():
        procimdir.mkdir(parents=True)
    if not posdir.exists():
        posdir.mkdir(parents=True)
    if not meshdir.exists():
        meshdir.mkdir(parents=True)

    ## load a few configuration parameters from the config file
    xyres = config.confocal.xyres  # xy resolution of images
    zstep = config.confocal.zstep  # z resolution of images
    xy_buffer = config.confocal.xy_buffer  # amount to buffer cropped images in xy
    z_buffer = config.confocal.z_buffer  # amount to buffer cropped images in z
    stackshape = config.confocal.stackshape  # shape of one z stack in pixels (z,y,x) format
    whatseg = config.confocal.whatseg  # what segmentation function to use for which cells

    for u in filelist_fl:

        ################## align trackmate data with region props data ################
        rpcsv = next(folder_fl.joinpath(u).glob('*region_props.csv'))
        rp = pd.read_csv(folder_fl.joinpath(u, rpcsv), index_col=0)
        tmcsv = next(folder_fl.joinpath(u).glob('*TrackMateLog.csv'))
        tm = pd.read_csv(folder_fl.joinpath(u, tmcsv))
        # fix trackmate columns to get names right and units in microns
        tm['x'] = tm.POSITION_X*xyres
        tm['y'] = tm.POSITION_Y*xyres
        tm['z'] = tm.POSITION_Z*zstep
        # make kdtree and query with trackmate log
        kd = KDTree(rp[['frame', 'x', 'y', 'z']].to_numpy())
        dd, ii = kd.query(tm[['FRAME', 'x', 'y', 'z']])
        df_track = pd.concat([tm.drop(columns=['POSITION_X', 'POSITION_Y', 'POSITION_Z']),
                              rp.iloc[ii].drop(columns=['frame', 'x', 'y', 'z', 'cell']).reset_index(drop=True)], axis=1)
        # add some identifiers and rename FRAME
        df_track = df_track.rename(columns={'FRAME': 'frame'})
        df_track['CellID'] = u.name + '_cell_' + df_track.TRACK_ID.astype(str)
        df_track['cell'] = df_track.CellID + '_frame_' + \
            df_track.frame.astype(int).astype(str)
        df_track.drop(columns=['TRACK_ID'], inplace=True)

        ############## find euclidean distance #############
        euclid = []
        for i, cell in df_track.groupby('CellID'):
            cell = cell.sort_values('frame').reset_index(drop=True)
            FL = cell.iloc[[0, -1]]
            euc_dist = distance.pdist(FL[['x', 'y', 'z']])
            euclid.append(
                {'CellID': cell.CellID.iloc[0], 'euc_dist': euc_dist[0]}
                ) 
        eucliddf = pd.DataFrame(euclid)
        cellsmorethan = eucliddf.loc[eucliddf['euc_dist'] > 10, 'CellID']
        df_track = df_track[df_track.CellID.isin(cellsmorethan)]

        ########remove edge cells############
        # only grab rows that aren't zero in z_min
        df_track = df_track.loc[df_track['x_min'] != 0]
        df_track = df_track.loc[df_track['y_min'] != 0]
        df_track = df_track.loc[df_track['z_min'] != 0]
        # remove rows where z_max matches z_range
        df_track = df_track.loc[df_track['x_max'] < stackshape[-1]]
        df_track = df_track.loc[df_track['y_max'] < stackshape[-2]]
        df_track = df_track.loc[df_track['z_max'] != (df_track['z_range'])]

        ##########remove small things that are likely dead cells or duplicate cells###########
        if whatseg == 'hl60':
            df_track = df_track[df_track['area'] > 4000]
        elif whatseg == 'el4':
            sizemeans = df_track.groupby('CellID').area.mean().reset_index()
            smallorbig = sizemeans[(sizemeans['area'] < 9000) | (
                sizemeans['area'] > 50000)].CellID.to_list()
            df_track = df_track[~df_track.CellID.isin(smallorbig)]

        # reset index after dropping all the rows
        df_track = df_track.reset_index(drop=True)

        if df_track.empty == False:
            for i, cell in df_track.groupby('CellID'):
                cell = cell.reset_index(drop=True)
                # use multiprocessing to perform segmentation and x,y,z determination
                pool = multiprocessing.Pool(processes=60)
                results = []
                for t, row in cell.iterrows():

                    tdir = raw_dir.joinpath(
                        u.name.split('_')[0], u.name, 'Default')

                    xmincrop = int(max(0, row.x_min-xy_buffer))
                    ymincrop = int(max(0, row.y_min-xy_buffer))
                    zmincrop = int(max(0, row.z_min-z_buffer))

                    zmaxcrop = int(min(row.z_max+z_buffer, stackshape[-3]))
                    ymaxcrop = int(min(row.y_max+xy_buffer, stackshape[-2])+1)
                    xmaxcrop = int(min(row.x_max+xy_buffer, stackshape[-1])+1)

                    # croparray
                    croparr = np.array(
                        [xmincrop, xmaxcrop, ymincrop, ymaxcrop, zmincrop, zmaxcrop])
                    # run the segmentation function
                    result = pool.apply_async(seg_confocal_40x_memonly_fromslices, args=(
                        tdir,
                        stackshape,
                        row,
                        procimdir,
                        xyres,
                        zstep,
                        croparr,
                        whatseg,
                    ))
                    results.append(result)
                pool.close()
                pool.join()

                print(f'Done segmenting {cell.CellID.iloc[0]}')

                # get results
                results = [r.get() for r in results]
                # make sure there's no None results from failed segmentations
                results = [x for x in results if x != None]
                if len(results) > 0:
                    # aggregate the dataframe
                    df = pd.DataFrame(results).sort_values(
                        by='frame').reset_index(drop=True)
                    # save
                    df.to_csv(posdir.joinpath(
                        df.CellID.iloc[0]+'_cellpos.csv'))


# GET TRAJECTORIES FROM POSITION INFO
def get_smooth_trajectories(
        imdir: Path,  # where to find the segmented images and position information
        config: Config,
        microscope: str,  # what microscope the data is from to determine which config parameters to use
):
    #save some variables from the config
    if microscope == 'confocal':
        mcon = config.confocal
    elif microscope == 'lls':
        mcon = config.lls
    time_interval = mcon.time_interval  # time interval between frames of movies
    smooth_factor = config.common.smooth_factor  # "s" parameter in the interpolate.splprep function

    # define directory stuff
    csvdir = imdir.joinpath('smooth_traj')
    posdir = imdir.joinpath('position_info')
    if not csvdir.exists():
        csvdir.mkdir(parents=True)

    # combine all of the cell csvs into one dataframe
    csvlist = [posdir.joinpath(x) for x in posdir.glob('*.csv')]
    celllist = []
    for c in csvlist:
        celllist.append(pd.read_csv(c, index_col=0))
    cellinfo = pd.concat(celllist).reset_index(drop=True)

    # add time to the confocal data
    if 'time' not in cellinfo.columns.to_list():
        cellinfo['time'] = cellinfo['frame'].values * time_interval

    for i, df in cellinfo.groupby('CellID'):

        # first get dataframe in time order and consecution timepoints
        df, runs = get_consecutive_timepoints(
            df[~df.x.isna()], 'time', time_interval)

        # save the df in case it gets broken up later
        brokendf = df.copy()

        for r in runs:
            if len(r) > 2:
                df = brokendf.iloc[r].reset_index(drop=True)
                # set the k order for interpolation to the max possible
                if len(df) < 6:
                    kay = len(df)-1
                else:
                    kay = 5

                # do speed and trajectory stuff
                pos = df[['x', 'y', 'z']]
                if bool(pos[pos.duplicated()].index.tolist()):
                    ######### FIND CELL TRAJECTORY AND EULER ANGLES ################
                    # if there is duplicate positions
                    dups = pos[pos.duplicated()].index.tolist()
                    pos_drop = pos.drop(dups, axis=0)
                    # if dropping the duplicates leads to less that three positions,
                    # just continue with the duplicates but don't smoothen
                    if pos_drop.shape[0] < 3:
                        traj = pos.to_numpy().copy()
                        trajsmo = pos.to_numpy().copy()
                    else:
                        # get trajectories without the duplicates
                        tck, u = interpolate.splprep(
                            pos_drop.to_numpy().T, k=kay, s=smooth_factor)
                        yderv = interpolate.splev(u, tck, der=1)
                        # get smoothened trajectory
                        traj = np.vstack(yderv).T
                        # get smoothened position
                        ysmo = interpolate.splev(u, tck, der=0)
                        trajsmo = np.vstack(ysmo).T
                        # re-insert duplicate row that was dropped
                        for d, dd in enumerate(dups):
                            traj = np.insert(traj, dd, traj[dd-1, :], axis=0)
                            trajsmo = np.insert(
                                trajsmo, dd, trajsmo[dd-1, :], axis=0)

                else:
                    ######### FIND CELL TRAJECTORY AND EULER ANGLES ################
                    # no duplicate positions
                    # interpolate and get tangent at midpoint
                    tck, b = interpolate.splprep(
                        pos.to_numpy().T, k=kay, s=smooth_factor)
                    yderv = interpolate.splev(b, tck, der=1)
                    traj = np.vstack(yderv).T
                    # get smoothened trajectory
                    ysmo = interpolate.splev(b, tck, der=0)
                    trajsmo = np.vstack(ysmo).T

                # add smoothened trajectory positions
                # change x y z names in the dataframe
                df.rename(columns={"x": "x_raw", "y": "y_raw",
                          "z": "z_raw"}, inplace=True)
                # add smoothened positions
                df['x'] = trajsmo[:, 0]
                df['y'] = trajsmo[:, 1]
                df['z'] = trajsmo[:, 2]

                ############## Bayesian persistence and activity #################
                persistence, activity, speed = get_pa(df, time_interval)
                df['persistence'] = np.concatenate(
                    [np.array([np.nan]*2), persistence])
                df['activity'] = np.concatenate(
                    [np.array([np.nan]*2), activity])
                df['speed'] = np.concatenate([np.array([np.nan]), speed])

                # add directional autocorrelations
                df['directional_autocorrelation'] = DA_3D(
                    df[['x', 'y', 'z']].to_numpy())

                # get the trajectory and the previous trajectory for each frame and
                # save as an individual dataframe for each cell and frame
                for v, row in df.iterrows():
                    if v == 0:
                        row['Prev_Trajectory_X'] = np.nan
                        row['Prev_Trajectory_Y'] = np.nan
                        row['Prev_Trajectory_Z'] = np.nan
                        row['Trajectory_X'] = traj[v, 0]
                        row['Trajectory_Y'] = traj[v, 1]
                        row['Trajectory_Z'] = traj[v, 2]
                        row['Turn_Angle'] = np.nan
                        pd.DataFrame(row.to_dict(), index=[0]).to_csv(
                            csvdir.joinpath(row.cell + '_cell_info.csv'))

                    if v > 0:
                        row['Prev_Trajectory_X'] = traj[v-1, 0]
                        row['Prev_Trajectory_Y'] = traj[v-1, 1]
                        row['Prev_Trajectory_Z'] = traj[v-1, 2]
                        row['Trajectory_X'] = traj[v, 0]
                        row['Trajectory_Y'] = traj[v, 1]
                        row['Trajectory_Z'] = traj[v, 2]
                        if all(traj[v-1, :] == traj[v, :]):
                            row['Turn_Angle'] = 0
                        else:
                            row['Turn_Angle'] = angle3D(
                                traj[v-1, 0], traj[v-1, 1], traj[v-1, 2], traj[v, 0], traj[v, 1], traj[v, 2])
                        pd.DataFrame(row.to_dict(), index=[0]).to_csv(
                            csvdir.joinpath(row.cell + '_cell_info.csv'))

        print(f'Finished tracking cell {i}')


############ FIND WIDTH ROTATIONS THAT DEPEND ON PREVIOUS FRAMES TO LIMIT ROTATION FLIPPING ################
def get_normal_rotations(
        imdir: Path,  # where to find the segmented images and position information
        config: Config,
):
    #save some variables from the config
    savedir = config.common.savedir  # where to save the normal rotations
    align_method = config.common.align_method # how to align the cells based on shparam_mod.find_normal_width_peaks function
    normal_method = config.common.normal_method # what method to use to find the normal rotation,

    meshdir = imdir.joinpath('meshes')
    csvdir = imdir.joinpath('smooth_traj')
    datadir = savedir.joinpath('shape_data')
    if not datadir.exists():
        datadir.mkdir(parents=True)

    # get the list of unique cells that we have trajectory info for
    #first get list of unique cells in the image folder for that experiment
    imlist = list(set([o.name.split('_frame')[0] for o in meshdir.glob('*')]))
    #next get unique cells in the whole dataset
    csvlist = list(set([o.name.split('_frame')[0] for o in csvdir.glob('*')]))
    #combine and just get cells from that experiment that I have trajectory info for
    uniquelist = [x for x in imlist if x in csvlist]
    
    # loop through the unique cells and open the segmented images to rotate
    # each mesh until you find the rotation angle for the widest axis perpendicular
    # to the trajectory
    # trajinfolist = [x.name for x in csvdir.glob('*cell_info.csv')]
    # segimlist = [x.name for x in procimdir.glob('*_segmented*')]
    allresults = []
    for u in uniquelist:
        # get list of all frames I have trajectory info on with this cell
        cellframelist = [c.name.split('_cell_info')[0] for c in csvdir.glob('*'+u+'_frame*')]
        
        ### calculate normal rotation if measuring by width perpendivular to trajectory
        if normal_method == 'width':
            # get all segmented images of this cell that I have trajectory info on
            # cellseglist = [j for j in segimlist if j.split('_segmented')[0] in cellframelist]
            results = []
            pool = multiprocessing.Pool(processes=60)
            for y in cellframelist:
                # get path to segmented image
                impath = meshdir.joinpath(y+'_cell_mesh.vtp')
                # put in the pool
                result = pool.apply_async(shparam_mod.find_normal_width_peaks, args=(
                    impath,
                    csvdir,
                    align_method,
                ))
                results.append(result)
            pool.close()
            pool.join()
    
            # get results
            results = [r.get() for r in results]
            results.sort(key=lambda x: float(
                re.findall('(?<=frame_)\d*', x[0])[0]))
            tempframe = pd.DataFrame(results, columns=['cell', 'Width_Peaks'])
            tempframe['frame'] = [
                float(re.findall('(?<=frame_)\d*', x[0])[0]) for x in results]
    
            tempframe, runs = get_consecutive_timepoints(tempframe, 'frame', 1)
    
            # find the minima in each frame that are closest to the minimum chosen in the last frame
            # aka the one that results in the least amount of consecutive rotation
            fullminlist = []
            for xx in runs:
                runframe = tempframe.iloc[xx]
                wplist = runframe.Width_Peaks.to_list()
                seeds = []
                allallmins = []
                # for all the starting peaks find the least different rotations through time
                for s in wplist[0]:
                    allmins = [s]
                    for wp in wplist[1:]:
                        if bool(len(wp) == 0):
                            allmins.append(allmins[-1])
                        else:
                            allmins.append(wp[np.argmin(abs(wp-(allmins[-1])))])
                    allallmins.append(allmins)
                    seeds.append(np.sum(abs(np.diff(allmins))))
                # add rotations of current run to the list
                fullminlist.extend(allallmins[np.argmin(seeds)])
    
            # add all mins to tempframe
            tempframe['Closest_minimums'] = fullminlist
            #add tempframe top the list of all tempframes
            allresults.append(tempframe)
            
        elif normal_method == 'planar':
            #read all the smoothened trajectory info about this cell into a dataframe
            cellinfo = [pd.read_csv(csvdir.joinpath(c+'_cell_info.csv'), index_col = 0) for c in cellframelist]
            infodf = pd.concat(cellinfo, ignore_index = True)
            
            #get the consecutive trajectory info and loop through those runs
            infodf, runs = get_consecutive_timepoints(infodf, 'frame',1)
            for r in runs:
                chunk = infodf.iloc[r]
                #get all the euler angles to align these to x axis
                trajchunk = chunk[['Trajectory_X','Trajectory_Y','Trajectory_Z']].values
                eulers = np.apply_along_axis(align_vec_to_xaxis_euler, 1, trajchunk)
                #apply rotations to the NEXT trajectory and get the rotation around the x-axis
                next_traj_rotated = np.zeros((len(chunk)-1,3))
                for e in range(len(chunk)-1):
                    #apply euler to the next trajectory
                    ro = R.from_euler('xyz', eulers[e], degrees = True)
                    next_traj_rotated[e] = ro.apply(trajchunk[e+1])
                #get the actual rotation angles around the x-axis needed to align
                #the next trajectory with the y-axis
                theta = np.arctan2(next_traj_rotated[:,2],next_traj_rotated[:,1])
                #ensure negative y direction
                theta += np.pi
                #convert to degrees
                deg = -np.rad2deg(theta)
                
                ### assemble dataframe to match the 'width' normal_method
                tempframe = chunk[['cell','frame']].copy()
                tempframe['Width_Peaks'] = np.nan
                tempframe['Closest_minimums'] = np.append(deg, np.nan)

                allresults.append(tempframe)
                
        ### rotate to somewhat preserve original frame
        elif normal_method == 'original':
            for c in cellframelist:
                ## open the current mesh
                mesh = read_polydata(meshdir.joinpath(c+'_cell_mesh.vtp'))
                ## rotate to align to long axis
                eulers, ro = shparam_mod.get_long_axis_eulers_mesh(mesh, True)
                    
                #apply euler to the original negative y direction
                next_traj_rotated = ro.apply([0,-1,0])
                #get the actual rotation angles around the x-axis needed to align
                #the next trajectory with the y-axis
                theta = np.arctan2(next_traj_rotated[2],next_traj_rotated[1])
                #apply these rotations and if y is positive, flip it
                theta += np.pi
                #convert to degrees
                deg = -np.rad2deg(theta)
                    
                ### assemble dataframe to match the 'width' normal_method
                tempframe = pd.DataFrame({
                    'cell': c,
                    'frame': int(c.split('_')[-1]),
                    'Width_Peaks': np.nan,
                    'Closest_minimums': deg,
                    }, index = [0])
    
                allresults.append(tempframe)

        print('Finished ' + u)

    # save the shape metrics dataframe
    bigdf = pd.concat(allresults, ignore_index = True)
    bigdf.to_csv(datadir.joinpath(f'Closest_Width_Peaks_{mindir.name}.csv'))


def seg_to_mesh(
    imdir, # where to find the segmented images 
    config: Config,
    microscope: str,  # what microscope the data is from to determine which config parameters to use
    ):
    #save some variables from the config
    if microscope == 'confocal':
        mcon = config.confocal
    elif microscope == 'lls':
        mcon = config.lls
    savedir = config.common.savedir  # where to save the meshes etc.
    xyres = mcon.xyres  # xy resolution
    zstep = mcon.zstep  # z resolution
    align_method = config.common.align_method  # how to align the cells
    l_order = config.common.l_order  # L order for SH coefficients

    ## get a few variables from the config file
    norm_rot = config.common.normal_method

    # make dirs if it doesn't exist
    datadir = savedir.joinpath('shape_data')
    csvdir = imdir.joinpath('smooth_traj')
    meshdir = imdir.joinpath('meshes')


    widthpeaks = pd.read_csv(datadir.joinpath(
        f'Closest_Width_Peaks_{imdir.name}.csv'), index_col=0)

    # get all segmented images that were analyzed
    datalist = [x.name.split('_cell_info.csv')[0] for x in csvdir.glob('*_cell_info.csv')]
    meshlist = [x for x in meshdir.glob('*_cell_mesh.vtp') if x.name.split('_cell_mesh.vtp')[0] in datalist]

    mapargs = []
    for i in meshlist:
        # assign the normal rotation value for that particular cell
        norm_rot = float(widthpeaks[widthpeaks.cell == i.name.split('_cell_mesh')[0]]['Closest_minimums'].values)#[0])
        if np.isnan(norm_rot):
            continue

        # append unique args to list
        mapargs.append((
            i,
            savedir,
            xyres,
            zstep,
            norm_rot,
            l_order,
            align_method,
        ))

    # parallel processing for all segmented images
    with multiprocessing.Pool(processes=60) as pool:
        results = list(tqdm(pool.imap(
            shparam_mod.shape_info_imap, mapargs), total=len(mapargs)))

    # get results
    dflist = [r for r in results]

    # save the shape metrics dataframe
    bigdf = pd.DataFrame(dflist)
    bigdf.to_csv(datadir.joinpath(
        f'Shape_Metrics_{imdir.name}.csv'))



################ SEGMENT AND TRACK CELLS FROM MANUALLY CROPPED LLS MOVIES #############
def segment_and_crop_LLS_manual(
        mindir,  # base directory with save folder and info folder
        raw_dir,  # directory where all of the cropped LLS images live
        cellstr,  # the name of the unique cell being cropped and segmented across multiple videos\
        decon=True,  # are these images deconvolved?
        orig_size=False,  # should we save the images at their original size?
        xy_buffer=25,  # crop buffer in x-y
        z_buffer=25,  # crrop buffer in z
        hilo=True,  # whether or not to do multiple thresholds for segmenting secondary signals
):

    savedir = mindir.joinpath('processed_images')
    posdir = mindir.joinpath('position_info')
    # make the savedir if it doesn't exist
    if not savedir.exists():
        savedir.mkdir(parents=True)
    if not posdir.exists():
        posdir.mkdir(parents=True)

    # get all of the images from a particular cell I was following
    curimlist = [x.name for x in raw_dir.glob(f'*{cellstr}*')]
    # find the total number of cells I cropped while following the cell of interest
    cellnums = list(set([re.findall(r'Subset-(\d+)', x)[0]
                    for x in curimlist]))
    cellnums.sort()
    for s in cellnums:
        # list to put all dataframes from all subsets
        dflist = []
        # get all the images of a given cell
        curcell = [x for x in curimlist if f'Subset-{s}' in x]
        # sort the current cell to be in chronological order
        curcell.sort(key=lambda x: float(re.findall(r'(\d+)-Subset', x)[0]))
        for n, c in enumerate(curcell):
            celldir = raw_dir.joinpath(c)
            # open the image
            czi = CziFile(celldir)
            imdata = czi.read_image()
            # absolute timepoint of first image
            if n == 0:
                timezero = metadata_funcs.adjustedstarttime(czi)
            # get time interval and number of frames and start time
            ti = metadata_funcs.gettimeinterval(czi)
            fn = metadata_funcs.framesinsubset(czi)
            ast = metadata_funcs.adjustedstarttime(czi)

            # get all the times at the current frame since the cell was initially observed
            times = [int(ast - timezero + (f*ti)) for f in range(fn)]

            # segment the cells and return the position info
            # get the file name
            image_name = celldir.name.split('.')[0]

            # choose structure name based on file name
            if 'actin' in image_name:
                struct = 'actin'
            elif ('Hoechst' in image_name) or ('DNA' in image_name):
                struct = 'nucleus'
            elif 'mysoin' in image_name:
                struct = 'myosin'
            else:
                struct = ''

            # get the pixel size from the metadata
            scale = metadata_funcs.getscale(czi)
            xyres = scale[0]
            zstep = scale[-1]
            # set image shape
            imshape = czi.shape
            # get the actual frame numbers from the original video
            first, last = metadata_funcs.frame_range_in_subset(czi)
            framelist = list(range(first-1, last))

            # get the crops for each frame based on coarse thresholding
            celldf = segment_LLS.getbb_movie(imdata[:, 1, :, :, :])
            celldf['actual_frame'] = framelist
            celldf['frame'] = list(range(len(celldf)))
            # add actual times that were previously calculated from metadata
            celldf['time'] = times
            # drop any na frames that weren't able to find bounding boxes
            celldf = celldf.dropna().reset_index(drop=True)

            # use multiprocessing to perform segmentation and x,y,z determination
            pool = multiprocessing.Pool(processes=60)
            results = []
            for t, row in celldf.iterrows():

                # segment the cropped images
                result = pool.apply_async(segment_LLS.LLSseg, args=(
                    savedir,
                    image_name,
                    row.to_dict(),
                    imdata[int(row.frame), :, :, :, :],
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

            # print progress
            print('Finished segmenting cropped images of '+c)

            # get results
            results = [r.get() for r in results]
            # deal with any frames that messed up
            bef = len(results)
            results = [l for l in results if l is not None]
            af = len(results)
            if af < bef:
                print(str(bef-af)+' frames dropped from ' + image_name)
            if af > 0:
                # aggregate the dataframe
                df = pd.DataFrame()
                for d in results:
                    df = df.append(pd.DataFrame(
                        d, columns=d.keys(), index=[0]))
                df = df.sort_values(by='frame').reset_index(drop=True)
                dflist.append(df)
            else:
                print(image_name + ' did not have enough segmented frames in movie')
        if len(dflist) > 0:
            # combine all of the subset dataframes and save
            fulldf = pd.concat(dflist).reset_index(drop=True)
            fulldf['CellID'] = [cellstr+f'_{s}']*len(fulldf)
            fulldf.to_csv(posdir.joinpath(cellstr + f'_{s}_cellpos.csv'))
        else:
            print('No images were recovered of cell ' +
                  re.split('-\d*-Subset', curcell[0])[0] + '-' + s)


# def get_pilr_regions(
#         mindir,
# ):

#     # make dirs if it doesn't exist
#     datadir = mindir.joinpath('shape_data')
#     pilrf = mindir.joinpath('PILRs')

#     # get a list of all of the PILR images
#     pilrlist = [x for x in pilrf.glob('*_PILR*')]
#     with multiprocessing.Pool(processes=60) as pool:
#         results = pool.map(read_pilr_regions, pilrlist)

#     pilrframe = pd.DataFrame(results).reset_index(drop=True)
#     pilrframe.to_csv(datadir.joinpath('PILR_regions.csv'))
