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
from .segment_cells2short import confocal_segmentation_wrapper, confocal_image_info_wrapper
from . import shparam_mod, metadata_funcs, segment_LLS
from .track_functions import segment_caax_tracks_confocal_40x_fromsingle
# from .PILRagg import read_pilr_regions
from .utils import get_consecutive_timepoints, smooth_trajectory_wrapper, align_vec_to_xaxis_euler
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
                    config.im_params.xyres,
                    config.im_params.zstep,
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
    xyres = config.im_params.xyres  # xy resolution of images
    zstep = config.im_params.zstep  # z resolution of images
    xy_buffer = config.im_params.xy_buffer  # amount to buffer cropped images in xy
    z_buffer = config.im_params.z_buffer  # amount to buffer cropped images in z
    stackshape = config.im_params.stackshape  # shape of one z stack in pixels (z,y,x) format
    whatseg = config.im_params.whatseg  # what segmentation function to use for which cells


    mapargs = []
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
                    ## run the segmentation function
                    mapargs.append((
                        tdir,
                        stackshape,
                        row,
                        procimdir,
                        xyres,
                        zstep,
                        croparr,
                        whatseg,
                    ))

    ### get the normal alignment vectors of all cells at once
    with multiprocessing.Pool(processes=60) as pool:
        results = list(tqdm(pool.imap(
            confocal_segmentation_wrapper, mapargs), total=len(mapargs)))

    # make sure there's no None results from failed segmentations
    results = [x for x in results if x != None]
    segdf = pd.DataFrame(results)
    for cellid, celldf in segdf.groupby('CellID'):
        celldf = celldf.sort_values('frame').reset_index(drop = True)
        # save
        celldf.to_csv(posdir.joinpath(
            cellid+'_cellpos.csv'))



# GET TRAJECTORIES FROM POSITION INFO
def get_smooth_trajectories(
        imdir: Path,  # where to find the segmented images and position information
        config: Config,
        ):
    #save some variables from the config
    time_interval = config.im_params.time_interval  # time interval between frames of movies
    smooth_factor = config.common.smooth_factor  # "s" parameter in the interpolate.splprep function

    # define directory stuff
    csvdir = imdir.joinpath('smooth_traj')
    posdir = imdir.joinpath('position_info')
    if not csvdir.exists():
        csvdir.mkdir(parents=True, exist_ok=True)

    # combine all of the cell csvs into one dataframe
    csvlist = list(posdir.glob('*.csv'))
    celllist = [pd.read_csv(c, index_col=0) for c in csvlist]
    cellinfo = pd.concat(celllist).reset_index(drop=True)

    # add time to the confocal data
    if 'time' not in cellinfo.columns.to_list():
        cellinfo['time'] = cellinfo['frame'].values * time_interval

    mapargs = []
    for i, celldf in cellinfo.groupby('CellID'):
        # first get dataframe in time order and consecution timepoints
        celldf, runs = get_consecutive_timepoints(
            celldf[~celldf.x_raw.isna()], 'time', time_interval)

        for r in runs:
            if len(r) > 2:
                df = celldf.iloc[r].reset_index(drop=True)
                mapargs.append([df, smooth_factor, time_interval])

    ### get the normal alignment vectors of all cells at once
    with multiprocessing.Pool(processes=60) as pool:
        results = list(tqdm(pool.imap(
            smooth_trajectory_wrapper, mapargs), total=len(mapargs)))

    allsmoothdf = pd.concat(results, ignore_index = True)
    allsmoothdf.to_csv(csvdir.joinpath(f'Smooth_Trajectories_{imdir.name}.csv'))


############ FIND WIDTH ROTATIONS THAT DEPEND ON PREVIOUS FRAMES TO LIMIT ROTATION FLIPPING ################
## some column lists for referece
major_ax = ['Cell_Major_Axis_Vec_X','Cell_Major_Axis_Vec_Y','Cell_Major_Axis_Vec_Z']
median_ax = ['Cell_Median_Axis_Vec_X','Cell_Median_Axis_Vec_Y','Cell_Median_Axis_Vec_Z']
traj_cols = ['Trajectory_Vec_X','Trajectory_Vec_Y','Trajectory_Vec_Z']

def get_alignment_angles(
        imdir: Path,  # where to find the segmented images and position information
        config: Config,
):
    #save some variables from the config
    savedir = config.common.savedir  # where to save the normal rotations
    align_method = config.common.align_method # how to align the cells based on shparam_mod.find_normal_width_peaks function
    normal_method = config.common.normal_method # what method to use to find the normal rotation,
    xyres = config.im_params.xyres
    zstep = config.im_params.zstep
    time_interval = config.im_params.time_interval

    meshdir = imdir.joinpath('meshes')
    posdir = imdir.joinpath('position_info')
    trajdir = imdir.joinpath('smooth_traj')
    datadir = savedir.joinpath('shape_data')
    if not datadir.exists():
        datadir.mkdir(parents=True, exist_ok=True)

    ## open all of the original cell positions for principal axes
    poslist = [pd.read_csv(c, index_col = 0) for c in posdir.glob('*.csv')]
    posdf = pd.concat(poslist, ignore_index = True)
    ## open smooth trajectories
    smoothdf = pd.read_csv(trajdir.joinpath(f'Smooth_Trajectories_{imdir.name}.csv'), index_col = 0)
    ## merge the two
    df = smoothdf.merge(posdf, how = 'left', on = 'cell')

    ## loop through the unique cells and measure/save the rotation angles according
    ## to the specified alignment method
    allresults = []
    mapargs = [] # specifically for trajectory_shape alignment to process all at once
    for cellid, celldf in df.groupby('CellID'):
        ### get continuous runs of dataframe
        celldf, runs = get_consecutive_timepoints(celldf, 'time', time_interval)
        celllist = celldf.cell.tolist()
        
        if normal_method == 'width':
            ### get the Euler angles for alignment from previously measured
            ### principal axes
            if align_method == 'long_axis':
                ### ensure all major axes are aligned similarly
                majors = celldf[major_ax].values
                # dot products between consecutive vectors
                dots = np.sum(majors[:-1] * majors[1:], axis=1)
                # get signs
                dot_signs = np.where(dots < 0, -1, 1)
                # add first position and get cum prod
                s = np.concatenate(([1], np.cumprod(dot_signs)))
                # correct signs of the actual vectors and store in celldf
                majors_aligned = majors * s[:, np.newaxis]
                celldf[major_ax] = majors_aligned
                ## get alignment eulers
                eulers = []
                for i, row in celldf.iterrows():
                    ax_align, _ = R.align_vectors(
                        [[1,0,0],[0,-1,0]],
                        [row[major_ax].values, row[median_ax].values]
                        )
                    xyz = ax_align.as_euler('xyz', degrees = True)
                    eulers.append(xyz)
                eulers = np.stack(eulers).T
                
                #build dataframe
                tempframe = pd.DataFrame({
                    'cell': celllist,
                    'Euler_Angles_X': eulers[0],
                    'Euler_Angles_Y': eulers[1],
                    'Euler_Angles_Z': eulers[2],
                    })
            
                allresults.append(tempframe)   
                
            ### calculate normal rotation if measuring by width perpendivular to trajectory
            elif align_method == 'trajectory':
                ## package arguments
                for cellstr in celllist:
                    mesh_path = meshdir.joinpath(cellstr+'_cell_mesh.vtp')
                    vec = celldf[celldf.cell == cellstr][traj_cols].values[0]
                    mapargs.append([
                        mesh_path,
                        vec,
                        ])
                
        elif normal_method == 'planar':
            ### for consecutive frames, align cells according to their current and
            ### next trajectory vectors
            for r in runs:
                chunk = celldf.iloc[r]
                #get the trajectory vectors
                trajchunk = chunk[traj_cols].values
                nexttrajchunk = chunk[['Next_'+x for x in traj_cols]].values

                eulerlist = []
                for v1, v2 in zip(trajchunk, nexttrajchunk):
                    #pass up any nan rows
                    if any(np.isnan((*v1,*v2))):
                        eulerlist.append(np.repeat(np.nan,3))
                    else:
                        #subtract v1 from v2
                        v2 = v2 - np.dot(v2, v1) * v1
                        v2 /= np.linalg.norm(v2)
                        ax_align, _ = R.align_vectors(
                                                np.array([[1,0,0],[0,-1,0]]),
                                                np.array([v1, v2])
                                                )
                        eulerlist.append(ax_align.as_euler('xyz', degrees = True))
            
                ### assemble dataframe to match the 'width' normal_method
                tempframe = chunk[['cell']].copy().reset_index(drop = True)
                #also add euler angles
                eulers = np.array(eulerlist)
                eulerframe = pd.DataFrame(eulers, columns = ['Euler_Angles_X','Euler_Angles_Y','Euler_Angles_Z'])
                tempframe = pd.concat((tempframe, eulerframe), axis = 1)
                allresults.append(tempframe)
        
        
    if (normal_method == 'width') and (align_method == 'trajectory'):
        ### get the normal alignment vectors of all cells at once
        with multiprocessing.Pool(processes=60) as pool:
            results = list(tqdm(pool.imap(
                shparam_mod.get_orthogonal_mass_vector_imap, mapargs), total=len(mapargs)))
        
        ## get eulers to align vecs
        original_vec_array = np.array([x[1] for x in mapargs])
        ortho_vec_array = np.array(results)
        eulerlist = []
        for v1, v2 in zip(original_vec_array, ortho_vec_array):
            ax_align, _ = R.align_vectors(
                            np.array([[1,0,0],[0,-1,0]]),
                            np.array([v1, v2])
                            )
            eulerlist.append(ax_align.as_euler('xyz', degrees = True))
            
        ### put all the alignment angles together in a dataframe
        eulers = np.stack(eulerlist).T
        allcelllist = [m[0].stem.split('_cell_mesh')[0] for m in mapargs]
        bigdf = pd.DataFrame({
            'cell': allcelllist,
            'Euler_Angles_X': eulers[0],
            'Euler_Angles_Y': eulers[1],
            'Euler_Angles_Z': eulers[2],
            })
        # save the shape metrics dataframe
        bigdf.to_csv(datadir.joinpath(f'Alignment_Angles_{imdir.name}.csv'))

    else:
        # save the shape metrics dataframe
        bigdf = pd.concat(allresults, ignore_index = True)
        bigdf.to_csv(datadir.joinpath(f'Alignment_Angles_{imdir.name}.csv'))




def extract_shape_metrics(
    imdir, # where to find the segmented images 
    config: Config,
    ):
    #save some variables from the config
    savedir = config.common.savedir  # where to save the meshes etc.
    l_order = config.common.l_order  # L order for SH coefficients

    # make dirs if it doesn't exist
    datadir = savedir.joinpath('shape_data')
    meshdir = imdir.joinpath('meshes')
    posdir = imdir.joinpath('position_info')
    trajdir = imdir.joinpath('smooth_traj')


    ### open all relevant data
    angledf = pd.read_csv(datadir.joinpath(
        f'Alignment_Angles_{imdir.name}.csv'), index_col=0)
    poslist = [pd.read_csv(c, index_col = 0) for c in posdir.glob('*.csv')]
    posdf = pd.concat(poslist, ignore_index = True)
    posonly = [x for x in posdf.columns if x not in angledf.columns]
    #merge
    angledf = angledf.merge(posdf[['cell']+posonly], how = 'left', on = 'cell')
    smoothdf = pd.read_csv(trajdir.joinpath(
            f'Smooth_Trajectories_{imdir.name}.csv'), index_col = 0)
    smoothonly = [x for x in smoothdf.columns if x not in angledf.columns]
    #merge
    angledf = angledf.merge(smoothdf[['cell']+smoothonly], how = 'left', on = 'cell')


    mapargs = []
    for i, row in angledf.iterrows():
        #move on if there's no rotation
        if np.isnan(row.Euler_Angles_X):
            continue

        # append unique args to list
        mapargs.append((
            row,
            meshdir,
            l_order,
            ))

    # parallel processing for all segmented images
    with multiprocessing.Pool(processes=60) as pool:
        results = list(tqdm(pool.imap(
            shparam_mod.shape_info_imap, mapargs), total=len(mapargs)))

    # save the shape metrics dataframe
    bigdf = pd.DataFrame(results)
    bigdf.to_csv(datadir.joinpath(
        f'Shape_Metrics_{imdir.name}.csv'))



################ SEGMENT AND TRACK CELLS FROM MANUALLY CROPPED LLS MOVIES #############
def segment_and_crop_LLS_manual(
        cellstr,  # the name of the unique cell being cropped and segmented across multiple videos\
        config: Config,
        main_cell_only: bool = True,  # whether to only segment the main cell in the cropped movies or also segment secondary cells that are in the crop
        ):

    decon=config.im_params.decon  # are these images deconvolved?
    xyres=config.im_params.xyres
    zstep=config.im_params.zstep
    orig_size=config.im_params.orig_size  # should we save the images at their original size?
    xy_buffer=config.im_params.xy_buffer  # crop buffer in x-y
    z_buffer=config.im_params.z_buffer  # crrop buffer in z
    hilo=config.im_params.hilo  # whether or not to do multiple thresholds for segmenting secondary signals

    raw_dir = config.experiment.lls.serverdir
    imdir = config.experiment.lls.localdir
    procimdir = imdir.joinpath('processed_images')
    posdir = imdir.joinpath('position_info')
    meshdir = imdir.joinpath('meshes')
    # make the dirs if they don't exist
    if not procimdir.exists():
        procimdir.mkdir(parents=True)
    if not posdir.exists():
        posdir.mkdir(parents=True)
    if not meshdir.exists():
        meshdir.mkdir(parents=True)

    # get all of the images from a particular cell I was following
    curimlist = [x.name for x in raw_dir.glob(f'*{cellstr}*')]
    # find the total number of cells I cropped while following the cell of interest
    if main_cell_only:
        cellnums = ['01']
    else:
        cellnums = list(set([re.findall(r'Subset-(\d+)', x)[0]
                        for x in curimlist]))
        cellnums.sort()
    for s in cellnums:
        # list to put all dataframes from all subsets
        wholecelldflist = []
        # get all the images of a given cell
        curcell = [x for x in curimlist if f'Subset-{s}' in x]
        # sort the current cell to be in chronological order
        curcell.sort(key=lambda x: float(re.findall(r'(\d+)-Subset', x)[0]))
        for n, c in enumerate(curcell):
            celldir = raw_dir.joinpath(c)
            # open the image
            czi = CziFile(celldir)
            imdata, _ = czi.read_image()
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

            # set whole image shape
            imshape = czi.size
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
                    procimdir,
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
                df = pd.DataFrame([r for r in results])
                wholecelldflist.append(df)
            else:
                print(image_name + ' did not have enough segmented frames in movie')
        if len(wholecelldflist) > 0:
            # combine all of the subset dataframes and save
            fulldf = pd.concat(wholecelldflist).reset_index(drop=True)
            fulldf['CellID'] = [cellstr+f'_{s}']*len(fulldf)
            fulldf.to_csv(posdir.joinpath(cellstr + f'_{s}_cellpos.csv'))
        else:
            print('No images were recovered of cell ' +
                  re.split(r'-\d*-Subset', curcell[0])[0] + '-' + s)


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




def construct_full_lls_movie(
        cellname: str,
        config: Config,
        xy_buffer: int = 7,
        z_buffer: int = 7,
        ):
    ##### directories from config
    serverdir = config.experiment.lls.serverdir
    savedir = config.experiment.lls.localdir

    ### build list of images to open
    namesplit = cellname.split('_')
    image_prefix = '_'.join(namesplit[:-1])
    cellnum = namesplit[-1]
    ## get list of unique movies and sort by movie number
    lst = set([x for x in serverdir.glob(f'{image_prefix}*') if f'Subset-{cellnum}' in x.name])
    sortedlst = sorted(lst, key=lambda x: int(re.findall(r'\d*(?=-Subset)', x.name)[0]))

    ####### loop through all of the movies that captured the cell of interest
    ####### MODIFIED FROM segment_LLS.getbb_movie
    centroids = []
    imagelist = []
    framelist = []
    for l in sortedlst:
        ## read the image
        big = CziFile(l)
        bigim, _ = big.read_image()

        ### get the bounding box info and the rescaled membrane channel
        cropdf, rescaled = segment_LLS.getbb_movie(bigim[:,1,:,:,:],return_rescaled = True)
        ### also rescale secondary channel
        with multiprocessing.Pool(processes=60) as pool: 
            rescaledother = pool.map(segment_LLS.quarter_scale, [i for i in bigim[:,0,:,:,:]])
        ### stack the rescaled channels back together
        rescaledall = np.stack((rescaled,rescaledother))
        ### get the shape of the rescaled to use later to leave out objects
        shape = rescaledall.shape

        

        ###aggregate cropping info from this particular image
        #first adjust the coordinates of the crop from original image to the rescaled image
        quarter_column_list = ['x','y','z','x_min','y_min','z_min','x_max','y_max','z_max']
        for qcl in quarter_column_list:
            cropdf[qcl] = (cropdf[qcl]/4)

        ### get the aggregate mins and maxes
        cropdf['x_min'] = int(max(0, cropdf['x_min'].min()-xy_buffer))
        cropdf['y_min'] = int(max(0, cropdf['y_min'].min()-xy_buffer))
        cropdf['z_min'] = int(max(0, cropdf['z_min'].min()-z_buffer))
        cropdf['z_max'] = int(min(cropdf['z_max'].max()+z_buffer, shape[-3]))
        cropdf['y_max'] = int(min(cropdf['y_max'].max()+xy_buffer, shape[-2])+1)
        cropdf['x_max'] = int(min(cropdf['x_max'].max()+xy_buffer, shape[-1])+1)
        mincrop = np.min(cropdf[['z_min','y_min','x_min']].values, axis = 0)
        maxcrop = np.max(cropdf[['z_max','y_max','x_max']].values, axis = 0)

        ## append the cropped image and the centroids of the cropped cell
        imagelist.append(rescaledall[:,:,
                                mincrop[0]:maxcrop[0],
                              mincrop[1]:maxcrop[1],
                              mincrop[2]:maxcrop[2]])
        centroids.append(cropdf[['z','y','x']].values-np.array([mincrop[0],mincrop[1],mincrop[2]]))
        #append all of the frame numbers and names that were used to compile this video
        framelist.extend([l.name+'_frame_'+'{:03d}'.format(r) for r in range(len(rescaled))])
        print(f'finished cropping move {l.name}')
    ###pull all the images into one movie with minimal dimensions
    #first make sure there's no nan, if there is replace it with the centroid before it
    #or the index after it if the nan is at 0
    for i, ce in enumerate(centroids):
        if np.any(np.isnan(ce)):
            naninds = np.unique(np.where(np.isnan(ce))[0])
            nanreplace = naninds-1
            if bool(0 in naninds):
                nanreplace[nanreplace==-1] = 1
            centroids[i][naninds] = centroids[i][nanreplace]
    #then get the relative positions of the first frames
    newc = []
    for u, c in enumerate(centroids):
        if u!=0:
            newc.append(centroids[u-1][-1]-c[0])
        else:
            newc.append(np.zeros(3))
    zeropos = np.stack(newc)
    zeropos = np.cumsum(zeropos,axis=0)
    zeropos = np.round(zeropos + abs(zeropos.min())+0.0001)
    shapes = [x.shape for x in imagelist]
    totalshape = np.round(np.max(zeropos + np.array(shapes)[:,-3:], axis = 0)-np.min(zeropos,axis=0)+0.001)
    finalimage = np.zeros((np.concatenate([[sum([x[1]for x in shapes])],[2],totalshape]).astype(int)))
    timecount = 0
    shifts = zeropos-np.min(zeropos,axis=0)
    for r in range(len(shapes)):
        s = shapes[r]
        z = shifts[r].astype(int)
        finalimage[timecount:timecount+s[1],
                   :,
                   z[-3]:z[-3]+s[-3],
                   z[-2]:z[-2]+s[-2],
                   z[-1]:z[-1]+s[-1],
                  ] = np.swapaxes(imagelist[r],0,1)
        timecount = timecount+s[1]

    indivdir = savedir / 'singlecells' / cellname
    if not indivdir.exists():
        indivdir.mkdir(parents=True)
    #save full image
    tifffile.imwrite(indivdir / f'{cellname}_full_movie.ome.tiff',
                     finalimage.astype('uint16'),
                     metadata={'axes': 'TCZYX'})
    #save maximum intensity projection
    maxproj = np.max(finalimage, axis = 2)
    tifffile.imwrite(indivdir / f'{cellname}_full_movie_maxproj.ome.tiff',
                     maxproj.astype('uint16'),
                     metadata={'axes': 'TCYX'})
    #save which frames from which videos were actually used
    pd.Series(framelist).to_csv(indivdir / f'{cellname}_framelist.csv')