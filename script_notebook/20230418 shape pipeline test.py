# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:31:42 2023

@author: Aaron
"""
import numpy as np
import os
import pandas as pd
import time
import math

# package for io 
from aicsimageio.writers import OmeTiffWriter
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader

from CustomFunctions.track_functions import segment_caax_tracks_iSIM_visiview_halfsize
import itertools
from skimage import measure as skmeasure
import datetime

# load in some stuff to speed up processing
# (following https://sedeh.github.io/python-pandas-multiprocessing-workaround.html)
import multiprocessing



############### SEGMENT AND SAVE CELLS ################################

############## alignment by chemical gradient ###################

from scipy.spatial import KDTree, distance
from itertools import groupby
from operator import itemgetter
from scipy import interpolate

from aicsimageio.readers.ome_tiff_reader import OmeTiffReader

#import the cell segmentation and rotation function
# import sys
# # insert at 1, 0 is the script path (or '' in REPL)
# import_path = 'C:/Users/Aaron/Documents/PythonScripts/CustomFunctions'
# sys.path.insert(1, import_path)
from CustomFunctions.segment_cells_iSIM import segment_cells_rotafter_VV

from CustomFunctions.persistance_activity import get_pa


# path to folder 
fol = 'C:/Users/Aaron/Data/Processed/Galvanotaxis/20230329/'
folder_fl = fol + 'Videos/'
savedir = fol + 'Cropped_Cells/'
rawdir = 'C:/Users/Aaron/Data/Raw/Galvanotaxis/20230329/'


#make meshdir if it doesn't exist
meshdir = fol + 'Meshes/'
if not os.path.exists(meshdir):
    os.makedirs(meshdir)


#parameters for segmentation
xy_buffer = 25 #pixels
z_buffer = 5 #pixels
sq_size = 250 #pixels
xyres = 0.1613 #um / pixel
zstep = 0.5 # um
interval = 30

filelist_fl = [x for x in os.listdir(folder_fl) if x.endswith('.tiff')]


u =filelist_fl[0]
# for u in filelist_fl:

    ################## align trackmate data with region props data ################
    rpcsv = u.split('seg')[0] + "region_props.csv"
    rp = pd.read_csv(folder_fl + rpcsv, index_col = 0)
    tmcsv = u.split('seg')[0] + "TrackMateLog.csv"
    tm = pd.read_csv(folder_fl + tmcsv)
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
    image_name = u.split('_seg')[0]
    direct = rawdir + image_name +'/'
    currentim = [x for x in os.listdir(direct) if x.endswith('.ome.tif')][0]
    im_temp_whole = TiffReader(direct+currentim)
    
    
    

    #############find distance travelled##################
    longdistmatrix = distance.pdist(df_track[['x','y','z']])
    shortdistmatrix = distance.squareform(longdistmatrix)
    shortdistmatrix = np.array(shortdistmatrix)
    dist = []
    for i in range(len(shortdistmatrix)):
        if i == 0:
            dist.append(0)
        else:
            dist.append(shortdistmatrix[i,i-1])
    df_track = df_track.reset_index(drop = True)
    df_track['dist'] = dist
    #drop first rows that have super long distances from previous cell
    df_track.loc[df_track.groupby('cell').head(1).index,'dist'] = 0



    ############## find euclidean distance #############
    euclid = []
    for i, cell in df_track.groupby('cell'):
        FL = cell.iloc[[0,-1]]
        euc_dist = distance.pdist(FL[['x','y','z']])
        euclid.append([cell.cell.iloc[0],euc_dist[0]])
    euclid = pd.DataFrame(euclid, columns=['cell','euc_dist'])
    cellsmorethan = euclid.loc[euclid['euc_dist']>5, 'cell']
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
    df_track = df_track.loc[df_track['x_max'] != im_temp_whole.shape[-1]]
    df_track = df_track.loc[df_track['y_max'] != im_temp_whole.shape[-2]]
    df_track = df_track.loc[df_track['z_max'] != (df_track['z_range']-1)]
    #reset index after dropping all the rows
    df_track = df_track.reset_index(drop = True)


    ##########remove small things that are likely dead cells or parts of cells###########
    df_track = df_track[df_track['area'] > 15000 ]


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


    
    #######get cells with at least three consecutive frames##########
    morethanthree = []
    if df_track.empty == False:
        for i, cells in df_track.groupby('cell'):
    #         cells = cells.reset_index(drop = True)
            runs = list()
            #######https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
            for k, g in groupby(enumerate(cells['frame']), lambda ix: ix[0] - ix[1]):
                currentrun = list(map(itemgetter(1), g))
                list.append(runs, currentrun)
            whichframes = np.array(max(runs, key=len), dtype=int)
            #get indices in original dataframe of the runs of consecutive cell frames
            indices = cells.loc[cells['frame'].isin(whichframes)].index.to_list()
            if len(indices) >= 3:
                morethanthree.extend(indices)
                
        morethanthreeframes = df_track.loc[morethanthree].reset_index(drop = True)
                

        for i, cell in morethanthreeframes.groupby('cell'):

            #segment the cell channel and get centroid
            df = pd.DataFrame()
            if __name__ ==  '__main__':
                # use multiprocessing to perform segmentation and x,y,z determination
                pool = multiprocessing.Pool(processes=60)
                results = []
                for t, row in cell.iterrows():

                    xmincrop = int(max(0, row.x_min-xy_buffer))
                    ymincrop = int(max(0, row.y_min-xy_buffer))
                    zmincrop = int(max(0, row.z_min-z_buffer))


                    zmaxcrop = int(min(row.z_max+z_buffer, im_temp_whole.shape[-3])+1)
                    ymaxcrop = int(min(row.y_max+xy_buffer, im_temp_whole.shape[-2])+1)
                    xmaxcrop = int(min(row.x_max+xy_buffer, im_temp_whole.shape[-1])+1)

                    pool.apply_async(segment_cells_rotafter_VV, args = (
                        direct,
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

                pool.close()
                pool.join()

                print(f'Done segmenting {image_name} cell {cell.cell.iloc[0]}')
