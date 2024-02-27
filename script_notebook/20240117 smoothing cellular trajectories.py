# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:20:35 2024

@author: Aaron
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import math
import re

# package for io 
from aicsimageio.writers import OmeTiffWriter
from aicsimageio.readers import tiff_reader, OmeTiffReader


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
from CustomFunctions.segment_cells2short import seg_confocal_40x_memonly_fromslices

from CustomFunctions.persistance_activity import get_pa, velocity_and_distance


# path to folder(s)
folder_fl = 'D:/Aaron/Data/Galvanotaxis_Confocal_40x_30C_10s/Tracking_Images/'
filelist_fl = os.listdir(folder_fl)
savedir = 'D:/Aaron/Data/Galvanotaxis_Confocal_40x_30C_10s/Processed_Data/'
#make the savedir if it doesn't exist
if not os.path.exists(savedir):
    os.makedirs(savedir)
raw_dir = '//10.158.28.37/ExpansionHomesA/avlnas/HL60 Galv/'

#parameters for segmentation
xy_buffer = 12 #pixels
z_buffer = 8 #pixels
xyres = 0.3394 #um / pixel 
zstep = 0.7 # um
interval = 10
intthresh = 120 #for the half shrunken images, determined by manually crossreferencing
imshape = (150,1024,1024)



u = filelist_fl[0]


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
    df_track['dist'][x] = df_track.dist.mean()

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
df_track = df_track.loc[df_track['x_max'] < imshape[-1]]
df_track = df_track.loc[df_track['y_max'] < imshape[-2]]
df_track = df_track.loc[df_track['z_max'] != (df_track['z_range'])]


##########remove small things that are likely dead cells or parts of cells###########
df_track = df_track[df_track['area'] > 4000 ]


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




#find the length of cell consecutive frames
results = []
for i, cells in df_track.groupby('cell'):
    cells = cells.sort_values('frame').reset_index(drop = True)
    runs = list()
    #######https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
    for k, g in groupby(enumerate(cells['frame']), lambda ix: ix[0] - ix[1]):
        currentrun = list(map(itemgetter(1), g))
        list.append(runs, currentrun)
    maxrun = max([len(l) for l in runs])
    actualrun = max(runs, key=len, default=[])
    results.append([i, maxrun, actualrun])
#find
stdf = pd.DataFrame(results, columns = ['cell','length_of_run','actual_run']).sort_values('length_of_run', ascending=False).reset_index(drop=True)
stdf.head(30)


df_track.groupby('cell').sum()[:30]

cell = df_track[df_track.cell==13].sort_values('frame').reset_index(drop=True)




ax = plt.figure().add_subplot(projection='3d')


ax.plot(cell.x, cell.y, cell.z, label='cell 13')
ax.legend()

plt.show()

tck, b = interpolate.splprep(cell[['x','y','z']].to_numpy().T, k=5)
yderv = interpolate.splev(b,tck,der=0)
traj = np.vstack(yderv).T



ax = plt.figure().add_subplot(projection='3d')

ax.plot(traj[:,0], traj[:,1], traj[:,2], label='cell 13 smooth')
ax.legend()

plt.show()

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

                        tdir = raw_dir +u.split('_')[0]+'/' +u+'/Default/'

                        xmincrop = int(max(0, row.x_min-xy_buffer))
                        ymincrop = int(max(0, row.y_min-xy_buffer))
                        zmincrop = int(max(0, row.z_min-z_buffer))

                        zmaxcrop = int(min(row.z_max+z_buffer, imshape[-3]))
                        ymaxcrop = int(min(row.y_max+xy_buffer, imshape[-2])+1)
                        xmaxcrop = int(min(row.x_max+xy_buffer, imshape[-1])+1)

                        pool.apply_async(seg_confocal_40x_memonly_fromslices, args = (
                            tdir,
                            imshape,
                            row,
                            u,
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

                    print(f'Done segmenting {u} cell {cell.cell.iloc[0]}')

                if any([x == None for x in results]):
                    ind = results.index(None)
                    if len(results[:ind])<3:
                        pass
                    else:
                        results = results[:ind]

                #aggregate the dataframe
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


                #add new distances from cropped image
                #############find distance travelled##################
                longdistmatrix = distance.pdist(df[['x','y','z']])
                shortdistmatrix = distance.squareform(longdistmatrix)
                shortdistmatrix = np.array(shortdistmatrix)
                dist = pd.Series([], dtype = 'float64')
                for count, i in enumerate(shortdistmatrix):
                    if count == 0:
                        tmp = pd.Series([0])
                        dist = dist.append(tmp, ignore_index=True)
                    else:
                        tmp = pd.Series(shortdistmatrix[count,count-1])
                        dist = dist.append(tmp, ignore_index=True)
                df['dist'] = dist



                ############## Bayesian persistence and activity #################
                persistence, activity, speed = get_pa(df, interval)
                df['persistence'] = np.concatenate([np.array([np.nan]*2), persistence])
                df['activity'] = np.concatenate([np.array([np.nan]*2), activity])
                df['speed'] = np.concatenate([np.array([np.nan]), speed])
                df['avg_persistence'] = np.array([persistence.mean()]*(len(persistence)+2))
                df['avg_activity'] = np.array([activity.mean()]*(len(activity)+2))
                df['avg_speed'] = np.array([speed.mean()]*(len(speed)+1))

                ################ Signal velocities and distance travelled ###############
                isvs, sds, tds, eds = velocity_and_distance(df, interval, signal_vector = [-1,0,0])
                df['Instant_Signal_Velocity'] = np.concatenate([np.array([np.nan]), np.array(isvs)])
                df['Signal_Velocity'] = np.concatenate([np.array([np.nan]), sds])
                df['Total_Distance_Travelled'] = np.concatenate([np.array([np.nan]), tds])
                df['Euclidean_Distance_Travelled'] = np.concatenate([np.array([np.nan]), eds])


                #set the k order for interpolation to the max possible
                if len(df)<6:
                    kay = len(df)-1
                else:
                    kay = 5

                #     seg_nucleus[t,:,:,:] = segment_nucleus(temp_im[t,1,:,:,:])
                pos = df[['x','y','z']]
                if bool(pos[pos.duplicated()].index.tolist()):
                    ######### FIND CELL TRAJECTORY AND EULER ANGLES ################
                    # if there is duplicate positions
                    dups = pos[pos.duplicated()].index.tolist()
                    pos_drop = pos.drop(dups, axis = 0)
                    if pos_drop.shape[0]==2:
                        traj = np.zeros((3,3))
                        traj[pos_drop.index.values.max(),:] = pos_drop.iloc[1].values- pos_drop.iloc[0].values
                    elif pos_drop.shape[0] < 2:
                        traj = np.zeros((pos.shape[0],3))
                    else:
                        #get trajectories without the duplicates
                        tck, u = interpolate.splprep(pos_drop.to_numpy().T, k=kay)
                        yderv = interpolate.splev(u,tck,der=1)
                        traj = np.vstack(yderv).T
                        #re-insert duplicate row that was dropped
                        for d, dd in enumerate(dups):
                            traj = np.insert(traj, dd, traj[dd-1,:], axis=0)

                else:
                    ######### FIND CELL TRAJECTORY AND EULER ANGLES ################
                    #no duplicate positions
                    #interpolate and get tangent at midpoint
                    tck, b = interpolate.splprep(pos.to_numpy().T, k=kay)
                    yderv = interpolate.splev(b,tck,der=1)
                    traj = np.vstack(yderv).T

                #get the trajectory and the previous trajectory for each frame and 
                #save as an individual dataframe for each cell and frame
                for v, row in df.iterrows():
                    row = df.loc[v]
                    if v==0:
                        row['Prev_Trajectory_X'] = np.nan
                        row['Prev_Trajectory_Y'] = np.nan
                        row['Prev_Trajectory_Z'] = np.nan
                        row['Trajectory_X'] = traj[v,0]
                        row['Trajectory_Y'] = traj[v,1]
                        row['Trajectory_Z'] = traj[v,2]
                        row['Turn_Angle'] = np.nan
                        pd.DataFrame(row.to_dict(),index=[0]).to_csv(savedir + row.cell + '_cell_info.csv')
                    if v>0:
                        row['Prev_Trajectory_X'] = traj[v-1,0]
                        row['Prev_Trajectory_Y'] = traj[v-1,1]
                        row['Prev_Trajectory_Z'] = traj[v-1,2]
                        row['Trajectory_X'] = traj[v,0]
                        row['Trajectory_Y'] = traj[v,1]
                        row['Trajectory_Z'] = traj[v,2]
                        row['Turn_Angle'] = angle_distance(traj[v-1,0], traj[v-1,1], traj[v-1,2], traj[v,0], traj[v,1], traj[v,2])
                        pd.DataFrame(row.to_dict(),index=[0]).to_csv(savedir + row.cell + '_cell_info.csv')
                        
                        
                        
                        
                        
########## find a cell of interest ###########
savedir = 'D:/Aaron/Data/Combined_Confocal_PCA/'
infrsavedir = savedir + 'Inframe_Videos/'

TotalFrame = pd.read_csv(savedir + 'Shape_Metrics_transitionPCbins.csv', index_col=0)


#find the length of cell consecutive frames
results = []
for i, cells in TotalFrame.groupby('CellID'):
    cells = cells.sort_values('frame').reset_index(drop = True)
    runs = list()
    #######https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
    for k, g in groupby(enumerate(cells['frame']), lambda ix: ix[0] - ix[1]):
        currentrun = list(map(itemgetter(1), g))
        list.append(runs, currentrun)
    maxrun = max([len(l) for l in runs])
    actualrun = max(runs, key=len, default=[])
    results.append([i, maxrun, actualrun])
#find
stdf = pd.DataFrame(results, columns = ['CellID','length_of_run','actual_run']).sort_values('length_of_run', ascending=False).reset_index(drop=True)
stdf.head(30)


#select cell from list above
row = stdf.loc[10]
print(row.CellID)
#get the data related to this run of this cell
data = TotalFrame[(TotalFrame.CellID==row.CellID) & (TotalFrame.frame.isin(row.actual_run))].sort_values('frame').reset_index(drop=True)


ax = plt.figure().add_subplot(projection='3d')
ax.plot(data.x, data.y, data.z, label='DMSO cell')
ax.legend()
plt.show()




#smoothen the track
tck, b = interpolate.splprep(data[['x','y','z']].to_numpy().T, k=5, s=120)
yderv = interpolate.splev(b,tck,der=0)
traj = np.vstack(yderv).T

ax = plt.figure().add_subplot(projection='3d')
ax.plot(traj[:,0], traj[:,1], traj[:,2], label='cell 13 smooth s1 k5')
ax.legend()
plt.show()


longdistmatrix = distance.pdist(data[['x','y','z']])
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
print(dist.sum())

########### EUCLIDEAN
FL = data[['x','y','z']].iloc[[0,-1]]
euc_dist = distance.pdist(FL[['x','y','z']])
print(euc_dist)


#### 8/1 s is 200
#### 2/1 is 50




#smoothen the track
tck, b = interpolate.splprep(data[['x','y','z']].to_numpy().T, k=5, s=200)
yderv = interpolate.splev(b,tck,der=1)
traj = np.vstack(yderv).T
savedir = 'C:/Users/Aaron/Documents/Python Scripts/temp/'
#get the trajectory and the previous trajectory for each frame and 
#save as an individual dataframe for each cell and frame
for v, row in data.iterrows():
    row = data.loc[v]
    if v==0:
        row['Prev_Trajectory_X'] = np.nan
        row['Prev_Trajectory_Y'] = np.nan
        row['Prev_Trajectory_Z'] = np.nan
        row['Trajectory_X'] = traj[v,0]
        row['Trajectory_Y'] = traj[v,1]
        row['Trajectory_Z'] = traj[v,2]
        row['Turn_Angle'] = np.nan
        pd.DataFrame(row.to_dict(),index=[0]).to_csv(savedir + row.cell + '_cell_info.csv')
    if v>0:
        row['Prev_Trajectory_X'] = traj[v-1,0]
        row['Prev_Trajectory_Y'] = traj[v-1,1]
        row['Prev_Trajectory_Z'] = traj[v-1,2]
        row['Trajectory_X'] = traj[v,0]
        row['Trajectory_Y'] = traj[v,1]
        row['Trajectory_Z'] = traj[v,2]
        row['Turn_Angle'] = angle_distance(traj[v-1,0], traj[v-1,1], traj[v-1,2], traj[v,0], traj[v,1], traj[v,2])
        pd.DataFrame(row.to_dict(),index=[0]).to_csv(savedir + row.cell + '_cell_info.csv')
        
        






from CustomFunctions import shparam_mod


savedir = 'C:/Users/Aaron/Documents/Python Scripts/temp/'
#make dirs if it doesn't exist
datadir = 'D:/Aaron/Data/HL60 ParaNitroBlebbistatin/Data_and_Figs/'
if not os.path.exists(datadir):
    os.makedirs(datadir)
meshf = savedir+'Meshes/'  
if not os.path.exists(meshf):
    os.makedirs(meshf)


    
xyres = 0.3394 #um / pixel 
zstep = 0.7 # um
align_method = 'trajectory'
norm_rot = 'widest weighted'
l_order = 10
nisos = [1,63]
sigma = 0
errorlist = []



#get all segmented images
imlist = [x for x in os.listdir(savedir) if x.endswith('segmented.tiff')]

start = 0
stop = 300
allresults = []
while start<len(imlist):
    print(f'Finished {start}, starting {start}-{stop}')
    if __name__ ==  '__main__':
        results = []
        pool = multiprocessing.Pool(processes=60)
        for i in imlist[start:stop]:
            #get path to segmented image
            impath = savedir + i

            if 'actin' in impath:
                str_name = 'actin'
            elif 'Hoechst' in impath:
                str_name = 'nucleus'
            elif 'myosin' in impath:
                str_name = 'myosin'
            elif 'DNA' in impath:
                str_name = 'nucleus'
            else:
                str_name = ''
                
            #put in the pool
            pool.apply_async(shparam_mod.shcoeffs_and_PILR_nonuc, args = (
                impath,
                savedir,
                xyres,
                zstep,
                str_name,
                errorlist,
                norm_rot,
                l_order,
                nisos,
                sigma,
                align_method,
                ),             
                callback = collect_results)

        pool.close()
        pool.join()
    
    allresults.extend(results)
    
    start = stop + 1
    stop = stop + 1000
    if stop>len(imlist):
        stop = len(imlist)

errorlist = []
bigdf = pd.DataFrame()

# for r in allresults:
    

#     Shape_Stats = pd.DataFrame([r[0].values()],
#                                   columns = list(r[0].keys()))
#     cell_coeffs = pd.DataFrame([r[1].values()],
#                                columns = list(r[1].keys()))

#     bigdf = bigdf.append(pd.concat([Shape_Stats,cell_coeffs], axis=1))

#     errorlist.extend(r[2])


# #save the shape metrics dataframe
# bigdf = bigdf.set_index('cell')
# bigdf.to_csv(datadir + 'Shape_Metrics.csv')

# #save list of cells that don't have centroid in shape
# pd.Series(errorlist).to_csv(datadir + 'ListToExclude.csv')

