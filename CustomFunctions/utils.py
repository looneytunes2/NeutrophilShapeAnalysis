

import pandas as pd
import numpy as np
import re



def get_consecutive_timepoints(
        df, #dataframe
        column: str, #string column to get consecutive timepoints from
        interval: int, #expected interval of "column"
        ):
    #sort the dataframe based on the column
    df_sorted = df.sort_values(column).reset_index(drop = True)
    #get differences over the column
    diff = df_sorted[column].diff()
    #create a list of all the places with time jumps starting with 0
    difflist = [0]
    difflist.extend(diff[diff>interval].index.to_list())
    if difflist[-1] < len(df_sorted):
        difflist.append(len(df_sorted))
    #make a list of lists with the indices of consecutive time points
    runs = [list(range(difflist[x], difflist[x+1])) for x in range(len(difflist)-1)]
    
    return df_sorted, runs
    

#get distance between two points in 3d
def dist_3d(p1,p2):
    return np.sqrt(np.sum((p2-p1)**2))


#project vector a onto vector b
def project_vector(a, b):
    b_norm_sq = np.dot(b, b)
    if b_norm_sq == 0:
        raise ValueError("Cannot project onto a zero vector.")
    projection = (np.dot(a, b) / b_norm_sq) * b
    return projection


#### sorts data for an individual cell and adds the raw speed projected onto
#### the smoothened trajectory
def project_raw_smooth(
        df, #dataframe of a cell with raw and smoothened x,y,z positions
        time_interval, #time between frames
        ):
    
    cell, runs = get_consecutive_timepoints(df, 'time', time_interval)
    #iterate through consecutive frames
    projected_speed = []
    for r in runs:
        rundf = cell.iloc[r].copy().reset_index(drop=True)
        #start with empty for timepoint zero in a consecutive run
        projected_speed.append(np.nan)
        for i in range(1, len(rundf)):
            #get rows of current and previous timepoints
            cur = rundf.iloc[i]
            prev = rundf.iloc[i-1]
            #get smooth and raw trajectory vectors
            smoothvec = np.array([cur.x-prev.x, cur.y-prev.y, cur.z-prev.z])
            rawvec = np.array([cur.x_raw-prev.x_raw, cur.y_raw-prev.y_raw, cur.z_raw-prev.z_raw])
            #project the raw vector and get the distance
            rawproj = project_vector(rawvec, smoothvec)
            projdist = dist_3d([0,0,0], rawproj)
            if (smoothvec[0]>0) and (rawproj[0]<0):
                projdist *= -1
            elif (smoothvec[0]<0) and (rawproj[0]>0):
                projdist *= -1
            projected_speed.append(projdist/time_interval)
            
            print(projdist/time_interval, cur.speed)
    cell['raw_projected_speed'] = projected_speed
    
    return cell
            
      
            
def filename_match_llscellid(
        cellid, #CellID of cell in question
        lst, #list of file names
        ):
    movie = '_'.join(cellid.split('_')[:-1])
    cellinmovie = cellid.split('_')[-1]
    filematches = []
    for l in lst:
        if movie in l:
            if re.search(r'\d+', l.split('Subset-')[-1])[0] == cellinmovie:
                filematches.append(l)
    return filematches