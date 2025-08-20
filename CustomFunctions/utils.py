

import pandas as pd
import numpy as np
import re
from scipy import interpolate


def running_mean_withna(x, N):
    means = []
    for i, r in enumerate(x):
        if np.isnan(r):
            means.append(np.nan)
        elif i<N:
            #get the window to average
            wind = x[:int(i+1)]
            #remove nan
            wind = wind[~np.isnan(wind)]
            #get average
            means.append(np.mean(wind))
        else:
            #get the indicies around the target value
            first = i - N//2+N%2
            second = first + N
            wind = x[first:second]
            #remove nan
            wind = wind[~np.isnan(wind)]
            #get average
            means.append(np.mean(wind))

    return np.array(means)



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
        image_interval, #time between frames
        timespan, #integer number of image intervals to calculate velocity 
        ):
    
    cell, runs = get_consecutive_timepoints(df, 'time', image_interval)
    #iterate through consecutive frames
    speeds = []
    velocities = []
    for r in runs:
        rundf = cell.iloc[r].copy().reset_index(drop=True)
        for i in range(len(rundf)):
            #get rows of current and previous timepoints
            cur = rundf.iloc[i]
            prev = rundf.shift(-timespan).iloc[i]
            #add nan if there's no data for this timepoint
            if all(prev.isna()):
                velocities.append(np.nan)
                speeds.append(np.nan)
            else:
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
                velocities.append(projdist/(image_interval*timespan))
                speeds.append(dist_3d([0,0,0], smoothvec)/(image_interval*timespan))
            
    cell.loc[:,f'velocity_span_{timespan}'] = velocities
    cell.loc[:,f'speed_span_{timespan}'] = speeds
    
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




def get_aer_state(
        cell, #dataframe of the cell to be thresholded
        time_interval, #imaging time interval
        derivthresh = 0.0007, #derivative threshold to call increasing or decreasing aer
        ):
    
    #ensure the cell is in time order
    cell = cell.sort_values('time').reset_index(drop=True)
    #get rid of NA in aer which will ruin cumulative sums etc.
    cellnona = cell[~cell.aer.isna()].copy()
    #### weight the points near gaps more
    diffs = cellnona.time.diff().values
    #get the indicies of jumps
    gaps = np.where(diffs>time_interval)[0]
    #add the indices before jumps
    gaps = np.concatenate((gaps,gaps-1))
    w = np.ones(diffs.shape)
    w[gaps] = 3

    # ####running mean method
    # deriv = np.gradient(utils.running_mean_withna(cell.aer.cumsum(), 25), cell.time.values)
    ####interpolation method
    #interpolate for smoothening
    tck, u = interpolate.splprep(np.array((cellnona.time.values,
                                           cellnona.aer.cumsum().values)),
                                 k=3, s = 1, w = w)#k=1, s=2, w = w)
    x, y = interpolate.splev(u, tck, der=0)
    #get the derivative of the smoothened curve
    deriv = np.gradient(y, x)
    #threshold with np.select
    threshs = [deriv>=derivthresh, deriv<=-derivthresh]
    choices = ['increasing', 'decreasing']
    statethresh = np.select(threshs, choices, default = 'unchanging')
    #add new values to dataframe
    cell.loc[cellnona.index,'aer_deriv'] = deriv
    cell.loc[cellnona.index,'aer_state'] = statethresh

    return cell, tck, w


#### bootstrap a confidence interval similar to seaborn
def bs_ci(values, #distribution to sample from
          iterations = 1000, #how many times to sample
          ):
    
    if type(values) != np.ndarray:
        values = np.array(values)
    #remove nan
    values = values[~np.isnan(values)]
    leng = len(values)
    iters = np.zeros((iterations))
    for i in range(iterations):
        sample_inds = np.random.randint(0,leng,leng)
        sample = values[sample_inds]
        iters[i] = sample.mean()
    #calculate 95% percentile interval
    lower = np.percentile(iters, 2.5)
    upper = np.percentile(iters, 97.5)
    
    return lower, upper