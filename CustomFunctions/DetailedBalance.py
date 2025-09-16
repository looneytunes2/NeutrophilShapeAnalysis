# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:52:46 2023

@author: Aaron
"""

from scipy import interpolate
from scipy.spatial import distance
import pandas as pd
import numpy as np
from random import shuffle
from CustomFunctions import utils
import multiprocessing
import itertools
import math
import tqdm




def signed_angle(u,v):
    return math.degrees(math.atan2( u[0]*v[1] - u[1]*v[0], u[0]*v[0] + u[1]*v[1] ))

def clock_counterclock_angle(u,v):
    return -signed_angle(u,v)

def angle_between_vectors(u, v):
    dot_product = sum(i*j for i, j in zip(u, v))
    norm_u = math.sqrt(sum(i**2 for i in u))
    norm_v = math.sqrt(sum(i**2 for i in v))
    cos_theta = dot_product / (norm_u * norm_v)
    if round(cos_theta,8) == 1:
        angle_rad = 0
        angle_deg = 0
    elif round(cos_theta,8) == -1:
        angle_rad = np.pi
        angle_deg = 180
    else:
        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)
    return angle_rad, angle_deg

def contour_coords_slant_corners(
        uple, #[x,y] list of upper left coordinate of rectangular contour
        lori, #[x,y] list of lower right coordinate of rectangular contour
        ):
    contourcoords = []
    #add upper side of box
    contourcoords.extend([[n,uple[1]] for n in range(uple[0]+1, lori[0])])
    #right side minus upper left corner
    contourcoords.extend([[lori[0],n] for n in reversed(range(lori[1]+1,uple[1]))])
    #lower side minus lower right corner
    contourcoords.extend([[n,lori[1]] for n in reversed(range(uple[0]+1,lori[0]))])
    #lower left to upper right
    contourcoords.extend([[uple[0],n] for n in range(lori[1]+1,uple[1])])
    #add the first coordinate to the end
    contourcoords.extend([contourcoords[0]])
    return contourcoords


def contour_coords(
        uple, #[x,y] list of upper left coordinate of rectangular contour
        lori, #[x,y] list of lower right coordinate of rectangular contour
        ):
    contourcoords = []
    #add upper side of box
    contourcoords.extend([[n,uple[1]] for n in range(uple[0], lori[0]+1)])
    #right side minus upper left corner
    contourcoords.extend([[lori[0],n] for n in reversed(range(lori[1],uple[1]))])
    #lower side minus lower right corner
    contourcoords.extend([[n,lori[1]] for n in reversed(range(uple[0],lori[0]))])
    #left side minus lower and upper left corners
    contourcoords.extend([[uple[0],n] for n in range(lori[1]+1,uple[1])])
    #add the first coordinate to the end
    contourcoords.extend([contourcoords[0]])
    return contourcoords

def raw_transitions(
        time_interval, # time interval between frames in seconds
        df, # pandas dataframe with cell, CellID, frame, and binned PCs
        whichpcs, #which pc #s are in the cgps in [x,y]
        ):
    
    trans = [] #list to append transitions
    ct = 0 #cumulative time count
    for i, row in df.reset_index(drop=True).iterrows():
        if i < len(df)-1:
            ct = ct + time_interval
            nextrow = df.iloc[i+1]
            #frame will reference the timepoint at the end of the transition
            trans.append([nextrow.time, nextrow.frame, row[f'PC{whichpcs[0]}bins'], row[f'PC{whichpcs[1]}bins'],
                          nextrow[f'PC{whichpcs[0]}bins'], nextrow[f'PC{whichpcs[1]}bins'], time_interval, ct])
            
        
    #combine the data
    alltrans = pd.DataFrame(trans, columns=['real_time','frame', 'from_x', 'from_y', 'to_x', 'to_y', 'time_elapsed','cumulative_time'])
    #add cell identification
    alltrans['CellID'] = df.CellID.to_list()[:-1]
    # 'cell' will reference the cell/frame at the end of the transition
    alltrans['cell'] = df.cell.to_list()[1:]
    
    return alltrans



def interpolate_2dtrajectory(
        t_int, # time interval between frames in seconds
        rawtrans, # continuous-time dataframe sorted by frame # 
        ):

    
    cellname = rawtrans.CellID.iloc[0]
    frames = rawtrans.frame.to_list()
    #get the CGPS POSITIONS for this trajectory segment
    traj = np.vstack((rawtrans[['from_x','from_y']].values,
                      rawtrans[['to_x','to_y']].iloc[-1].values))
    
    
    #remove duplicate coordinates
    #which breaks the interpolation function
    #first make sure numpy array dtype is correct
    traj = traj.astype(np.float32)
    #find the indicies of the duplicates
    duplicates = [i for i,w in enumerate(traj) if all(w==traj[i-1])]
    #add a small number to the duplicates so they're not the same, but not meaningfully different
    for d in duplicates:
        traj[d,:] = traj[d,:]+0.001
    
    #interpolate based on path
    tck, b = interpolate.splprep(traj.T, u=range(len(traj)),k=1, s=0)
    
    #measure the trajectory and interpolate evenly by distance
    interlist = []
    
    for t in range(len(traj)-1):
        di = distance.pdist([traj[t,:],traj[t+1,:]])[0]
        intt = round(di/0.1)
        #if there's at least one bin position change during this frame, interpolate to find when it happens
        if intt>0:
            interpoints = np.linspace(start=t, stop = t+1, num = intt, endpoint = False)
            x, y = interpolate.splev(interpoints,tck)
            x = [round(i) for i in x]
            y = [round(i) for i in y]
            fr = [frames[t]]*len(interpoints)
            interlist.append(np.stack([fr,x,y,interpoints]).T)
        #if the cell doesn't actually change bin positions in this frame, just add it's info
        else:
            fr = frames[t]
            interlist.append(np.array([[fr,traj[t][0],traj[t][1],t]]))
    
    #add last position
    interlist.append(np.array([[frames[-1], traj[-1,0], traj[-1,1], frames[-1]]]))
    #concatenate all
    fulltr = pd.DataFrame(np.concatenate(interlist), columns=['frame','x','y','t'])
    
    #find all single move transitions
    trans = []
    prev = pd.Series([frames[0],traj[0,0],traj[0,1],0], index=['frame','x','y','t'])
    for i, g in fulltr.diff().iterrows():
        ### provide an escape if the interpolation is still not good enough and there
        ### is a >1 jump in the trajectory
        if (abs(g.x)>=1) and (abs(g.y)>=1):
            extra = np.linspace(prev.t,fulltr.iloc[i].t,30)
            ex, ey = interpolate.splev(extra,tck)
            ex = [round(i) for i in ex]
            ey = [round(i) for i in ey]
            ef = [fulltr.iloc[i].frame]*len(extra)
            exdf = pd.DataFrame(np.stack([ef,ex,ey,extra]).T, columns=['frame','x','y','t'])
            for h, j in exdf.diff().iterrows():
                ### if there's STILL a transition by more than a single move
                ### then it means the slope of the transition is 1 and needs to
                ### have the transitions to adjacent boxes decided randomly
                if (abs(j.x)>=1) and (abs(j.y)>=1):
                    cur = exdf.iloc[h]
                    possible = ['x','y']
                    shuffle(possible)
                    if possible[0]=='x':
                        trans.append([cur.frame, prev.x, prev.y, cur.x, prev.y, (cur.t-prev.t)/2, cur.t])
                        trans.append([cur.frame, cur.x, prev.y, cur.x, cur.y, (cur.t-prev.t)/2, cur.t+(cur.t-prev.t)/2])
                        prev = cur.copy()
                    else:
                        trans.append([cur.frame, prev.x, prev.y, prev.x, cur.y, (cur.t-prev.t)/2, cur.t])
                        trans.append([cur.frame, prev.x, cur.y, cur.x, cur.y, (cur.t-prev.t)/2, cur.t+(cur.t-prev.t)/2])
                        prev = cur.copy()
                elif (abs(j.x)==1) or (abs(j.y)==1):
                    cur = exdf.iloc[h]
                    trans.append([cur.frame, prev.x, prev.y, cur.x, cur.y, cur.t-prev.t, cur.t])
                    prev = cur.copy()
        #collect all of the 1 moves
        elif (abs(g.x)==1) or (abs(g.y)==1):
            cur = fulltr.iloc[i]
            trans.append([cur.frame, prev.x, prev.y, cur.x, cur.y, cur.t-prev.t, cur.t])
            prev = cur.copy()
        #ignore timepoints that don't transition
        else:
            pass

        
    #combine the data
    alltrans = pd.DataFrame(trans, columns=['frame', 'from_x', 'from_y', 'to_x', 'to_y', 'time_elapsed','cumulative_time'])
    #add cell name
    alltrans['CellID'] = cellname
    #also add the frame identifier just in case
    celllist = rawtrans.cell.to_list()
    alltrans['cell'] = [celllist[frames.index(x)] for x in alltrans.frame.to_list()]
    #adjust time elapsed and cumulative time to real time
    alltrans['time_elapsed'] = alltrans['time_elapsed']*t_int
    alltrans['cumulative_time'] = alltrans['cumulative_time']*t_int
    #add real image time so that data can be sorted even if it's not
    #from the same video
    alltrans['real_time'] = alltrans.cumulative_time + rawtrans.real_time.iloc[0]
    
    return alltrans.to_dict('records')



def interpolate_3dtrajectory(
        t_int,
        cellname,
        frames,
        traj,
        ):
    #remove duplicate coordinates
    #which breaks the interpolation function
    duplicates = [i for i,w in enumerate(traj) if all(w==traj[i-1])]
    for d in duplicates:
        traj[d,:] = traj[d,:]+0.001
    
    #interpolate based on path
    tck, b = interpolate.splprep(traj.T)
    #time between frames normalized between 0 and 1
    int_int = 1/(len(traj)-1)
    
    #measure the trajectory and interpolate evenly by distance
    interlist = []
    for t in range(len(traj)-1):
        di = distance.pdist([traj[t,:],traj[t+1,:]])[0]
        intt = round(di/0.1)
        interpoints = np.linspace(start=t*int_int, stop = t*int_int+int_int, num = intt, endpoint = False)
        x, y, z = interpolate.splev(interpoints,tck)
        x = [round(i) for i in x]
        y = [round(i) for i in y]
        z = [round(i) for i in z]
        fr = [frames[t]]*len(interpoints)
        interlist.append(np.stack([fr,x,y,z,interpoints]).T)
    
    #add last position
    interlist.append(np.array([[frames[-1], traj[0,-1], traj[1,-1], traj[2,-1], 1]]))
    #concatenate all
    fulltr = pd.DataFrame(np.concatenate(interlist), columns=['frame','x','y','z','t'])
    
    #find all single move transitions
    trans = []
    prev = pd.Series([frames[0],traj[0,0],traj[0,1],traj[0,2],0], index=['frame','x','y','z','t'])
    for i, g in fulltr.diff().iterrows():
        ### provide an escape if the interpolation is still not good enough and there
        ### is a >1 jump in the trajectory
        if ((abs(g.x)>=1) and (abs(g.y)>=1)) or ((abs(g.x)>=1) and (abs(g.z)>=1)) or ((abs(g.y)>=1) and (abs(g.z)>=1)):
            extra = np.linspace(prev.t,fulltr.iloc[i].t,30)
            ex, ey, ez = interpolate.splev(extra,tck)
            ex = [round(i) for i in ex]
            ey = [round(i) for i in ey]
            ez = [round(i) for i in ez]
            ef = [fulltr.iloc[t].frame]*len(extra)
            exdf = pd.DataFrame(np.stack([ef,ex,ey,ez,extra]).T, columns=['frame','x','y','z','t'])
            for h, j in exdf.diff().iterrows():
                ### if there's STILL a transition by more than a single move
                ### then it means the slope of the transition is 1 and needs to
                ### have the transitions to adjacent boxes decided randomly
                if ((abs(j.x)>=1) and (abs(j.y)>=1)):
                    cur = exdf.iloc[h]
                    possible = ['x','y']
                    shuffle(possible)
                    if possible[0]=='x':
                        trans.append([prev.frame, prev.x, prev.y, prev.z, cur.x, prev.y, cur.z, (cur.t-prev.t)/2, cur.t])
                        trans.append([prev.frame, cur.x, prev.y, cur.x, cur.y, cur.z, (cur.t-prev.t)/2, cur.t])
                        prev = cur.copy()
                    else:
                        trans.append([prev.frame, prev.x, prev.y, prev.z, prev.x, cur.y, cur.z, (cur.t-prev.t)/2, cur.t])
                        trans.append([prev.frame, prev.x, cur.y, cur.z, cur.x, cur.y, cur.z, (cur.t-prev.t)/2, cur.t])
                        prev = cur.copy()
                elif ((abs(j.x)>=1) and (abs(j.z)>=1)):
                    cur = exdf.iloc[h]
                    possible = ['x','z']
                    shuffle(possible)
                    if possible[0]=='x':
                        trans.append([prev.frame, prev.x, prev.y, prev.z, cur.x, cur.y, prev.z, (cur.t-prev.t)/2, cur.t])
                        trans.append([prev.frame, cur.x, cur.y, prev.z, cur.x, cur.y, cur.z, (cur.t-prev.t)/2, cur.t])
                        prev = cur.copy()
                    else:
                        trans.append([prev.frame, prev.x, prev.y, prev.z, prev.x, cur.y, cur.z, (cur.t-prev.t)/2, cur.t])
                        trans.append([prev.frame, prev.x, cur.y, cur.z, cur.x, cur.y, cur.z, (cur.t-prev.t)/2, cur.t])
                        prev = cur.copy()
                elif ((abs(j.y)>=1) and (abs(j.z)>=1)):
                    cur = exdf.iloc[h]
                    possible = ['y','z']
                    shuffle(possible)
                    if possible[0]=='y':
                        trans.append([prev.frame, prev.x, prev.y, prev.z, cur.x, cur.y, prev.z, (cur.t-prev.t)/2, cur.t])
                        trans.append([prev.frame, cur.x, cur.y, prev.z, cur.x, cur.y, cur.z, (cur.t-prev.t)/2, cur.t])
                        prev = cur.copy()
                    else:
                        trans.append([prev.frame, prev.x, prev.y, prev.z, cur.x, prev.y, cur.z, (cur.t-prev.t)/2, cur.t])
                        trans.append([prev.frame, cur.x, prev.y, cur.z, cur.x, cur.y, cur.z, (cur.t-prev.t)/2, cur.t])
                        prev = cur.copy()
                elif (abs(j.x)==1) or (abs(j.y)==1) or (abs(j.z)==1):
                    cur = exdf.iloc[h]
                    trans.append([prev.frame, prev.x, prev.y, prev.z, cur.x, cur.y, cur.z, cur.t-prev.t, cur.t])
                    prev = cur.copy()
        #collect all of the 1 moves
        elif (abs(g.x)==1) or (abs(g.y)==1) or (abs(g.z)==1):
            cur = fulltr.iloc[i]
            trans.append([prev.frame, prev.x, prev.y, prev.z, cur.x, cur.y, cur.z, cur.t-prev.t, cur.t])
            prev = cur.copy()
        #ignore timepoints that don't transition
        else:
            pass

        
    #combine the data
    alltrans = pd.DataFrame(trans, columns=['frame', 'from_x', 'from_y', 'from_z', 'to_x', 'to_y', 'to_z', 'time_elapsed','cumulative_time'])
    #add cell name
    alltrans['CellID'] = cellname
    #adjust time elapsed and cumulative time to real time
    alltrans['time_elapsed'] = alltrans['time_elapsed']*t_int*(len(traj)-1)
    alltrans['cumulative_time'] = alltrans['cumulative_time']*t_int*(len(traj)-1)
    
    #also get transition pairs for boostrapping
    pairs = [trans[i]+trans[i+1] for i in range(len(trans[:-1]))] 
    transpairs = pd.DataFrame(pairs, columns=['frame', 'from_x', 'from_y', 'from_z', 'to_x', 'to_y', 'to_z', 'time_elapsed','cumulative_time', \
                                              'frame_two', 'from_x_two', 'from_y_two', 'from_z_two', 'to_x_two', 'to_y_two', 'to_z_two', 'time_elapsed_two','cumulative_time_two'])
    #add cell name
    transpairs['CellID'] = cellname
    transpairs['time_elapsed'] = transpairs['time_elapsed']*t_int*(len(traj)-1)
    transpairs['cumulative_time'] = transpairs['cumulative_time']*t_int*(len(traj)-1)
    transpairs['time_elapsed_two'] = transpairs['time_elapsed_two']*t_int*(len(traj)-1)
    transpairs['cumulative_time_two'] = transpairs['cumulative_time_two']*t_int*(len(traj)-1)
    
    #double check for bad 
    # any((abs(alltrans.from_x-alltrans.to_x) + abs(alltrans.from_y-alltrans.to_y))!=1)
    return [x.to_dict() for i, x in alltrans.iterrows()], [x.to_dict() for i, x in transpairs.iterrows()]


def get_transition_counts(
        x,
        y,
        fromm, #all the transitions from a particular box
        to, #all the transitions to that same box
        ttot, #total time represented by the experiment
        ):
    
    #get the rate going over the - x side of the box (rate going left)
    x_minus_count_for = len([fromm['to_x'][a] for a in fromm['to_x'] if fromm['to_x'][a]<x])
    x_minus_for_rate = x_minus_count_for/ttot
    x_minus_count_rev = len([to['from_x'][a] for a in to['from_x'] if to['from_x'][a]<x])
    x_minus_rev_rate = x_minus_count_rev/ttot
    x_minus_rate = (x_minus_count_for - x_minus_count_rev)/ttot
    
    #get the rate going over the + x side of the box (rate going right)
    x_plus_count_for = len([fromm['to_x'][a] for a in fromm['to_x'] if fromm['to_x'][a]>x])
    x_plus_for_rate = x_plus_count_for/ttot
    x_plus_count_rev = len([to['from_x'][a] for a in to['from_x'] if to['from_x'][a]>x])
    x_plus_rev_rate = x_plus_count_rev/ttot
    x_plus_rate = (x_plus_count_for - x_plus_count_rev)/ttot
    
    #get the rate going over the - y side of the box (rate going down)
    y_minus_count_for = len([fromm['to_y'][a] for a in fromm['to_y'] if fromm['to_y'][a]<y])
    y_minus_for_rate = y_minus_count_for/ttot
    y_minus_count_rev = len([to['from_y'][a] for a in to['from_y'] if to['from_y'][a]<y])
    y_minus_rev_rate = y_minus_count_rev/ttot
    y_minus_rate = (y_minus_count_for - y_minus_count_rev)/ttot
    
    #get the rate going over the + y side of the box (rate going up)
    y_plus_count_for = len([fromm['to_y'][a] for a in fromm['to_y'] if fromm['to_y'][a]>y])
    y_plus_for_rate = y_plus_count_for/ttot
    y_plus_count_rev = len([to['from_y'][a] for a in to['from_y'] if to['from_y'][a]>y])
    y_plus_rev_rate = y_plus_count_rev/ttot
    y_plus_rate = (y_plus_count_for - y_plus_count_rev)/ttot

    trans_count = {
        'x':x,
        'y':y,
        'x_minus_count':x_minus_count_for,
        'x_minus_count_rev':x_minus_count_rev,
        'x_minus_for_rate':x_minus_for_rate,
        'x_minus_rev_rate':x_minus_rev_rate,
        'x_minus_rate':x_minus_rate,
        'x_plus_count':x_plus_count_for,
        'x_plus_count_rev':x_plus_count_rev,
        'x_plus_for_rate':x_plus_for_rate,
        'x_plus_rev_rate':x_plus_rev_rate,
        'x_plus_rate':x_plus_rate,
        'y_minus_count':y_minus_count_for,
        'y_minus_count_rev':y_minus_count_rev,
        'y_minus_for_rate':y_minus_for_rate,
        'y_minus_rev_rate':y_minus_rev_rate,
        'y_minus_rate':y_minus_rate,
        'y_plus_count':y_plus_count_for,
        'y_plus_count_rev':y_plus_count_rev,
        'y_plus_for_rate':y_plus_for_rate,
        'y_plus_rev_rate':y_plus_rev_rate,
        'y_plus_rate':y_plus_rate
            }
    return trans_count

def get_transition_counts_3d(
        x,
        y,
        z,
        fromm, #all the transitions from a particular box
        to, #all the transitions to that same box
        ttot, #total time represented by the experiment
        ):

    x_minus_count = len([fromm['to_x'][a] for a in fromm['to_x'] if fromm['to_x'][a]<x])
    x_minus_count_rev = len([to['from_x'][a] for a in to['from_x'] if to['from_x'][a]<x])
    x_minus_rate = (x_minus_count - x_minus_count_rev)/ttot
    
    x_plus_count = len([fromm['to_x'][a] for a in fromm['to_x'] if fromm['to_x'][a]>x])
    x_plus_count_rev = len([to['from_x'][a] for a in to['from_x'] if to['from_x'][a]>x])
    x_plus_rate = (x_plus_count - x_plus_count_rev)/ttot
    
    y_minus_count = len([fromm['to_y'][a] for a in fromm['to_y'] if fromm['to_y'][a]<y])
    y_minus_count_rev = len([to['from_y'][a] for a in to['from_y'] if to['from_y'][a]<y])
    y_minus_rate = (y_minus_count - y_minus_count_rev)/ttot
    
    y_plus_count = len([fromm['to_y'][a] for a in fromm['to_y'] if fromm['to_y'][a]>y])
    y_plus_count_rev = len([to['from_y'][a] for a in to['from_y'] if to['from_y'][a]>y])
    y_plus_rate = (y_plus_count - y_plus_count_rev)/ttot

    z_minus_count = len([fromm['to_z'][a] for a in fromm['to_z'] if fromm['to_z'][a]<z])
    z_minus_count_rev = len([to['from_z'][a] for a in to['from_z'] if to['from_z'][a]<z])
    z_minus_rate = (z_minus_count - z_minus_count_rev)/ttot
    
    z_plus_count = len([fromm['to_z'][a] for a in fromm['to_z'] if fromm['to_z'][a]>z])
    z_plus_count_rev = len([to['from_z'][a] for a in to['from_z'] if to['from_z'][a]>z])
    z_plus_rate = (z_plus_count - z_plus_count_rev)/ttot


    trans_count = {
        'x':x,
        'y':y,
        'z':z,
        'x_minus_count':x_minus_count,
        'x_minus_count_rev':x_minus_count_rev,
        'x_minus_rate':x_minus_rate,
        'x_plus_count':x_plus_count,
        'x_plus_count_rev':x_plus_count_rev,
        'x_plus_rate':x_plus_rate,
        'y_minus_count':y_minus_count,
        'y_minus_count_rev':y_minus_count_rev,
        'y_minus_rate':y_minus_rate,
        'y_plus_count':y_plus_count,
        'y_plus_count_rev':y_plus_count_rev,
        'y_plus_rate':y_plus_rate,
        'z_minus_count':z_minus_count,
        'z_minus_count_rev':z_minus_count_rev,
        'z_minus_rate':z_minus_rate,
        'z_plus_count':z_plus_count,
        'z_plus_count_rev':z_plus_count_rev,
        'z_plus_rate':z_plus_rate
            }
    
    return trans_count




def bootstrap_trajectories(
        imap_args
        ):
        
    #unpack args
    # combodf: multi-indexed dataframe with tansition_combination and trandition_index names
    # ttot: int total time for the simulation
    # ntrans: int number of consecutive transitions to sample
    # avoiddead: bool whether or not to avoid dead ends in the trajectory
    combodf,ttot,ntrans,avoiddead = imap_args
    
    #get just the first transition of each combination
    firsttrans = combodf.xs(0,level='transition_index')
    
    #start time at 0
    ct = 0
    #create an empty dataframe with the correct columns and indexing
    allbs = pd.DataFrame(columns=combodf.columns)
    allbs.index = pd.MultiIndex.from_arrays([[],[]], names = ['transition_combination','transition_index'])
    #find the first random position
    rando = combodf.index.levels[0].to_list()
    shuffle(rando)
    pick = combodf.loc[combodf.index.get_level_values('transition_combination') == rando[0]]
    allbs = pd.concat((allbs, pick))
    while ct<ttot:
        #find the next postition after the second transition
        cur = allbs.iloc[-1][['to_x','to_y']].values
        #get all the transitions at the new position
        allat = firsttrans[(firsttrans.from_x == cur[0]) & (firsttrans.from_y == cur[1])]
        
        #if the next transition doesn't have any future transitions, don't go there and pick a new one
        if allat.empty:
            if avoiddead:
                #drop the "dead" transition
                allbs.drop(allbs.index[-ntrans:], inplace=True)
                #check is this happened at the beginning of the simulation and it needs to be started again 
                #from another position, otherwise trim the last transition and continue
                if len(allbs)==0:
                    ct = 0
                    allbs = pd.DataFrame(columns=combodf.columns)
                    allbs.index = pd.MultiIndex.from_arrays([[],[]], names = ['transition_combination','transition_index'])
                    #find the first random position
                    rando = combodf.index.levels[0].to_list()
                    shuffle(rando)
                    pick = combodf.loc[combodf.index.get_level_values('transition_combination') == rando[0]]
                    #add the random pick to the dataframe
                    allbs = pd.concat((allbs, pick))
                #subtract the time these transitions take
                ct = ct - pick.time_elapsed.sum()
                #set a timer for extreme cases of single transitions to deadends
                loops = 0
                while allat.empty:
                    #find the next postition after the second transition
                    cur = allbs.iloc[-1][['to_x','to_y']].values
                    #get all the transitions at the new position
                    allat = firsttrans[(firsttrans.from_x == cur[0]) & (firsttrans.from_y == cur[1])]
                    #randomly select a transition pair
                    rando = allat.index.to_list()
                    shuffle(rando)
                    pick = combodf.loc[combodf.index.get_level_values('transition_combination') == rando[0]]
                    #add to the timer for extreme cases
                    loops = loops + 1
                    #if the current position only has one transition (to the empty position)
                    #then trim it back an additional transition as well
                    #or if this while loop has gone for 20 iterations and still not found a suitable transition
                    #back up an additional transition
                    if (len(allat)==1) or (loops == 20):
                        #subtract the time these transitions take
                        ct = ct - allbs.iloc[-ntrans:].time_elapsed.sum()
                        #delete a further two transitions
                        allbs.drop(allbs.index[-ntrans:], inplace=True)
                        #check if this happened at the beginning of the simulation and it needs to be started again 
                        #from another position, otherwise trim the last transition and continue
                        if len(allbs)==0:
                            ct = 0
                            allbs = pd.DataFrame(columns=combodf.columns)
                            allbs.index = pd.MultiIndex.from_arrays([[],[]], names = ['transition_combination','transition_index'])
                            #find the first random position
                            rando = combodf.index.levels[0].to_list()
                            shuffle(rando)
                            pick = combodf.loc[combodf.index.get_level_values('transition_combination') == rando[0]]
                            #add the random pick to the dataframe
                            allbs = pd.concat((allbs, pick))
                        #find the next postition after the second transition
                        cur = allbs.iloc[-1][['to_x','to_y']].values
                        #get all the transitions at the new position
                        allat = firsttrans[(firsttrans.from_x == cur[0]) & (firsttrans.from_y == cur[1])]
                        #randomly select a transition pair
                        rando = allat.index.to_list()
                        shuffle(rando)
                        pick = combodf.loc[combodf.index.get_level_values('transition_combination') == rando[0]]
                #append the pair of transitions to a list
                allbs = pd.concat((allbs, pick))
                #add the time these transitions take
                ct = ct + pick.time_elapsed.sum()
            else:
                break
        else:
            #randomly select a transition pair
            rando = allat.index.to_list()
            shuffle(rando)
            pick = combodf.loc[combodf.index.get_level_values('transition_combination') == rando[0]]
            #append the pair of transitions to a list
            allbs = pd.concat((allbs, pick))
            #add the time these transitions take
            ct = ct + pick.time_elapsed.sum()
    
    #make cumulative time actually cumulative time
    allbs.loc[:,'cumulative_time'] = allbs['time_elapsed'].cumsum()
    #make a mock "real_time" so that simulated dataframes match real ones
    allbs.loc[:,'real_time'] = allbs.cumulative_time

    return allbs.reset_index(drop=True)



def transition_count_wrapper(
        args # tuple of arguments
        ):
    #unpack args from imap
    #bsdf: transition dataframe from bootstrap_trajectories()
    #nbins: bins in the CGPS
    # ct: the cumulative time actually observed during the simulation, especially important for simulations that terminate early
    bsdf, nbins, ct = args
    
    ############## get the Boosttrapped counts of each bin position ############
    results = []
    for x in range(nbins):
        for y in range(nbins):
            fromm = bsdf[(bsdf['from_x'] == x+1) & (bsdf['from_y'] == y+1)].reset_index(drop=True).to_dict()
            to = bsdf[(bsdf['to_x'] == x+1) & (bsdf['to_y'] == y+1)].reset_index(drop=True).to_dict()
            results.append(get_transition_counts(
                x+1,
                y+1,
                fromm,
                to,
                ct, #use the time actually observed during the simulation, especially important for simulations that terminate early
                ))

    bstrans_rate_df = pd.DataFrame(results)
    bstrans_rate_df = bstrans_rate_df.sort_values(by = ['x','y']).reset_index(drop=True)
    
    return bstrans_rate_df



def contour_integral(
    cdf, #dataframe that contains the transition rates in and out of each state space position
    uple, #[x,y] list of upper left coordinate of rectangular contour
    lori, #[x,y] list of lower right coordinate of rectangular contour
    norm: bool = True,
    ):

    contourcoords = []
    #add upper side of box
    contourcoords.extend([[n,uple[1]] for n in range(uple[0], lori[0]+1)])
    #right side minus upper left corner
    contourcoords.extend([[lori[0],n] for n in reversed(range(lori[1],uple[1]))])
    #lower side minus lower right corner
    contourcoords.extend([[n,lori[1]] for n in reversed(range(uple[0],lori[0]))])
    #left side minus lower and upper left corners
    contourcoords.extend([[uple[0],n] for n in range(lori[1]+1,uple[1])])

    corners = [uple,lori,[lori[0], uple[1]], [uple[0], lori[1]]]
    omega = 0
    dottlist = []
    for i, c in enumerate(contourcoords):
        #get tangent vector
        current = cdf[(cdf.x == c[0]) & (cdf.y == c[1])]
        if current.empty:
            tanv = [0,0]
        else:
            xcurrent = (current.x_plus_rate - current.x_minus_rate)/2
            ycurrent = (current.y_plus_rate - current.y_minus_rate)/2
            tanv = [xcurrent.values[0],ycurrent.values[0]]
        #avoid [positions where positive and negative rates are perfectly balanced]
        if tanv == [0,0]:
            dottlist.append([cdf.bs_iteration.values[0], c[0], c[1], 0])
        else:
            if norm:
                unittan = tanv/np.linalg.norm(tanv)
            else:
                unittan = tanv.copy()
            #get derivative of the contour
            der = -1*(np.array(contourcoords[i-1]) - np.array(c))
            if c in corners:
                cornerline = -1*(np.array(contourcoords[i-1]) - np.array(c) + np.array(c) - np.array(contourcoords[i+1]))
                unitcon = cornerline/np.linalg.norm(cornerline)
                dott = np.dot(unitcon, unittan)
                if not np.isnan(dott):
                    omega = omega + dott
                dottlist.append([cdf.bs_iteration.values[0], c[0], c[1], dott])
            elif der[0]==0:
                dott = np.dot(der, unittan)
                if not np.isnan(dott):
                    omega = omega + dott
                dottlist.append([cdf.bs_iteration.values[0], c[0], c[1], dott])
            elif der[1]==0:
                dott = np.dot(der, unittan)
                omega = omega + dott
                dottlist.append([cdf.bs_iteration.values[0], c[0], c[1], dott])
    return omega, dottlist


   




######## get area enclosing rates the "real" way with individual interpolated transitions
def get_area_enclosing_rate(
        args
        ):
    
    #unpack args for imap
    #cell dataframe with the consecutive CGPS transitions
    #nbins, #number of bins in the CGPS
    #xyscaling = list, # list of the PC factors by which to scale the x and y coordinates of the CGPS in [x,y] format
    #center = 'center',
    cell, nbins, xyscaling, center = args
    
    #get values to shift coordinates to the origin of the current
    if type(center) == list:
        shiftbyx = center[0]
        shiftbyy = center[1]
    elif center == 'center':
        shiftbyx = round(nbins/2)
        shiftbyy = round(nbins/2)
    #calculate aer per transition
    aerlist = []
    avlist = []
    for i, row in cell.iterrows():
        #center the row values on zero and scale them
        row['from_x'] = (row['from_x'] - shiftbyx) * xyscaling[0]
        row['to_x'] = (row['to_x'] - shiftbyx) * xyscaling[0]
        row['from_y'] = (row['from_y'] - shiftbyy) * xyscaling[1]
        row['to_y'] = (row['to_y'] - shiftbyy) * xyscaling[1]
        aerlist.append(
            ((row.from_y*row.to_x) - (row.from_x*row.to_y)) / 
                (2*row.time_elapsed)
                        )
        ######## "For instance, we could track a pair of degrees of freedom 𝐱r={𝑥𝑖,𝑥𝑗} and measure the time average of the angular velocity ⟨̇𝛽𝑖⁢𝑗⟩,
        ######## or equivalently, the rate at which the trajectory revolves around the origin in this reduced two-dimensional subspace (Fig. 2). 
        ######## This simple measurement does not require any discretization of phase space or inference of the force field. 
        ######## We shall refer to ⟨̇𝛽𝑖⁢𝑗⟩ as the cycling frequency.
        ######## https://doi.org/10.1103/PhysRevE.99.052406
        angle_deg = clock_counterclock_angle([row.from_x,row.from_y],[row.to_x,row.to_y])
        avlist.append(
            angle_deg/row.time_elapsed
            )
    cell['aer'] = aerlist
    cell['angular_velocity'] = avlist
    return cell




def get_raw_cgps_trajectories(
        TotalFrame, #pandas dataframe with all of the cgps binned data
        whichpcs, #which two PCs to use in the cgps [x,y]
        time_interval, #real time between datapoints
        savedir, #where to save the aggregated trajectories
        ):
    migresults = []
    for m, Mig in TotalFrame.groupby('Treatment'):
        results = []
        with multiprocessing.Pool(processes=60) as pool:
            for i, cells in Mig.groupby('CellID'):
                cells, runs = utils.get_consecutive_timepoints(cells, 'time', time_interval)
                for r in runs:
                    #skip runs less than 2 frames long
                    if len(r)<2:
                        pass
                    else:
                        cell = cells.iloc[r]
    
                        result = pool.apply_async(raw_transitions, args = (
                            time_interval,
                            cell,
                            whichpcs,
                            ))
                        results.append(result)

            #get results
            results = [r.get() for r in results]
        rawtrans = pd.concat(results, ignore_index=True)
        rawtrans = rawtrans.sort_values(by = ['CellID','real_time']).reset_index(drop=True)
        rawtrans['Treatment'] = m
        migresults.append(rawtrans)
        
    rawtrans = pd.concat(migresults, ignore_index=True)
    rawtrans.to_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_transitions_separated.csv')

    print('Aggregated transitions')
    
    return rawtrans


def get_interpolated_cgps_trajectories(
        rawtrans, #pandas dataframe with raw transitions from get_raw_cgps_trajectories
        whichpcs, #which two PCs to use in the cgps [x,y]
        time_interval, #real time between datapoints
        savedir, #where to save the aggregated trajectories
        ):
    migresults = []
    for m, Mig in rawtrans.groupby('Treatment'):
        results = []
        with multiprocessing.Pool(processes=60) as pool:
            for i, cells in Mig.groupby('CellID'):
                cells, runs = utils.get_consecutive_timepoints(cells, 'real_time', time_interval)
                for r in runs:
                    #skip runs less than 2 frames long
                    if len(r)<2:
                        pass
                    else:
                        cell = cells.iloc[r]
                        result = pool.apply_async(interpolate_2dtrajectory, args = (
                            time_interval,
                            cell,
                            ))
                        results.append(result)
    
            #get results
            results = [r.get() for r in results]
        #separate results into transtions and transition pairs
        transdf_sep = pd.DataFrame([x for r in results for x in r])
        transdf_sep = transdf_sep.sort_values(by = ['CellID','real_time']).reset_index(drop=True)
        transdf_sep['Treatment'] = m
        migresults.append(transdf_sep)

    transdf_sep = pd.concat(migresults)
    transdf_sep.to_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_interpolated_transitions_separated.csv')
    print('Finished interpolating trajectories')
    
    return transdf_sep
    
############## get the counts of cells leaving 
def aggregate_transition_counts(
        transdf_sep, #transdf_sep from get_interpolated_cgps_trajectories
        whichpcs, #which two PCs to use in the cgps [x,y]
        savedir, #where to save the aggregated counts
        nbins, #how many bins in the x and y cgps axes
        ):
    trresults = []
    for m, mig in transdf_sep.groupby('Treatment'):
        ttot = mig.time_elapsed.sum()
        print(f'Total time observed in this CGPS was {ttot/60} minutes')
        pool = multiprocessing.Pool(processes=60)
        results = []
        for x in range(nbins):
            for y in range(nbins):
                fromm = mig[(mig['from_x'] == x+1) & (mig['from_y'] == y+1)].reset_index(drop=True).to_dict()
                to = mig[(mig['to_x'] == x+1) & (mig['to_y'] == y+1)].reset_index(drop=True).to_dict()
                result = pool.apply_async(get_transition_counts, args = (
                    x+1,
                    y+1,
                    fromm,
                    to,
                    ttot,
                    ))
                results.append(result)
        pool.close()
        pool.join()
        
        #get results
        results = [r.get() for r in results]
        trans_rate_df_sep = pd.DataFrame(results)
        trans_rate_df_sep['Treatment'] = m
        trans_rate_df_sep = trans_rate_df_sep.sort_values(by = ['x','y']).reset_index(drop=True)
        trresults.append(trans_rate_df_sep)

    trans_rate_df_sep = pd.concat(trresults)
    trans_rate_df_sep.to_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_binned_transition_rates_separated.csv')
    print('Finished finding transition rates')
    
    return trans_rate_df_sep



############## BOOTSTRAP MANY TRAJECTORIES ##########
def get_bootstrapped_cgps_trajectories(
        rawtrans, #raw transitions from get_raw_cgps_trajectories
        whichpcs, #which two PCs to use in the cgps [x,y]
        time_interval, #real time between datapoints
        savedir, #where to save the aggregated counts
        nbins, #how many bins in the x and y cgps axes
        ttot, #set the total bootstrap time
        ntrans = 1, #how many transitions to sample at each step
        bsiter = 3000, #number of times to bootstrap
        ):
    
    #make a bunch of lists that I will append things to as I go for each treatment
    bstrans = []
    bsint = []
    bsframe_sep_full = []
    
    #bootstrap from raw trajectories
    for m, mig in rawtrans.groupby('Treatment'):            
        
        combolist = []
        for cidc, cell in mig.groupby('CellID'):
            #sort data and get continuous transitions in order
            cell = cell.sort_values('real_time').reset_index(drop = True)
            #resets in cumulative time represent a change between non-consecutive
            #series of interpolated transitions
            diff = cell.cumulative_time.diff()
            difflist = [0]
            difflist.extend(diff[diff<0].index.to_list())
            if difflist[-1] < len(cell):
                difflist.append(len(cell))
            #make a list of lists with the indices of consecutive time points
            runs = [list(range(difflist[x], difflist[x+1])) for x in range(len(difflist)-1)]
            
            for r in runs:
                limdf = cell.iloc[r]
                for i in range(len(limdf) - ntrans + 1):
                    combo = limdf.iloc[i:i + ntrans].copy()
                    combolist.append(combo)

        # Combine into a single DataFrame with MultiIndex
        combodf = pd.concat(combolist)

        #create multiindex for the overall dataframe
        miarray = [np.repeat(range(int(len(combodf)/ntrans)),ntrans), np.tile(list(range(ntrans)),int(len(combodf)/ntrans))]
        miindex = pd.MultiIndex.from_arrays(miarray, names = ['transition_combination','transition_index'])

        #add correct multiindex to the dataframe of combinations
        combodf.index = miindex
        
        #get list of tuples of arguments to pass to imap
        mapargs = [(combodf,ttot,ntrans, False) for _ in range(bsiter)]
        #boostrap with multiprocessing
        print(f'Boostrapping trajectories with {ntrans} transition samples for {m}')
        with multiprocessing.Pool(processes=60) as pool:
            results = list(tqdm.tqdm(pool.imap(bootstrap_trajectories, mapargs), total=bsiter))
        
        #get results
        migboot = pd.concat(results, ignore_index=True)
        migboot['iter'] = list(itertools.chain.from_iterable([[k]*len(res) for k,res in enumerate(results)]))
        migboot['Treatment'] = m
        #append to the larger list of dataframes
        bstrans.append(migboot)

        ###### now interpolate the bootstrapped trajectories ######
        print(f'Interpolating trajectories for {m}')
        results = []
        with multiprocessing.Pool(processes=60) as pool:
            for i, d in migboot.groupby('iter'):
                cell = d.sort_values('cumulative_time').reset_index(drop = True)
                result = pool.apply_async(interpolate_2dtrajectory, args = (
                    time_interval,
                    cell,
                    ))
                results.append(result)


            #get results
            results = [r.get() for r in results]
        bsinttrans = pd.DataFrame([x for r in results for x in r])
        bsinttrans['iter'] = list(itertools.chain.from_iterable([[k]*len(res) for k,res in enumerate(results)]))
        bsinttrans = bsinttrans.sort_values(by = ['CellID','cumulative_time']).reset_index(drop=True)
        bsinttrans['Treatment'] = m
        bsint.append(bsinttrans)


        ###### now get transition rates
        #get list of tuples of arguments to pass to imap
        mapargs = [(it, nbins, it.time_elapsed.sum()) for i, it in bsinttrans.groupby('iter')]
        #boostrap with multiprocessing
        print(f'Calculating bootstrapped CGPS transition rates for {m}')
        with multiprocessing.Pool(processes=60) as pool:
            results = list(tqdm.tqdm(pool.imap(transition_count_wrapper, mapargs), total=bsiter))


        #combine and add other info
        migrate = pd.concat(results, ignore_index=True)
        migrate['Treatment'] = m
        migrate['bs_iteration'] = np.repeat(np.arange(bsiter),nbins**2)
        bsframe_sep_full.append(migrate)
        

    ####### pull everything together and save
    bstrans = pd.concat(bstrans, ignore_index=True)
    bstrans.to_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_bootstrapped_{ntrans}_transitions.csv')
    bsint = pd.concat(bsint, ignore_index=True)
    bsint.to_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_bootstrapped_{ntrans}_interpolated_transitions.csv')
    bsframe_sep_full = pd.concat(bsframe_sep_full, ignore_index=True)
    bsframe_sep_full.to_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_bootstrapped_{ntrans}_transition_rates.csv')
    print('Finished bootstrapping')
    
    return bstrans, bsint, bsframe_sep_full
    

############# open average bootstrapped currents ###################
def get_avg_current_error(
        bsframe_sep_full, #transition rates in the cgps from get_bootstrapped_cgps_trajectories
        whichpcs, #which two PCs to use in the cgps [x,y]
        savedir, #where to save the aggregated counts
        nbins, #how many bins in the x and y cgps axes
        ntrans = 1, #how many transitions to sample at each step
        ):
    #### get current field for this bootstrap realization ######
    ####### this is for looking at data spread for the current field ############
    bsfield = []
    for m, mig in bsframe_sep_full.groupby('Treatment'):
        for x in range(nbins):
            for y in range(nbins):
                current = mig[(mig['x'] == x+1) & (mig['y'] == y+1)]
                js = np.array([[(row.x_plus_rate - row.x_minus_rate)/2,(row.y_plus_rate - row.y_minus_rate)/2] for i, row in current.iterrows()])
                js_centered = js - np.mean(js, axis = 0)
                avgjs = np.cov(js_centered.T)
                evals, evecs = np.linalg.eigh(avgjs)
                bsfield.append({'x':x+1,
                                'y':y+1,
                                'eval1':evals[1],
                                'eval2':evals[0],
                               'evec1x':evecs[0,1],
                               'evec1y':evecs[1,1],
                               'evec2x':evecs[0,0],
                               'evec2y':evecs[1,0],
                              'Treatment':m})

    bsfield_sep = pd.DataFrame(bsfield)
    bsfield_sep.to_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_bootstrapped_{ntrans}_transitions_average_currents.csv')
    
    return bsfield_sep


########## calculate all the aers and cycling frequencies from the bootstrapped data
def get_aer_cf(
        bstrans, #boostrapped transitions from get_bootstrapped_cgps_trajectories
        nbins, #how many bins in the x and y cgps axes
        xyscaling, #scaling of the bins in real units of whatever the CGPS axis parameters are
        center, #origin in [x bin,y bin]
        savedir, #where to save calculated aers and cfs
        whichpcs, #which two PCs to use in the cgps [x,y]
        ntrans = 1, #how many transitions to sample at each step
        ):


    #make list of imap arguments
    bsiter = bstrans.iter.max()+1
    mapargs = [(df.sort_values('cumulative_time').reset_index(drop = True),nbins,xyscaling,center) for i, df in bstrans.groupby(['Treatment','iter'])]

    with multiprocessing.Pool(processes=60) as pool:
        results = list(tqdm.tqdm(pool.imap(get_area_enclosing_rate, mapargs), total=bsiter))

    allaers = pd.concat(results, ignore_index=True)
    allaers.to_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_{ntrans}_transition_Area_Enclosing_Rates.csv')





######## from https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def filter_dataframe(df,
                     factor,
                     thresh = 0.05,
                     N = 20,
                     ):
    allcellsabv = []
    df = df.sort_values(by='frame').reset_index(drop=True)
    for i, cells in df.groupby('CellID'):
        cells, runs = utils.get_consecutive_timepoints(cells, 'frame', 1)
        for r in runs:
            #skip runs less than 3 frames long
            if len(r)<2:
                pass
            else:
                cell = cells.iloc[r]
                N=20
                #shrink the convolution window if the track isn't long enough
                if len(cell)<N:
                    N=round(len(cell)/3)
                ####### alternatively use:
                ####### con = np.convolve(np.nan_to_num(data.speed), np.ones(N)/N, mode='valid')
                con = running_mean(np.nan_to_num(cell[factor]),N)
                abvthresh = np.where(con>thresh)[0]
                if len(abvthresh)>0:
                    indtopull = abvthresh + (N-1)
                    if abvthresh[0] == 0:
                        indtopull = np.insert(indtopull, 0, range(N-1))
                    cellabv = cell.iloc[indtopull].copy()
                    allcellsabv.append(cellabv)

    return pd.concat(allcellsabv).reset_index(drop=True)