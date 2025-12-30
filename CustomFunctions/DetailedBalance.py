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
import random



def signed_angle(u,v):
    return math.degrees(math.atan2( u[0]*v[1] - u[1]*v[0], u[0]*v[0] + u[1]*v[1] ))

def clock_counterclock_angle(u,v):
    return -signed_angle(u,v)


## build a rectangle with diagonal corners
## this is for taking contour integrals around a specific flux path
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

## build a rectangle from upper left and lower right coordinates
## this is for taking contour integrals around a specific flux path
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
    #how many dimensions is the space
    dims = len(whichpcs)
    
    #### get coordinates
    alltrans = df[[f'PC{w}bins' for w in whichpcs]].copy()
    alltrans.columns = [f'from_{["x","y","z"][i]}' for i in range(dims)]
    #### get transitions
    alltrans[[f'to_{["x","y","z"][i]}' for i in range(dims)]] = alltrans.shift(-1)
    alltrans = alltrans.dropna()
    
    ##### add a bunch of other info
    #frame will reference the timepoint at the end of the transition
    alltrans['real_time'] = df[1:].time.values
    alltrans['frame'] = df[1:].frame.values
    #add the time elapsed in each transition (imaging interval)
    alltrans['time_elapsed'] = time_interval
    alltrans['cumulative_time'] = np.arange(time_interval, len(df)*time_interval, time_interval)
    #add cell identification
    alltrans['CellID'] = df.CellID.to_list()[:-1]
    # 'cell' will reference the cell/frame at the end of the transition
    alltrans['cell'] = df.cell.to_list()[1:]
    
    return alltrans




def interpolate_2dtrajectory(
        t_int, # time interval between frames in seconds
        rawtrans, # continuous-time dataframe with transitions sorted by frame # 
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
    interlist.append(np.array([[frames[-1], traj[-1,0], traj[-1,1], len(rawtrans)]]))
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



def interpolate_trajectory(
        t_int, # time interval between frames in seconds
        rawtrans, # continuous-time dataframe with transitions sorted by frame # 
        ):
    
    #how many dimensions is the space
    dims = len([x for x in rawtrans.columns.to_list() if 'from_' in x])
    
    ##get a list of movie frames for tracking time and frame identity
    frames = rawtrans.frame.to_list()
    
    #get the CGPS POSITIONS for this trajectory segment
    traj = np.vstack((rawtrans[[x for x in rawtrans.columns.to_list() if 'from_' in x]].values,
                      rawtrans[[x for x in rawtrans.columns.to_list() if 'to_' in x]].iloc[-1].values))
    
    #remove duplicate coordinates
    #which breaks the interpolation function
    #first make sure numpy array dtype is correct
    traj = traj.astype(np.float32)
    #find the indicies of the duplicates
    duplicates = [i for i,w in enumerate(traj) if all(w==traj[i-1])]
    #add a small number to the duplicates so they're not the same, but not meaningfully different
    for d in duplicates:
        traj[d,:] = traj[d,:]+0.001
    
    #interpolate based on path based on real time
    maxtime = len(traj)*t_int
    time_units = np.arange(0,maxtime, t_int)
    tck, b = interpolate.splprep(traj.T, u=time_units ,k=1, s=0)
    
    
    #start the transition list with a dummy transition that will be dropped later
    trans = [ [frames[0]] + list(traj[0]) + list(traj[0]) + [0,0] ]
    for t in range(len(traj)-1):
        #determine if there's a transition in this frame
        frame_to_frame_diff = traj[t+1]-traj[t]
        statechange = abs(frame_to_frame_diff).sum()
        #if there's a single bin change add the transition
        if statechange == 1:
            current_coord = traj[t+1]
            ###determine the current time
            ##round up to when this transition "started" 
            current_time = t*t_int + t_int/2 #single transitions always take t_int/2
            trans.append([frames[t]] + trans[-1][int(1+dims):int(1+2*dims)] + list(current_coord) + [current_time-trans[-1][-1], current_time])
        #if there's more than one bin position change during this frame, interpolate to find when it happens
        elif statechange>1:
            #measure the trajectory and interpolate evenly by distance
            di = np.sqrt(np.sum(frame_to_frame_diff**2))
            intt = round(di/0.01)
            #get interpolated coordinates
            interpoints = np.linspace(start=t*t_int, stop = (t+1)*t_int, num = intt, endpoint = False)
            splev_coords = interpolate.splev(interpoints,tck)
            interp_coords = np.round(splev_coords).T
            #get all the spatial differences between the interpolated coordinates
            interp_diffs = abs(np.diff(interp_coords, axis = 0))
            interp_diff_ind = np.where(np.sum(interp_diffs, axis = 1)>0)[0]
            
            #loop to find single transitions or deal with multi transitions
            for i in interp_diff_ind:
                #absolute value of transitions
                ai_d = interp_diffs[i]
                #update current time and position
                current_coord = interp_coords[i+1]
                current_time = interpoints[i]#round(interpoints[i]) if interpoints[i]%5-5>-0.01 else interpoints[i]
                # if i == interp_diff_ind[2]:
                #     break
                #collect all of the single moves
                if ai_d.sum() == 1:
                    trans.append([frames[t]] + trans[-1][int(1+dims):int(1+2*dims)] + list(current_coord) + [current_time-trans[-1][-1], current_time])
                
                elif ai_d.sum() > 1:
                    ### if there's STILL a transition by more than a single move
                    ### then it means the slope of the transition is 1 and needs to
                    ### have the transitions to adjacent boxes decided randomly
                    multi_cross = False
                    if ai_d.sum() == 3:
                        multi_cross = [0,1,2]
                    #check x and y first to allow for 2d cases
                    elif ai_d[0]>=1 and ai_d[1]>=1:
                        multi_cross = [0,1]
                    elif ai_d[0]>=1 and ai_d[2]>=1:
                        multi_cross = [0,2]
                    elif ai_d[1]>=1 and ai_d[2]>=1:
                        multi_cross = [1,2]
                    #### handle the diagonal border crossing
                    if multi_cross:
                        #randomize` transition order
                        shuffle(multi_cross)
                        ## define time elapsed in each interpolated step
                        te = (current_time-trans[-1][-1])/len(multi_cross) if len(multi_cross)!=dims else t_int/len(multi_cross)
                        for m, mc in enumerate(multi_cross):
                            #define cumulative time
                            ct = trans[-1][-1] + te
                            #get current coordinate and replace elements for each step of the "multi cross"
                            tempcur = trans[-1][int(1+dims):int(1+2*dims)]
                            tempcur[mc] = current_coord[mc]
                            trans.append([frames[t]] + trans[-1][int(1+dims):int(1+2*dims)] + tempcur + [te, ct])
    
        
    #drop the dummy first "transition"
    trans = trans[1:]
    #add the final transition
    
    #combine the data
    alltrans = pd.DataFrame(trans, columns=['frame'] +
                            [x for x in rawtrans.columns.to_list() if 'from_' in x] +
                            [x for x in rawtrans.columns.to_list() if 'to_' in x] +
                            ['time_elapsed','cumulative_time'])
    
    #add cell name
    alltrans['CellID'] = rawtrans.CellID.iloc[0]
    #also add the frame identifier just in case
    celllist = rawtrans.cell.to_list()
    alltrans['cell'] = [celllist[frames.index(x)] for x in alltrans.frame.to_list()]
    #add real image time so that data can be sorted even if it's not
    #from the same video
    alltrans['real_time'] = alltrans.cumulative_time + rawtrans.real_time.iloc[0]

    return alltrans


def get_transition_counts(
        coord, #array-like with coordinate (in xyz order) to count transitions in and out of
        bsdf, #dataframe with all transitions
        ttot, #total time represented by the experiment
        ):

    #get the number of dimensions in the CGPS from the coordinate
    dims = ['x','y','z'][:len(coord)]
    
    #get the dataframe with transitions FROM the coordinate of interest
    frombool = np.array([bsdf['from_'+dim] == coord[d] for d, dim in enumerate(dims)])
    frommask = np.where(np.all(frombool, axis = 0))
    fromm = bsdf.iloc[frommask]
    
    #get the dataframe with transitions TO the coordinate of interest
    tobool = np.array([bsdf['to_'+dim] == coord[d] for d, dim in enumerate(dims)])
    tomask = np.where(np.all(tobool, axis = 0))
    to = bsdf.iloc[tomask]
    
    #### iteritively build dict of transition counts and rates
    trans_count = {dims[i]:c for i, c in enumerate(coord)}
    for d, dim in enumerate(dims):
        #get the rate going over the - x side of the box (rate going left)
        minus_count_for = fromm[fromm['to_'+dim]<coord[d]].shape[0]
        minus_for_rate = minus_count_for/ttot
        minus_count_rev = to[to['from_'+dim]<coord[d]].shape[0]
        minus_rev_rate = minus_count_rev/ttot
        minus_rate = (minus_count_for - minus_count_rev)/ttot
    
        #get the rate going over the + x side of the box (rate going right)
        plus_count_for = fromm[fromm['to_'+dim]>coord[d]].shape[0]
        plus_for_rate = plus_count_for/ttot
        plus_count_rev = to[to['from_'+dim]>coord[d]].shape[0]
        plus_rev_rate = plus_count_rev/ttot
        plus_rate = (plus_count_for - plus_count_rev)/ttot
    
        #add all to dict
        trans_count.update({
            dim+'_minus_count':minus_count_for,
            dim+'_minus_count_rev':minus_count_rev,
            dim+'_minus_for_rate':minus_for_rate,
            dim+'_minus_rev_rate':minus_rev_rate,
            dim+'_minus_rate':minus_rate,
            dim+'_plus_count':plus_count_for,
            dim+'_plus_count_rev':plus_count_rev,
            dim+'_plus_for_rate':plus_for_rate,
            dim+'_plus_rev_rate':plus_rev_rate,
            dim+'_plus_rate':plus_rate
            })
    
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
    
    #get dims
    dims = [x.split('from_')[-1] for x in combodf.columns if 'from_' in x]
    
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
        cur = allbs.iloc[-1][['to_'+c for c in dims]].values
        #get all the transitions at the new position
        frombool = np.array([firsttrans['from_'+dim] == cur[d] for d, dim in enumerate(dims)])
        frommask = np.where(np.all(frombool, axis = 0))
        allat = firsttrans.iloc[frommask]
        
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
                    cur = allbs.iloc[-1][['to_'+c for c in dims]].values
                    #get all the transitions at the new position
                    frombool = np.array([firsttrans['from_'+dim] == cur[d] for d, dim in enumerate(dims)])
                    frommask = np.where(np.all(frombool, axis = 0))
                    allat = firsttrans.iloc[frommask]
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
                        cur = allbs.iloc[-1][['to_'+c for c in dims]].values
                        #get all the transitions at the new position
                        frombool = np.array([firsttrans['from_'+dim] == cur[d] for d, dim in enumerate(dims)])
                        frommask = np.where(np.all(frombool, axis = 0))
                        allat = firsttrans.iloc[frommask]
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
    
    ## determine dimensions in the CGPS
    dims = [x.split('from_')[-1] for x in bsdf.columns if 'from_' in x]
    ## build all the possible coordinates in the space
    axes = [np.arange(1,nbins+1)] * len(dims)
    grid = np.meshgrid(*axes)
    coords = np.stack(grid, axis=-1).reshape(-1, len(dims))
    
    ############## get the Boosttrapped counts of each bin position ############
    results = []
    for coord in coords:
        results.append(get_transition_counts(
            coord,
            bsdf,
            ct, #use the time actually observed during the simulation, especially important for simulations that terminate early
            ))

    bstrans_rate_df = pd.DataFrame(results)
    bstrans_rate_df = bstrans_rate_df.sort_values(by = dims).reset_index(drop=True)
    
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
    #center = 'center', or coordinates of origin
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
        group_factor = 'Treatment', #column with factor to separate the data on
        ):
    migresults = []
    for m, Mig in TotalFrame.groupby(group_factor):
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
        rawtrans[group_factor] = m
        migresults.append(rawtrans)
        
    rawtrans = pd.concat(migresults, ignore_index=True)
    rawtrans.to_csv(savedir.joinpath('-'.join(f"PC{w}" for w in whichpcs)+'_transitions_separated.csv'))

    print('Aggregated transitions')
    
    return rawtrans


def get_interpolated_cgps_trajectories(
        rawtrans, #pandas dataframe with raw transitions from get_raw_cgps_trajectories
        whichpcs, #which two PCs to use in the cgps [x,y]
        time_interval, #real time between datapoints
        savedir, #where to save the aggregated trajectories
        group_factor = 'Treatment', #column with factor to separate the data on
        ):
    migresults = []
    for m, Mig in rawtrans.groupby(group_factor):
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
                        result = pool.apply_async(interpolate_trajectory, args = (
                            time_interval,
                            cell,
                            ))
                        results.append(result)
    
            #get results
            results = [r.get() for r in results]
        #separate results into transtions and transition pairs
        transdf_sep = pd.concat(results)
        transdf_sep = transdf_sep.sort_values(by = ['CellID','real_time']).reset_index(drop=True)
        transdf_sep[group_factor] = m
        migresults.append(transdf_sep)

    transdf_sep = pd.concat(migresults)
    transdf_sep.to_csv(savedir.joinpath('-'.join(f"PC{w}" for w in whichpcs)+'_interpolated_transitions_separated.csv'))
    print('Finished interpolating trajectories')
    
    return transdf_sep
    
############## get the counts of cells leaving 
def aggregate_transition_counts(
        transdf_sep, #transdf_sep from get_interpolated_cgps_trajectories
        whichpcs, #which two PCs to use in the cgps [x,y]
        savedir, #where to save the aggregated counts
        nbins, #how many bins in the x and y cgps axes
        group_factor = 'Treatment', #column with factor to separate the data on
        ):
    trresults = []
    for m, mig in transdf_sep.groupby(group_factor):
        ## get time observed in this group
        ttot = mig.time_elapsed.sum()
        print(f'Total time observed in this CGPS was {ttot/60} minutes')
        trans_rate_df_sep = transition_count_wrapper((mig, nbins, ttot))
        ## add group_factor
        trans_rate_df_sep[group_factor] = m
        ## append to list of all group transition counts
        trresults.append(trans_rate_df_sep)

    trans_rate_df_sep = pd.concat(trresults)
    trans_rate_df_sep.to_csv(savedir.joinpath('-'.join(f"PC{w}" for w in whichpcs)+'_binned_transition_rates_separated.csv'))
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
        group_factor = 'Treatment', #column with factor to separate the data on
        ):
    
    #make a bunch of lists that I will append things to as I go for each treatment
    bstrans = []
    bsint = []
    bsframe_sep_full = []
    
    #bootstrap from raw trajectories
    for m, mig in rawtrans.groupby(group_factor):            
        
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

        #create multiindex for the overall dataframe (mostly applies for multiple transitions)
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
        bsinttrans = bsinttrans.sort_values(by = ['iter','cumulative_time']).reset_index(drop=True)
        bsinttrans[group_factor] = m
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
        migrate[group_factor] = m
        migrate['bs_iteration'] = np.repeat(np.arange(bsiter),nbins**2)
        bsframe_sep_full.append(migrate)
        

    ####### pull everything together and save
    bstrans = pd.concat(bstrans, ignore_index=True)
    bstrans.to_csv(savedir.joinpath('-'.join(f"PC{w}" for w in whichpcs)+'_bootstrapped_{ntrans}_transitions.csv'))
    bsint = pd.concat(bsint, ignore_index=True)
    bsint.to_csv(savedir.joinpath('-'.join(f"PC{w}" for w in whichpcs)+'_bootstrapped_{ntrans}_interpolated_transitions.csv'))
    bsframe_sep_full = pd.concat(bsframe_sep_full, ignore_index=True)
    bsframe_sep_full.to_csv(savedir.joinpath('-'.join(f"PC{w}" for w in whichpcs)+'_bootstrapped_{ntrans}_transition_rates.csv'))
    print('Finished bootstrapping')
    
    return bstrans, bsint, bsframe_sep_full
    

############# open average bootstrapped currents ###################
def get_avg_current_error(
        bsframe_sep_full, #transition rates in the cgps from get_bootstrapped_cgps_trajectories
        whichpcs, #which two PCs to use in the cgps [x,y]
        savedir, #where to save the aggregated counts
        nbins, #how many bins in the x and y cgps axes
        ntrans = 1, #how many transitions to sample at each step
        group_factor = 'Treatment', #column with factor to separate the data on
        ):
    #### get current field for this bootstrap realization ######
    ####### this is for looking at data spread for the current field ############
    bsfield = []
    for m, mig in bsframe_sep_full.groupby(group_factor):
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
                              group_factor:m})

    bsfield_sep = pd.DataFrame(bsfield)
    bsfield_sep.to_csv(savedir.joinpath('-'.join(f"PC{w}" for w in whichpcs)+'_bootstrapped_{ntrans}_transitions_average_currents.csv'))
    
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
        group_factor = 'Treatment', #column with factor to separate the data on
        ):


    #make list of imap arguments
    bsiter = bstrans.iter.max()+1
    mapargs = [(df.sort_values('cumulative_time').reset_index(drop = True),nbins,xyscaling,center) for i, df in bstrans.groupby([group_factor,'iter'])]

    with multiprocessing.Pool(processes=60) as pool:
        results = list(tqdm.tqdm(pool.imap(get_area_enclosing_rate, mapargs), total=bsiter))

    allaers = pd.concat(results, ignore_index=True)
    allaers.to_csv(savedir.joinpath('-'.join(f"PC{w}" for w in whichpcs)+'_{ntrans}_transition_Area_Enclosing_Rates.csv'))




########## measure gap frequency and duration
def get_gap_stats(
        df, #dataframe
        group, #what is the identifier to group by as a str
        time_interval, #frame rate of the data
        gapcol = 'aer', #column that I care about gaps in, will be dropped
        ):
    gaps = []
    gap_freqs = []
    for c, cell in df.groupby(group):
        #drop spots where there's no AER
        cell = cell[~cell[gapcol].isna()]
        cell = cell.sort_values('time').reset_index(drop = True)
        #find time differences between frames
        diffs = cell.time.diff()
        #find gaps larger than the imaging interval
        bigdiffs = diffs[diffs.abs()>time_interval].to_list()
        #measure the frequency of gaps in # / sec
        celldiff_freq = len(bigdiffs)/cell.time.max()
        
        #append the gap lengths and frequencies
        gaps.extend(bigdiffs)
        gap_freqs.append(celldiff_freq)
        
    ### array of observed gaps in #'s of frames
    gap_frame_num = np.round(np.array(gaps)/time_interval)
    
    ## average frequency of gaps in seconds
    gap_prob = np.mean(gap_freqs)
    # probability of gaps in # / frame
    gap_prob_frame = gap_prob*time_interval
    
    return gap_prob_frame, gap_frame_num




######### get dataframe of bootstrapped rows to drop to mimic LLS data gaps
def bootstrap_gaps(
    bsdf, #dataframe with bootstrap iterations (doesn't actually need aer)
    gap_prob, #gap probability NOTE: this won't necessarily match the gap frequency in the output
    gap_frame_num, #distribution of gap lengths in numbers of frames
    ):
    
    bs_gapped_list = []
    for i, it in bsdf.groupby('iter'):
        it = it.sort_values('real_time').reset_index(drop = True)
        ## loop through the bootstrap iteration and put in gaps with similar
        ## probability and duration to those in the real cells
        ftk = 0 ## frames to keep
        ftklist = []
        cur_prob = gap_prob ## current probability
        while ftk<len(it):
            if random.random()<cur_prob:
                ## pick a gap length from the distribution
                gp = random.choice(gap_frame_num)
                ftk = ftk+gp
                cur_prob = 0 #don't put gaps after gaps
            else:
                ftk = ftk+1
                cur_prob = gap_prob
            ftklist.append(ftk)
        
        ## drop the rows that are now gaps
        dropped = it.loc[np.array(ftklist[:-1]).astype(int)]
        bs_gapped_list.append(dropped)
        
    #combine into one dataframe    
    bs_gap_df = pd.concat(bs_gapped_list, ignore_index = True)
    #restrict it just to identifier info only
    identifiers = bs_gap_df[['iter','real_time']]

    return identifiers






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