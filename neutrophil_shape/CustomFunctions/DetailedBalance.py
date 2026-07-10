# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:52:46 2023

@author: Aaron
"""

from scipy import interpolate
import pandas as pd
import numpy as np
import random
from . import utils
import multiprocessing
import itertools
import math
import tqdm
from scipy.stats import gaussian_kde
from ..config.models import Config

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
    froms = [f'from_{["x","y","z"][i]}' for i in range(dims)]
    tos = [f'to_{["x","y","z"][i]}' for i in range(dims)]
    
    
    #### get coordinates
    alltrans = df[[f'PC{w}bins' for w in whichpcs]].copy().reset_index(drop = True)
    alltrans.columns = froms
    #### get transitions
    alltrans[tos] = alltrans.shift(-1)
    alltrans = alltrans.dropna()
    
    ##### add a bunch of other info
    #frame will reference the timepoint at the end of the transition
    alltrans['real_time'] = df[1:].time.values
    alltrans['frame'] = df[1:].frame.values
    #add the cumulative time based on the imaging interval 
    alltrans['cumulative_time'] = np.arange(time_interval, len(df)*time_interval, time_interval)
    #add cell identification
    alltrans['CellID'] = df.CellID.to_list()[:-1]
    
    #drop stalled "transitions" so that only true transitions are counted
    stallmask = (alltrans[froms].values == alltrans[tos].values).all(axis = 1)
    alltrans = alltrans[~stallmask]
    #if there's still transitions to write after dropping the stalls
    if not alltrans.empty:
        #now that stalls are dropped calculated the time elapsed for each transition
        alltrans['time_elapsed'] = alltrans.cumulative_time.diff()
        #fill the time_elapsed nan accounting for possible stalls in the first transition
        alltrans.at[alltrans.index[0],'time_elapsed'] = (alltrans.index[0] + 1) * time_interval
        
        return alltrans





def interpolate_trajectory(
        rawtrans, # continuous-time dataframe with transitions sorted by frame # 
        ):
    
    #reset index just in case
    rawtrans = rawtrans.reset_index(drop = True)
    
    #how many dimensions is the space
    dims = len([x for x in rawtrans.columns.to_list() if 'from_' in x])
    
    ##get a list of movie frames for tracking time and frame identity
    frames = rawtrans.frame.to_list()
    
    #get the CGPS POSITIONS for this trajectory segment
    traj = np.vstack((rawtrans[[x for x in rawtrans.columns.to_list() if 'from_' in x]].values,
                      rawtrans[[x for x in rawtrans.columns.to_list() if 'to_' in x]].iloc[-1].values))
    
    #interpolate based on path based on real time
    time_units = rawtrans.time_elapsed.cumsum().values
    time_units = np.insert(time_units, 0,0)
    tck, b = interpolate.splprep(traj.T.astype(float), u=time_units.astype(float), k=1, s=0)
    
    
    #start the transition list with a dummy transition that will be dropped later
    trans = [ [frames[0]] + list(traj[0]) + list(traj[0]) + [0,0] ]
    for t in range(len(traj)-1):
        if t == 35:
            break
        #determine if there's a transition in this frame
        frame_to_frame_diff = abs(traj[t+1]-traj[t])
        statechange = frame_to_frame_diff.sum()
        #if there's a single bin change add the transition
        if statechange == 1:
            current_coord = traj[t+1]
            ###determine the current time
            ##round up to when this transition "started" 
            current_time = time_units[t:t+2].mean() #single transitions take half the time since the last transition
            trans.append([frames[t]] + trans[-1][int(1+dims):int(1+2*dims)] + list(current_coord) + [current_time-trans[-1][-1], current_time])
        #manually handle direct diagonal transitions because they interpolate weirdly
        elif np.all(frame_to_frame_diff == frame_to_frame_diff[0]):
            
            #how many diagonal crossing are there
            diag_num = int(frame_to_frame_diff[0])
            #what's the direction of single diagonal transitions
            trans_template = (traj[t+1]-traj[t])/diag_num
            #total transition time
            diag_trans_time_total = np.diff(time_units[t:t+2])[0]
            ## define time elapsed in each interpolated step
            te = (diag_trans_time_total/diag_num) / len(trans_template)
            # print('diagonal', diag_num)
            
            #loop random transition selection for each diagonal cross
            for d in range(diag_num):
                #get the current coordinate and time
                current_coord = traj[t] + trans_template * (d + 1)
                current_time = time_units[t] + (diag_trans_time_total/diag_num) * (d+1)
                #get the randomized transition list
                multi_cross = list(range(len(trans_template)))
                random.shuffle(multi_cross)
                #make a temporary coordinate to update as transitions happen randomly
                tempcur = trans[-1][int(1+dims):int(1+2*dims)]
                for m, mc in enumerate(multi_cross):
                    #define cumulative time, including "remaining" time from the previous frame's transitions
                    ct = trans[-1][-1] + te + (time_units[t]-trans[-1][-1]) if (d==0) and (m==0) else trans[-1][-1] + te
                    #get current coordinate and replace elements for each step of the "multi cross"
                    tempcur[mc] = current_coord[mc]
                    time_elapsed = ct - trans[-1][-1]
                    trans.append([frames[t]] + trans[-1][int(1+dims):int(1+2*dims)] + tempcur + [time_elapsed, round(ct, 10)])
                    
        #if there's more than one bin position change during this frame, interpolate to find when it happens
        elif statechange>1:
            #measure the trajectory and interpolate evenly by distance
            di = np.sqrt(np.sum(frame_to_frame_diff**2))
            intt = round(di/0.001)
            #get interpolated coordinates
            interpoints = np.linspace(start=time_units[t], stop = time_units[t+1], num = intt, endpoint = False)
            splev_coords = interpolate.splev(interpoints,tck)
            interp_coords = np.round(splev_coords).T
            #get all the spatial differences between the interpolated coordinates
            interp_diffs = abs(np.diff(interp_coords, axis = 0))
            interp_diff_ind = np.where(np.sum(interp_diffs, axis = 1)>0)[0]
            
            #loop to find single transitions or deal with multi transitions
            for i, idi in enumerate(interp_diff_ind):
                #absolute value of transitions
                ai_d = interp_diffs[idi]
                #update current time and position
                current_coord = interp_coords[idi+1]
                current_time = interpoints[idi]#round(interpoints[i]) if interpoints[i]%5-5>-0.01 else interpoints[i]
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
                        # print('triple cross', interp_coords[interp_diff_ind])
                    #check x and y first to allow for 2d cases
                    elif ai_d[0]>=1 and ai_d[1]>=1:
                        multi_cross = [0,1]
                    elif ai_d[0]>=1 and ai_d[2]>=1:
                        multi_cross = [0,2]
                    elif ai_d[1]>=1 and ai_d[2]>=1:
                        multi_cross = [1,2]
                    #### handle the diagonal border crossing
                    if multi_cross:
                        #randomize transition order
                        random.shuffle(multi_cross)
                        ## define time elapsed in each interpolated step
                        te = (current_time-time_units[t])/len(multi_cross) if i == 0 else (current_time-trans[-1][-1])/len(multi_cross)
                        #make a temporary coordinate to update as transitions happen randomly
                        tempcur = trans[-1][int(1+dims):int(1+2*dims)]
                        for m, mc in enumerate(multi_cross):
                            #define cumulative time, including "remaining" time from the previous frame's transitions
                            ct = trans[-1][-1] + te + (time_units[t]-trans[-1][-1]) if (i==0) and (m==0) else trans[-1][-1] + te
                            #get current coordinate and replace elements for each step of the "multi cross"
                            tempcur[mc] = current_coord[mc]
                            time_elapsed = ct - trans[-1][-1]
                            trans.append([frames[t]] + trans[-1][int(1+dims):int(1+2*dims)] + tempcur + [time_elapsed, round(ct, 10)])
    
    
    #drop the dummy first "transition"
    trans = trans[1:]
    
    #convert to dataframe and name columns
    alltrans = pd.DataFrame(trans, columns=['frame'] +
                            [x for x in rawtrans.columns.to_list() if 'from_' in x] +
                            [x for x in rawtrans.columns.to_list() if 'to_' in x] +
                            ['time_elapsed','cumulative_time'])
    #add real image time so that data can be sorted even if it's not
    #from the same video
    alltrans['real_time'] = alltrans.cumulative_time + rawtrans.real_time.iloc[0] - rawtrans.time_elapsed.iloc[0]
    #add cell name
    alltrans['CellID'] = rawtrans.CellID.iloc[0]
    
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




def bootstrap_trajectory(
        imap_args
        ):
        
    #unpack args
    # combodf: multi-indexed dataframe with tansition_combination and trandition_index names
    # ttot: int total time for the simulation
    # avoiddead: bool whether or not to avoid dead ends in the trajectory
    combodf,ttot,avoiddead = imap_args

    #get dims
    dims = [x.split('from_')[-1] for x in combodf.columns if 'from_' in x]

    #get just the first transition of each combination
    firsttrans = combodf.xs(0,level='transition_index')

    #create an empty dataframe with the correct columns and indexing
    allbs = []
    #find the first random position
    rando = combodf.index.levels[0].to_list()
    random_choice = random.choice(rando)
    pick = combodf.loc[combodf.index.get_level_values('transition_combination') == random_choice]
    allbs.append(pick)
    ## start time with first pick
    ct = pick.time_elapsed.sum()
    while ct<ttot:
        #find the next postition after the second transition
        cur = allbs[-1][['to_'+c for c in dims]].values[-1]
        #get the indices of all the transitions at the new position
        frombool = np.array([firsttrans['from_'+dim] == cur[d] for d, dim in enumerate(dims)])
        allat = np.where(np.all(frombool, axis = 0))[0]

        #if the next transition doesn't have any future transitions, don't go there and pick a new one
        if len(allat) == 0:
            if avoiddead:
                #drop the "dead" transition
                allbs = allbs[:-1]
                #check is this happened at the beginning of the simulation and it needs to be started again 
                #from another position, otherwise trim the last transition and continue
                if len(allbs)==0:
                    allbs = []
                    #find the first random position
                    rando = combodf.index.levels[0].to_list()
                    random_choice = random.choice(rando)
                    pick = combodf.loc[combodf.index.get_level_values('transition_combination') == random_choice]
                    #restart time
                    ct = pick.time_elapsed.sum()
                    #add the random pick to the dataframe
                    allbs.append(pick)
                #subtract the time these transitions take
                ct = ct - pick.time_elapsed.sum()
                #set a timer for extreme cases of single transitions to deadends
                loops = 0
                while len(allat) == 0:
                    #find the next postition after the second transition
                    cur = allbs[-1][['to_'+c for c in dims]].values[-1]
                    #get all the transitions at the new position
                    frombool = np.array([firsttrans['from_'+dim] == cur[d] for d, dim in enumerate(dims)])
                    allat = np.where(np.all(frombool, axis = 0))[0]

                    #randomly select a transition pair
                    random_choice = random.choice(allat)
                    pick = combodf.loc[combodf.index.get_level_values('transition_combination') == random_choice]
                    #add to the timer for extreme cases
                    loops = loops + 1
                    #if the current position only has one transition (to the empty position)
                    #then trim it back an additional transition as well
                    #or if this while loop has gone for 20 iterations and still not found a suitable transition
                    #back up an additional transition
                    if (len(allat)==1) or (loops == 20):
                        #subtract the time these transitions take
                        print('subtracting time for deadend')
                        ct = ct - allbs[-1].time_elapsed.sum()
                        #delete a further two transitions
                        allbs = allbs[:-1]
                        #check if this happened at the beginning of the simulation and it needs to be started again 
                        #from another position, otherwise trim the last transition and continue
                        if len(allbs)==0:
                            allbs = []
                            #find the first random position
                            rando = combodf.index.levels[0].to_list()
                            random_choice = random.choice(rando)
                            pick = combodf.loc[combodf.index.get_level_values('transition_combination') == random_choice]
                            #restart time
                            ct = pick.time_elapsed.sum()
                            #add the random pick to the dataframe
                            allbs.append(pick)
                        #find the next postition after the second transition
                        cur = allbs[-1][['to_'+c for c in dims]].values[-1]
                        #get all the transitions at the new position
                        frombool = np.array([firsttrans['from_'+dim] == cur[d] for d, dim in enumerate(dims)])
                        allat = np.where(np.all(frombool, axis = 0))[0]

                        #randomly select a transition pair
                        random_choice = random.choice(allat)
                        pick = combodf.loc[combodf.index.get_level_values('transition_combination') == random_choice]
                #append the pair of transitions to a list
                allbs.append(pick)
                #add the time these transitions take
                ct = ct + pick.time_elapsed.sum()
            else:
                break
        else:
            #randomly select a transition pair
            random_choice = random.choice(allat)
            pick = combodf.loc[combodf.index.get_level_values('transition_combination') == random_choice]
            #append the pair of transitions to a list
            allbs.append(pick)
            #add the time these transitions take
            ct = ct + pick.time_elapsed.sum()
    #convert allbs list to a dataframe
    allbs = pd.concat(allbs, ignore_index=True)
    #make cumulative time actually cumulative time
    allbs.loc[:,'cumulative_time'] = allbs['time_elapsed'].cumsum()
    #make a mock "real_time" so that simulated dataframes match real ones
    allbs.loc[:,'real_time'] = allbs.cumulative_time

    return allbs


def bootstrap_trajectory_wrapper(_):
    return bootstrap_trajectory((shared_combodf, shared_ttot, shared_avoiddead))

def bs_init_worker(combodf, ttot, avoiddead):
    global shared_combodf, shared_ttot, shared_avoiddead
    shared_combodf = combodf
    shared_ttot = ttot
    shared_avoiddead = avoiddead


def transition_count_wrapper(
        args # tuple of arguments
        ):
    #unpack args from imap
    #bsdf: transition dataframe from bootstrap_trajectory()
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
    cell, nbins, xyscaling, origin = args
    
    #get values to shift coordinates to the origin of the current
    shiftbyx = origin[0]
    shiftbyy = origin[1]

    #calculate aer per transition
    aerlist = []
    avlist = []
    pcspeedlist = []
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

        ### also calculate "pc_speed"
        pcspeedlist.append(
            np.sqrt((row.to_x - row.from_x)**2 + (row.to_y - row.from_y)**2) / row.time_elapsed
        )

    cell['aer'] = aerlist
    cell['angular_velocity'] = avlist
    cell['pc_speed'] = pcspeedlist
    return cell



def rate_fit_bs_wrap(
        args
        ):
    
    ### unpack args
    # df, #dataframe containing "iter" bootstrap iteration ID, some group_factor, and "aer"
    # group_factor,
    # time_interval, #time interval of the imaging data
    df, group_factor = args
    
    ## create a dict with ID info
    id_dict = {
        group_factor: df.iloc[0][group_factor],
        'iter': df.iloc[0].iter,
        }
    ## fit rate
    rate_fit_dict = utils.fit_rates_linear(df, ['aer','angular_velocity','pc_speed'])
    ## update dict
    id_dict.update(rate_fit_dict)
    return id_dict


def get_raw_cgps_trajectories(
        TotalFrame, #pandas dataframe with all of the cgps binned data
        whichpcs, #which two PCs to use in the cgps [x,y]
        config: Config,
        group_factor: str = 'Treatment', #column with factor to separate the data on
        ):
    ## get settings from config
    time_interval = config.im_params.time_interval
    dbsavedir = config.common.savedir / 'detailed_balance'
    if not dbsavedir.exists():
        dbsavedir.mkdir()

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
    rawtrans.to_csv(dbsavedir.joinpath(utils.whichpc_string(whichpcs)+'_transitions_separated.csv'))

    print('Aggregated transitions')
    
    return rawtrans


def get_interpolated_cgps_trajectories(
        rawtrans, #pandas dataframe with raw transitions from get_raw_cgps_trajectories
        whichpcs, #which two PCs to use in the cgps [x,y]
        config: Config,
        group_factor: str = 'Treatment', #column with factor to separate the data on
        ):
    
    ## get settings from config
    dbsavedir = config.common.savedir / 'detailed_balance'
    
    migresults = []
    for m, Mig in rawtrans.groupby(group_factor):
        mapargs = []
        for i, cell in Mig.groupby('CellID'):
            cell, runs = utils.get_consecutive_transitions(cell)
            for r in runs:
                #skip runs less than 2 frames long
                if len(r)>1:
                    mapargs.append(cell.iloc[r])

        with multiprocessing.Pool(processes=60) as pool:
            results = list(pool.imap(interpolate_trajectory, mapargs))

        #separate results into transtions and transition pairs
        transdf_sep = pd.concat(results)
        transdf_sep = transdf_sep.sort_values(by = ['CellID','real_time']).reset_index(drop=True)
        transdf_sep[group_factor] = m
        migresults.append(transdf_sep)

    transdf_sep = pd.concat(migresults)
    transdf_sep.to_csv(dbsavedir.joinpath(utils.whichpc_string(whichpcs)+'_interpolated_transitions_separated.csv'))
    print('Finished interpolating trajectories')
    
    return transdf_sep
    
############## get the counts of cells leaving 
def aggregate_transition_counts(
        transdf_sep, #transdf_sep from get_interpolated_cgps_trajectories
        whichpcs, #which two PCs to use in the cgps [x,y]
        config: Config,
        group_factor: str = 'Treatment', #column with factor to separate the data on
        ):
    
    ## get settings from config
    nbins = config.db_params.nbins
    dbsavedir = config.common.savedir / 'detailed_balance'
    
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
    trans_rate_df_sep.to_csv(dbsavedir.joinpath(utils.whichpc_string(whichpcs)+'_binned_transition_rates_separated.csv'))
    print('Finished finding transition rates')
    
    return trans_rate_df_sep



############## BOOTSTRAP MANY TRAJECTORIES ##########
def get_bootstrapped_cgps_trajectories(
        rawtrans, #raw transitions from get_raw_cgps_trajectories
        whichpcs, #which two PCs to use in the cgps [x,y]
        config: Config,
        bssavedir: str, #where to save the bootstrapped dataframes
        group_factor: str = 'Treatment', #column with factor to separate the data on
        ):
    
    ### get some settings from config
    dbbssavedir = config.common.savedir / 'detailed_balance' / bssavedir #where to save the aggregated counts
    if not dbbssavedir.exists():
        dbbssavedir.mkdir()
    nbins = config.db_params.nbins #how many bins in the x and y cgps axes
    ttot = config.db_params.ttot #set the total bootstrap time
    ntrans = config.db_params.ntrans #how many transitions to sample at each step
    bsiter = config.db_params.bsiter #number of times to bootstrap


    #make a bunch of lists that I will append things to as I go for each treatment
    bstrans = []
    bsint = []
    bsframe_sep_full = []
    
    #bootstrap from raw trajectories
    for m, mig in rawtrans.groupby(group_factor):            
        if ntrans == 1:
            combodf = mig.copy()
        else:
            combolist = []
            for cidc, cell in mig.groupby('CellID'):
                #sort data and get continuous transitions in order
                cell, runs = utils.get_consecutive_transitions(cell)
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
        
        # #get list of tuples of arguments to pass to imap
        # mapargs = [(combodf,ttot,False) for _ in range(bsiter)]
        # #boostrap with multiprocessing
        # print(f'Boostrapping trajectories with {ntrans} transition samples for {m}')
        # with multiprocessing.Pool(processes=60) as pool:
        #     results = list(tqdm.tqdm(pool.imap(bootstrap_trajectory, mapargs), total=bsiter))
        
        with multiprocessing.Pool(processes=60,
                                  initializer=bs_init_worker,
                                  initargs=(combodf, ttot, False)) as pool:
            results = list(tqdm.tqdm(pool.imap_unordered(bootstrap_trajectory_wrapper, range(bsiter)), total=bsiter))

        #get results
        migboot = pd.concat(results, ignore_index=True)
        migboot['iter'] = list(itertools.chain.from_iterable([[k]*len(res) for k,res in enumerate(results)]))
        #append to the larger list of dataframes
        bstrans.append(migboot)

        ###### now interpolate the bootstrapped trajectories ######
        print(f'Interpolating trajectories for {m}')
        mapargs = [d for i, d in migboot.groupby('iter')]
        with multiprocessing.Pool(processes=60) as pool:
            results = list(tqdm.tqdm(pool.imap(interpolate_trajectory, mapargs), total=bsiter))
                    
        bsinttrans = pd.concat(results, ignore_index=True)
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
    bstrans.to_csv(dbbssavedir.joinpath(utils.whichpc_string(whichpcs)+f'_bootstrapped_{ntrans}_transitions.csv'))
    bsint = pd.concat(bsint, ignore_index=True)
    bsint.to_csv(dbbssavedir.joinpath(utils.whichpc_string(whichpcs)+f'_bootstrapped_{ntrans}_interpolated_transitions.csv'))
    bsframe_sep_full = pd.concat(bsframe_sep_full, ignore_index=True)
    bsframe_sep_full.to_csv(dbbssavedir.joinpath(utils.whichpc_string(whichpcs)+f'_bootstrapped_{ntrans}_transition_rates.csv'))
    print('Finished bootstrapping')
    
    return bstrans, bsint, bsframe_sep_full
    

############# open average bootstrapped currents ###################
def get_avg_current_error(
        bsframe_sep_full, #transition rates in the cgps from get_bootstrapped_cgps_trajectories
        whichpcs, #which two PCs to use in the cgps [x,y]
        config: Config,
        bssavedir: str, #where to save the bootstrapped dataframes
        group_factor: str = 'Treatment', #column with factor to separate the data on
        ):
    ### get some settings from config
    dbbssavedir = config.common.savedir / 'detailed_balance' / bssavedir #where to save the aggregated counts
    nbins = config.db_params.nbins #how many bins in the x and y cgps axes
    ntrans = config.db_params.ntrans #how many transitions to sample at each step

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
    bsfield_sep.to_csv(dbbssavedir.joinpath(utils.whichpc_string(whichpcs)+f'_bootstrapped_{ntrans}_transitions_average_currents.csv'))
    
    return bsfield_sep


########## calculate all the aers and cycling frequencies from the bootstrapped data
def get_aer_cf(
        bstrans, #boostrapped transitions from get_bootstrapped_cgps_trajectories
        whichpcs, #which two PCs to use in the cgps [x,y]
        config: Config,
        bssavedir: str, #where to save the bootstrapped dataframes
        group_factor: str = 'Treatment', #column with factor to separate the data on
        ):
    
    ### get some settings from config
    time_interval = config.im_params.time_interval
    savedir = config.common.savedir
    dbbssavedir = savedir / 'detailed_balance' / bssavedir #where to save the aggregated counts
    nbins = config.db_params.nbins #how many bins in the x and y cgps axes
    ntrans = config.db_params.ntrans #how many transitions to sample at each step
    bsiter = config.db_params.bsiter #number of times to bootstrap
    pc_combos = config.common.pc_combos #unique PC pairs
    origins = config.db_params.origins #flux origins for this dataset and alignment
    origin = origins[pc_combos.index(whichpcs)]

    ## open the CGPS bins to get scaling
    datadir = savedir / 'shape_data'
    centers = pd.read_csv(datadir.joinpath('PC_bin_centers.csv'), index_col=0)
    #scaling of the bins in real units of whatever the CGPS axis parameters are
    xyscaling = [centers[f'PC{wpc}'].diff().mean() for wpc in whichpcs]


    #make list of imap arguments
    mapargs = [(df.sort_values('cumulative_time').reset_index(drop = True),nbins,xyscaling,origin) for i, df in bstrans.groupby([group_factor,'iter'])]

    with multiprocessing.Pool(processes=60) as pool:
        results = list(tqdm.tqdm(pool.imap(get_area_enclosing_rate, mapargs), total=bsiter))

    allaers = pd.concat(results, ignore_index=True)
    allaers.to_csv(dbbssavedir.joinpath(utils.whichpc_string(whichpcs)+f'_bootstrapped_{ntrans}_Area_Enclosing_Rates.csv'))
    
    ### get AE rate and fit 
    lrmapargs = [(df.sort_values('cumulative_time').reset_index(drop = True),group_factor) for i, df in allaers.groupby([group_factor,'iter'])]
    with multiprocessing.Pool(processes=60) as pool:
        lrresults = list(tqdm.tqdm(pool.imap(rate_fit_bs_wrap, lrmapargs), total=bsiter))

    ## save the AE rate and fit
    pd.DataFrame(lrresults).to_csv(dbbssavedir.joinpath(utils.whichpc_string(whichpcs)+f'_bootstrapped_{ntrans}_Area_Enclosed_Linear_Reg.csv'))


def get_run_stats(
        df, #dataframe containing aer info
        group, #what is the identifier to group by as a str
        config: Config, #what was the imaging interval for this data
        ):
    ### get settings from config
    time_interval = config.im_params.time_interval

    allrunlengths = []
    allrunlengthmeans = []
    allgaplengths = []
    allgaplengthmeans = []
    allgapfrequencies = []
    for c, cell in df.groupby(group):
        ### drop aer nans just in case
        cell = cell[~cell.aer.isna()].copy()
        cell, runs = utils.get_consecutive_transitions(cell)
        run_lengths = [len(r) for r in runs]
        ##gap indexes
        gapinds = [r[0] for r in runs[1:]]
        ##gap lengths
        gap_lengths = np.array([cell.real_time.iloc[i] - cell.real_time.iloc[i-1] for i in gapinds], dtype = float)
        #gap_lenths units from # of seconds to # of frames
        gap_lengths /= time_interval
        #average run length for this cell
        meanrunlength = np.mean(run_lengths)
        #average gap length for this cell
        meangaplength = np.mean(gap_lengths)
        #frequency of gaps for this cell in number of gaps
        #per total time observed
        meangapfreq = len(gap_lengths)/cell.time_elapsed.sum()
        
        
        allrunlengths.extend(run_lengths)
        allrunlengthmeans.append(meanrunlength)
        allgaplengths.extend(gap_lengths)
        allgaplengthmeans.append(meangaplength)
        allgapfrequencies.append(meangapfreq)
    return allrunlengths, allrunlengthmeans, allgaplengths, allgaplengthmeans, allgapfrequencies





######### get dataframe of bootstrapped rows to drop to mimic LLS data gaps
def bootstrap_runs(
    bsdf, #dataframe with bootstrap iterations (doesn't actually need aer)
    allrunlengths, #the sample of movies lengths in seconds
    allgaplengths, #the sample of non-movie gap lengths in seconds
    ):

    ### get the kde's of movie_lengths and non_movie_gaps
    run_length_kde = gaussian_kde(allrunlengths)
    gap_length_kde = gaussian_kde(allgaplengths)

    bs_gapped_list = []
    for i, it in bsdf.groupby('iter'):
        it = it.sort_values('real_time').reset_index(drop = True)
        ## loop through the bootstrap iteration and put in gaps with similar
        ## probability and duration to those in the real cells
        current_frame = 0 ## frames to keep
        ftklist = []
        while current_frame<len(it):
            
            ### sample a movie length to use (in number of frames)
            current_run = round(run_length_kde.resample(1)[0][0])
            ### ensure that the movie length is positive since the KDE is continuous over zero
            while current_run<1:
                current_run = round(run_length_kde.resample(1)[0][0])

            ftklist.append(np.arange(current_frame, current_frame + current_run))
            
            ### sample a movie length to use (in number of frames)
            current_gap = round(gap_length_kde.resample(1)[0][0])
            ### ensure that the movie length is positive since the KDE is continuous over zero
            while current_gap<1:
                current_gap = round(gap_length_kde.resample(1)[0][0])

            current_frame = ftklist[-1][-1] + current_gap


        ### movie while loop will result in bootstraps going long
        ### so only get frames that actually exist
        ftkarray = np.concatenate(ftklist)
        ftkmask = ftkarray[ftkarray<len(it)]
        ## drop the rows that are now gaps
        dropped = it.loc[ftkmask]
        bs_gapped_list.append(dropped)
        
    #combine into one dataframe    
    bs_gap_df = pd.concat(bs_gapped_list, ignore_index = True)
    #restrict it just to identifier info only
    identifiers = bs_gap_df[['iter','real_time']]

    return identifiers




def get_lls_gapped_bootstrap(
    whichpcs: tuple, #which two PCs to use (x,y)
    config: Config,
    ):

    #get constants from config
    ntrans = config.db_params.ntrans #how many transitions to sample at each step

    ## get directories from config
    savedir = config.common.savedir
    dbdir = savedir / 'detailed_balance'
    dbbsdir = dbdir / 'separatedatabs'

    justaers = pd.read_csv(dbdir.joinpath(utils.whichpc_string(whichpcs)+'_raw_transition_aer_cf.csv'), index_col = 0)

    ########## measure gap frequency and duration
    allrunlengths, allrunlengthmeans, allgaplengths, allgaplengthmeans, allgapfrequencies = get_run_stats(
            justaers, #dataframe
            'CellID', #what is the identifier to group by as a str
            config, #frame rate of the data
            )
    print(f'Average track run length mean for real data is {np.mean(allrunlengthmeans)} and mean gap frequency is {np.mean(allgapfrequencies)})')

    #### get bs data with gaps
    bsaers = pd.read_csv(dbbsdir.joinpath(utils.whichpc_string(whichpcs)+f'_bootstrapped_{ntrans}_Area_Enclosing_Rates.csv'), index_col=0)
    bstrans = pd.read_csv(dbbsdir.joinpath(utils.whichpc_string(whichpcs)+f'_bootstrapped_{ntrans}_transitions.csv'), index_col = 0)

    bs_with_gaps = bootstrap_runs(
        bsaers, #dataframe with bootstrap iterations (doesn't actually need aer)
        allrunlengths, #the sample of movies lengths in seconds
        allgaplengths, #the sample of non-movie gap lengths in seconds
        )

    ### measure the gap probability in the newly gapped bootstrap data
    #change real_time to just time
    bs_gap_measure = bs_with_gaps.merge(bsaers[['iter','real_time','cumulative_time','time_elapsed']], on = ['iter','real_time'], how = 'left')
    #add dummy column
    bs_gap_measure['aer'] = 0
    bsallrunlengths, bsallrunlengthmeans, bsallgaplengths, bsallgaplengthmeans, bsallgapfrequencies = get_run_stats(
            bs_gap_measure, #dataframe
            'iter', #what is the identifier to group by as a str
            config, #frame rate of the data
            )

    print(f'Average track run length mean for bootstrapped data is {np.mean(bsallrunlengthmeans)} and mean gap frequency is {np.mean(bsallgapfrequencies)})')

    ### save the gapped bootstrap data
    bs_with_gaps.to_csv(dbbsdir.joinpath(utils.whichpc_string(whichpcs)+f'_bootstrapped_{ntrans}_Area_Enclosing_Rates_gaps.csv'))


    ### merged gaps with actual bootstrapped aers so we can do linear regression with the gapped data
    aers_with_gaps = bs_with_gaps.merge(bsaers, on = ['iter','real_time'], how = 'left')

    ### get AE rate and fit 
    lrmapargs = [(df.sort_values('cumulative_time').reset_index(drop = True),'Treatment') for i, df in aers_with_gaps.groupby(['Treatment','iter'])]
    with multiprocessing.Pool(processes=60) as pool:
        lrresults = list(tqdm.tqdm(pool.imap(rate_fit_bs_wrap, lrmapargs), total=config.db_params.bsiter))

    ## save the AE rate and fit
    fitframe = pd.DataFrame(lrresults)
    fitframe.to_csv(dbbsdir.joinpath(utils.whichpc_string(whichpcs)+f'_bootstrapped_{ntrans}_Area_Enclosed_Linear_Reg_gaps.csv'))
