# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:52:46 2023

@author: Aaron
"""

from scipy import interpolate
import pandas as pd
import numpy as np
from pathlib import Path
import random
from . import utils
import multiprocessing
import itertools
import math
import tqdm
from scipy.stats import gaussian_kde
from ..config.models import Config
from .utils import whichpc_string
def signed_angle(u,v):
    return math.degrees(math.atan2( u[0]*v[1] - u[1]*v[0], u[0]*v[0] + u[1]*v[1] ))

def clock_counterclock_angle(u,v):
    return -signed_angle(u,v)




_DIM_LABELS = ('x', 'y', 'z')

def raw_transitions(
        time_interval, # time interval between frames in seconds
        df, # pandas dataframe with cell, CellID, frame, and binned PCs
        whichpcs, #which pc #s are in the cgps in [x,y]
        ):
    #how many dimensions is the space
    dims = len(whichpcs)
    froms = [f'from_{_DIM_LABELS[i]}' for i in range(dims)]
    tos = [f'to_{_DIM_LABELS[i]}' for i in range(dims)]
    
    
    #### get coordinates
    wpc_list = [f'PC{w}bins' if w>0 else f'PC{abs(w)}bins_abs' for w in whichpcs]
    alltrans = df[wpc_list].copy().reset_index(drop = True)
    alltrans.columns = froms
    #### get transitions
    alltrans[tos] = alltrans.shift(-1)
    alltrans = alltrans.dropna()
    
    ##### add a bunch of other info
    #frame will reference the timepoint at the end of the transition
    alltrans['real_time'] = df.time.to_numpy()[1:]
    alltrans['frame'] = df.frame.to_numpy()[1:]
    #add the cumulative time based on the imaging interval 
    alltrans['cumulative_time'] = np.arange(time_interval, len(df)*time_interval, time_interval)
    #add cell identification
    #'cell' is unique per frame in the source data (unlike CellID/Treatment,
    #which are constant for the whole run), so it needs a per-transition value
    #aligned with each transition's ending frame, same as 'frame'/'real_time' above
    alltrans['cell'] = df.cell.to_numpy()[1:]
    alltrans['CellID'] = df.CellID.iloc[0]
    alltrans['Treatment'] = df.Treatment.iloc[0]
    
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


def raw_transitions_wrapper(args):
    return raw_transitions(*args)


def interpolate_trajectory(
        rawtrans, # continuous-time dataframe with transitions sorted by frame # 
        time_interval, # time between frames of the data
        ):
    
    #reset index just in case
    rawtrans = rawtrans.reset_index(drop = True)
    
    #how many dimensions is the space
    dims = len([x for x in rawtrans.columns.to_list() if 'from_' in x])
    
    ##get a list of movie frames for tracking time and frame identity
    frames = rawtrans.frame.to_list()

    ### the mimimal time_elapsed for a given transition is the 
    
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
            ## get interpolated coordinates
            interpoints = np.linspace(start=time_units[t], stop = time_units[t+1], num = intt + 1)
            splev_coords = interpolate.splev(interpoints,tck)
            interp_coords = np.round(splev_coords).T
            # if there was longer than one time_interval from the last transition,
            # the observed transition only occurred during the most recent time interval,
            # so change the timing after the actual interpolation such that spatial interpolation
            # stays the same, but the timescale changes
            if np.diff(time_units[t:t+2])[0] > time_interval:
                interpoints = np.linspace(
                    start=time_units[t+1]-time_interval,
                    stop = time_units[t+1],
                    num = intt+1,
                    )
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
    #add cell name and Treatment
    #'cell' is unique per source frame (unlike CellID/Treatment, which are
    #constant for the whole window); a single raw transition can expand into
    #multiple output rows above (e.g. diagonal crossings), but they all share
    #the same originating 'frame', so map each output row's frame back to the
    #raw transition that produced it to get the correct per-row cell value
    frame_to_cell = dict(zip(rawtrans.frame, rawtrans.cell))
    alltrans['cell'] = alltrans['frame'].map(frame_to_cell)
    alltrans['CellID'] = rawtrans.CellID.iloc[0]
    alltrans['Treatment'] = rawtrans.Treatment.iloc[0]

    return alltrans

def interpolate_trajectory_wrapper(args):
    return interpolate_trajectory(*args)


def get_transition_counts(
        bsdf, #dataframe with all transitions
        nbins, #bins in the CGPS
        ttot, #total time represented by the experiment
        dims, #list of dimension names, e.g. ['x', 'y']
        ):
    """
    Vectorized replacement for the old per-coordinate loop: for a bin's
    outgoing "for" counts, group membership already forces from_dim==coord[dim],
    so "to_dim < coord[dim]" is just "to_dim < from_dim" -- a per-transition
    direction flag that doesn't depend on which bin it's in. Symmetrically,
    a bin's incoming "rev" counts use the same per-transition flag, just
    grouped by the to_ coordinate instead of the from_ coordinate. So every
    bin's counts can be obtained with two groupby-sums over the whole
    dataframe instead of nbins**len(dims) separate full-dataframe scans.
    """
    #per-transition, per-dimension direction flags (independent of bin)
    flag_cols = []
    flags = pd.DataFrame(index=bsdf.index)
    for dim in dims:
        flags[f'{dim}_minus'] = (bsdf[f'to_{dim}'] < bsdf[f'from_{dim}']).astype(int)
        flags[f'{dim}_plus'] = (bsdf[f'to_{dim}'] > bsdf[f'from_{dim}']).astype(int)
        flag_cols += [f'{dim}_minus', f'{dim}_plus']

    #outgoing ("_for") counts: group by the transition's starting bin
    from_cols = [f'from_{dim}' for dim in dims]
    counts_for = pd.concat([bsdf[from_cols], flags], axis=1).groupby(from_cols)[flag_cols].sum()
    counts_for.index = counts_for.index.set_names(dims)

    #incoming ("_rev") counts: group by the transition's ending bin
    to_cols = [f'to_{dim}' for dim in dims]
    counts_rev = pd.concat([bsdf[to_cols], flags], axis=1).groupby(to_cols)[flag_cols].sum()
    counts_rev.index = counts_rev.index.set_names(dims)

    #full coordinate grid so every bin appears, even ones with zero transitions
    axes = [np.arange(1, nbins+1)] * len(dims)
    grid = np.meshgrid(*axes, indexing='ij')
    coords = np.stack(grid, axis=-1).reshape(-1, len(dims))
    full_index = pd.MultiIndex.from_arrays(coords.T, names=dims)

    counts_for = counts_for.reindex(full_index, fill_value=0)
    counts_rev = counts_rev.reindex(full_index, fill_value=0)

    trans_count = pd.DataFrame(index=full_index)
    for dim in dims:
        minus_for = counts_for[f'{dim}_minus']
        plus_for = counts_for[f'{dim}_plus']
        #an arrival "from the minus side" (from_dim < to_dim) is a transition
        #that moved in the + direction, so the rev columns read the opposite
        #flag from the for columns
        minus_rev = counts_rev[f'{dim}_plus']
        plus_rev = counts_rev[f'{dim}_minus']

        trans_count[f'{dim}_minus_count'] = minus_for
        trans_count[f'{dim}_minus_count_rev'] = minus_rev
        trans_count[f'{dim}_minus_for_rate'] = minus_for/ttot
        trans_count[f'{dim}_minus_rev_rate'] = minus_rev/ttot
        trans_count[f'{dim}_minus_rate'] = (minus_for - minus_rev)/ttot
        trans_count[f'{dim}_plus_count'] = plus_for
        trans_count[f'{dim}_plus_count_rev'] = plus_rev
        trans_count[f'{dim}_plus_for_rate'] = plus_for/ttot
        trans_count[f'{dim}_plus_rev_rate'] = plus_rev/ttot
        trans_count[f'{dim}_plus_rate'] = (plus_for - plus_rev)/ttot

    return trans_count.reset_index()




def build_graph(combodf, dims):
    """
    combodf : MultiIndex (transition_combination, transition_index) DataFrame
              with columns from_<dim>, to_<dim>, time_elapsed.
    dims    : list of dimension names, e.g. ['x', 'y']
 
    Returns a dict describing the transition graph as flat integer arrays:
      - to_node[c]     : ending position (node id) of combo c
      - total_time[c]  : summed time_elapsed of combo c's sub-transitions
      - offsets / flat_combo_idx : CSR index -> "which combos start at node p"
      - combo_rows[c]  : the original sub-transition rows for combo c
                         (used only at the very end, to rebuild the df)
    """

    firsttrans = combodf.xs(0, level='transition_index')
    lasttrans = combodf.groupby(level='transition_combination').tail(1)
    combo_ids = firsttrans.index.to_numpy()
 
    # get from and to position tuples
    from_tuples = list(zip(*[firsttrans['from_' + d].to_numpy() for d in dims]))
    to_tuples = list(zip(*[lasttrans['to_' + d].to_numpy() for d in dims]))
    # all position nodes observed and give them id
    all_nodes = sorted(set(from_tuples) | set(to_tuples))
    node_to_id = {node: i for i, node in enumerate(all_nodes)}
    # translate position tuples to node ids
    from_node = np.array([node_to_id[t] for t in from_tuples], dtype=np.int64)
    to_node = np.array([node_to_id[t] for t in to_tuples], dtype=np.int64)
    # get total time elapsed in each transition combo
    total_time = (
        combodf.groupby(level='transition_combination')['time_elapsed']
        .sum()
        .loc[combo_ids]
        .to_numpy()
    )
 
    # CSR layout: sort combos by their starting node so each node's outgoing
    # combos are contiguous; offsets[p]:offsets[p+1] gives that node's slice.
    order = np.argsort(from_node, kind='stable')
    from_node_sorted = from_node[order]
    n_nodes = len(all_nodes)
    offsets = np.searchsorted(from_node_sorted, np.arange(n_nodes + 1))
    #keep only the identity columns needed to look up precomputed raw/interpolated
    #data for a sampled transition
    #combodf is built (in get_bootstrapped_cgps_trajectories) with combo ids
    #0..n_combos-1 assigned in contiguous, ascending row order, ntrans rows
    #each -- so a per-column reshape to (n_combos, ntrans) lines up exactly
    #with combo_ids, one array per column 
    id_cols = ['Treatment', 'cell', 'CellID', 'frame']
    n_combos = len(combo_ids)
    ntrans = len(combodf) // n_combos
    combo_col_arrays = {
        col: combodf[col].to_numpy().reshape(n_combos, ntrans)
        for col in id_cols
    }

    return {
        'node_to_id': node_to_id,
        'to_node': to_node,
        'total_time': total_time,
        'offsets': offsets,
        'flat_combo_idx': order,
        'combo_ids': combo_ids,
        'id_cols': id_cols,
        'combo_col_arrays': combo_col_arrays,
        'n_nodes': n_nodes,
    }


 
def _handle_dead_end(graph, history, stuck_pos, dead_idx, rng, max_backtrack=20):
    """
    Very small, simple backtracking stand-in: since dead ends are rare,
    this can afford to just pick a fresh random start node rather than
    doing true path backtracking. Swap in your original backtracking
    logic here if you need exact behavioral parity.
    """
    n_nodes = graph['n_nodes']
    offsets = graph['offsets']
    # for _ in range(max_backtrack):
    #     back_combo = history[-r][dead_idx]
    #     back_combo_idx = flat_combo_ix[]
    #     candidate = rng.integers(0, n_nodes)
    #     if offsets[candidate + 1] > offsets[candidate]:
    #         return candidate, 0.0, True
    # return stuck_pos, 0.0, False
 
 
def batched_walk(
        graph,
        B,
        ttot,
        avoid_dead=False,
        max_steps=10000,
        ):
    """
    Advance B independent bootstrap replicates at once.
 
    Returns:
      history : list of 1D int arrays, one per step, giving the chosen
                combo index (into graph['combo_ids']) for each *active*
                replicate at that step, or -1 for replicates already done.
      cum_time: (B,) array of each replicate's final cumulative time.
    """
    offsets = graph['offsets']
    flat_combo_idx = graph['flat_combo_idx']
    to_node = graph['to_node']
    total_time = graph['total_time']
    n_nodes = graph['n_nodes']

    
    # start every replicate at a random node
    rng = np.random.default_rng()
    valid_starts = np.where(np.diff(graph['offsets']) > 0)[0]
    positions = rng.choice(valid_starts, size=B)
    cum_time = np.zeros(B)
    active = np.ones(B, dtype=bool)
    history = []
 
    for _ in range(max_steps):
        if not active.any():
            break
 
        idx = np.where(active)[0]
        pos = positions[idx]
 
        start = offsets[pos]
        end = offsets[pos + 1]
        degree = end - start
 
        dead = degree == 0
        if dead.any():
            # rare path: fix up stuck replicates one at a time
            if avoid_dead:
                for i in idx[dead]:
                    new_pos, extra_time, ok = _handle_dead_end(
                        graph, positions[i], rng
                    )
                    if not ok:
                        active[i] = False   # nowhere left to go; stop this one
                    else:
                        positions[i] = new_pos
                        cum_time[i] += extra_time
            else:
                active[idx[dead]] = False
            # recompute the active subset now that dead ones were resolved
            idx = np.where(active)[0]
            pos = positions[idx]
            start = offsets[pos]
            end = offsets[pos + 1]
            degree = end - start
 
        # vectorized random choice among each replicate's outgoing combos
        rand_offset = (rng.random(len(idx)) * degree).astype(np.int64)
        chosen_slot = start + rand_offset
        chosen_combo = flat_combo_idx[chosen_slot]
 
        step_choices = np.full(B, -1, dtype=np.int64)
        step_choices[idx] = chosen_combo
        history.append(step_choices)
 
        positions[idx] = to_node[chosen_combo]
        cum_time[idx] += total_time[chosen_combo]
 
        active[idx] = cum_time[idx] < ttot
 
    return history, cum_time
 


 
 
def reconstruct_trajectories(graph, history):
    """
    Reconstruct identity-only transition dataframes (Treatment, cell, CellID,
    frame) from graph positions mapped to combodf indices. time_elapsed/
    cumulative_time/real_time are not tracked here; see
    construct_bstrans_from_lookup to add those back in from the raw
    transitions.

    Fully vectorized: builds one flat list of (replicate, step) choices for
    every valid step across the whole walk, then looks up all their identity
    values with a single numpy fancy-index per column, instead of looping
    over replicates and pd.concat-ing pieces one at a time.
    """
    id_cols = graph['id_cols']
    combo_col_arrays = graph['combo_col_arrays']
    ntrans = next(iter(combo_col_arrays.values())).shape[1]
    B = history[0].shape[0]

    #(B, n_steps) grid of chosen local combo index per replicate/step, -1 = inactive
    Ht = np.stack(history, axis=1)
    valid = Ht != -1
    #row-major nonzero order = replicate-major, ascending step within each replicate
    b_idx, step_idx = np.nonzero(valid)
    combo_local = Ht[b_idx, step_idx]

    data = {
        col: combo_col_arrays[col][combo_local].reshape(-1)
        for col in id_cols
    }
    data['iter'] = np.repeat(b_idx, ntrans)

    return pd.DataFrame(data)


def construct_bstrans_from_lookup(id_bstrans, raw_data, id_cols=('Treatment', 'cell', 'CellID', 'frame')):
    """
    Rebuild the full data for an identity-only bootstrapped trajectory (from
    reconstruct_trajectories, or a saved identity-only bstrans CSV) by
    looking up each sampled transition's row in raw_data (any dataframe
    indexed by the same identity, e.g. rawtrans for the full raw trajectory,
    or a raw_..._aer_cf table for per-transition aer/angular_velocity/
    pc_speed) on demand, then recomputing cumulative_time/real_time for the
    spliced replicate order.
    """
    id_cols = list(id_cols)
    raw_lookup = raw_data.set_index(id_cols)
    raw_rows = raw_lookup.reindex(pd.MultiIndex.from_frame(id_bstrans[id_cols])).reset_index(drop=True)
    bstrans = pd.concat([id_bstrans.reset_index(drop=True), raw_rows], axis=1)
    bstrans['cumulative_time'] = bstrans.groupby('iter')['time_elapsed'].cumsum()
    bstrans['real_time'] = bstrans['cumulative_time']
    return bstrans


def interpolate_bootstrapped_trajectories(migboot, interpolated_trans, id_cols=('Treatment', 'cell', 'CellID', 'frame')):
    """
    Reconstruct the interpolated trajectories for every bootstrapped
    replicate at once by merging the identity-only bstrans (from
    reconstruct_trajectories) against the precomputed interpolated
    transitions (from get_interpolated_cgps_trajectories) on identity.
    """
    id_cols = list(id_cols)
    bsinttrans = migboot.merge(interpolated_trans, on=id_cols, how='left')
    bsinttrans['cumulative_time'] = bsinttrans.groupby('iter')['time_elapsed'].cumsum()
    bsinttrans['real_time'] = bsinttrans['cumulative_time']
    return bsinttrans
 
 



def transition_count_wrapper(
        args, # tuple of arguments
        sparse: bool = True, #remove zeros if true
        ):
    #unpack args from imap
    #bsdf: transition dataframe from bootstrap_trajectory()
    #nbins: bins in the CGPS
    bsdf, nbins = args

    ## get ttot
    ttot = bsdf.time_elapsed.sum()

    ## determine dimensions in the CGPS
    dims = [x.split('from_')[-1] for x in bsdf.columns if 'from_' in x]

    ############## get the counts of each bin position, vectorized over the
    ############## whole dataframe at once instead of looping per coordinate
    bstrans_rate_df = get_transition_counts(bsdf, nbins, ttot, dims)
    bstrans_rate_df = bstrans_rate_df.sort_values(by = dims).reset_index(drop=True)
    
    if sparse:
        rate_count_cols = [c for c in bstrans_rate_df.columns if any([col in c for col in ['count','rate']])]
        nonzero_mask = (bstrans_rate_df[rate_count_cols] != 0).any(axis=1)
        bstrans_rate_df = bstrans_rate_df[nonzero_mask].reset_index(drop=True)

    return bstrans_rate_df
    


def load_and_fill_transition_counts(
        filepath: Path,
        nbins: int,
        group_factor: str = None,
        ):
    """
    Load a sparse transition-rate csv (containing only coordinates with
    non-zero counts) and reconstruct the full CGPS grid, filling every
    missing coordinate's counts/rates with 0.
    """
    sparse_df = pd.read_csv(filepath, index_col=0)
    ## get column names
    dims = list(np.unique([c.split('_')[0] for c in sparse_df.columns if 'count' in c]))
    value_cols = [c for c in sparse_df.columns if any([val in c for val in ['rate','count']])]

    # build the full coordinate grid once
    axes = [np.arange(1, nbins + 1)] * len(dims)
    grid = np.meshgrid(*axes, indexing='ij')
    coords = np.stack(grid, axis=-1).reshape(-1, len(dims))
    coords_df = pd.DataFrame(coords, columns=dims)

    ### get non-value columns as list
    groups = ['Treatment', group_factor] if group_factor else ['Treatment']
    idx_cols = groups + dims

    # unique factor combinations actually present in the data
    factor_combos = sparse_df[groups].drop_duplicates()

    # single cross join
    coords_df['_key'] = 1
    factor_combos = factor_combos.assign(_key=1)
    full_index_df = factor_combos.merge(coords_df, on='_key').drop(columns='_key')

    # index-based join
    full_index_df = full_index_df.set_index(idx_cols)
    sparse_indexed = sparse_df.set_index(idx_cols)[value_cols]
    full_df = full_index_df.join(sparse_indexed, how='left')
    full_df[value_cols] = full_df[value_cols].fillna(0)

    full_df = full_df.reset_index().sort_values(by=idx_cols).reset_index(drop=True)

    return full_df[idx_cols + value_cols]



######## get area enclosing rates the "real" way with individual interpolated transitions
def get_area_enclosing_rate(
        transdf, #dataframe with from_x/from_y/to_x/to_y/time_elapsed columns (any row grouping)
        xyscaling, #list of the PC factors by which to scale the x and y coordinates of the CGPS in [x,y] format
        origin, #coordinates of the flux origin in [x,y] format
        ):
    """
    Instantaneous area enclosing rate, angular velocity (cycling frequency),
    and pc_speed for every transition in transdf.
    """
    transdf = transdf.copy()

    #center coordinates on the flux origin and scale them
    from_x = (transdf['from_x'] - origin[0]) * xyscaling[0]
    to_x = (transdf['to_x'] - origin[0]) * xyscaling[0]
    from_y = (transdf['from_y'] - origin[1]) * xyscaling[1]
    to_y = (transdf['to_y'] - origin[1]) * xyscaling[1]

    transdf['aer'] = ((from_y*to_x) - (from_x*to_y)) / (2*transdf['time_elapsed'])

    ######## "For instance, we could track a pair of degrees of freedom 𝐱r={𝑥𝑖,𝑥𝑗} and measure the time average of the angular velocity ⟨̇𝛽𝑖⁢𝑗⟩,
    ######## or equivalently, the rate at which the trajectory revolves around the origin in this reduced two-dimensional subspace (Fig. 2).
    ######## This simple measurement does not require any discretization of phase space or inference of the force field.
    ######## We shall refer to ⟨̇𝛽𝑖⁢𝑗⟩ as the cycling frequency.
    ######## https://doi.org/10.1103/PhysRevE.99.052406
    #vectorized clock_counterclock_angle(from, to) = -signed_angle(from, to)
    angle_deg = -np.degrees(np.arctan2(from_x*to_y - from_y*to_x, from_x*to_x + from_y*to_y))
    transdf['angular_velocity'] = angle_deg / transdf['time_elapsed']

    ### also calculate "pc_speed"
    transdf['pc_speed'] = np.sqrt((to_x - from_x)**2 + (to_y - from_y)**2) / transdf['time_elapsed']

    return transdf



def get_linear_rates_wrap(
        args
        ):
    
    ### unpack args
    # df, #dataframe containing "iter" bootstrap iteration ID, some group_factor, and "aer"
    df, group_factor = args

    ## fit rate
    rate_fit_dict = utils.calculate_rates(
        df,
        group_factor,
        )
    return rate_fit_dict


def get_raw_cgps_trajectories(
        TotalFrame, #pandas dataframe with all of the cgps binned data
        whichpcs, #which two PCs to use in the cgps [x,y]
        config: Config,
        ):
    ## get settings from config
    time_interval = config.im_params.time_interval
    dbsavedir = config.common.savedir / 'detailed_balance'
    if not dbsavedir.exists():
        dbsavedir.mkdir()

    mapargs = []
    for i, cells in TotalFrame.groupby(['Treatment','CellID']):
        cells, runs = utils.get_consecutive_timepoints(cells, 'time', time_interval)
        for r in runs:
            #only use runs with 3 or more frames
            if len(r)>2:
                mapargs.append((
                    time_interval,
                    cells.iloc[r],
                    whichpcs,
                ))
    print(f'Aggregating {whichpc_string(whichpcs)} transitions')
    with multiprocessing.Pool(processes=60) as pool:
        results = list(tqdm.tqdm(pool.imap(raw_transitions_wrapper, mapargs), total=len(mapargs)))
    rawtrans = pd.concat(results)
    rawtrans = rawtrans.sort_values(by = ['Treatment','CellID','real_time']).reset_index(drop=True)
    rawtrans.to_csv(dbsavedir.joinpath(whichpc_string(whichpcs)+'_raw_transitions.csv'))
    
    return rawtrans


def get_interpolated_cgps_trajectories(
        rawtrans, #pandas dataframe with raw transitions from get_raw_cgps_trajectories
        whichpcs, #which two PCs to use in the cgps [x,y]
        config: Config,
        ):
    
    ## get settings from config
    dbsavedir = config.common.savedir / 'detailed_balance'
    time_interval = config.im_params.time_interval

    mapargs = []
    for i, cell in rawtrans.groupby('CellID'):
        cell, runs = utils.get_consecutive_transitions(cell)
        for r in runs:
            #interpolate_trajectory works fine on a single-transition run (just
            #the from/to endpoints), so every run is interpolated -- otherwise an
            #isolated single-transition run would be sampleable by
            #get_bootstrapped_cgps_trajectories but missing from the lookup here
            mapargs.append((cell.iloc[r], time_interval))
    print(f'Interpolating {whichpc_string(whichpcs)} trajectories')
    with multiprocessing.Pool(processes=60) as pool:
        results = list(tqdm.tqdm(pool.imap(interpolate_trajectory_wrapper, mapargs), total=len(mapargs)))

    #separate results into transtions and transition pairs
    transdf_sep = pd.concat(results)
    transdf_sep = transdf_sep.sort_values(by = ['Treatment','CellID','real_time']).reset_index(drop=True)
    transdf_sep.to_csv(dbsavedir.joinpath(whichpc_string(whichpcs)+'_interpolated_transitions.csv'))
    
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
        trans_rate_df_sep = transition_count_wrapper((mig, nbins))
        ## add group_factor
        trans_rate_df_sep[group_factor] = m
        ## append to list of all group transition counts
        trresults.append(trans_rate_df_sep)

    trans_rate_df_sep = pd.concat(trresults)
    trans_rate_df_sep.to_csv(dbsavedir.joinpath(whichpc_string(whichpcs)+'_binned_transition_rates.csv'))
    
    return trans_rate_df_sep



def match_dataset_distribution(
        real_df: pd.DataFrame,
        bs_df: pd.DataFrame,
        num_replicates: int = 1,
        ):
    """
    Trims bootstrapped iterations to the size and number of
    real tracks from the original dataset a number of times equal
    to num_replicates.

    """
    treatments = bs_df.Treatment.unique()
    all_indices = []
    dataset_replicates = []

    for treat in treatments:
        real_df_treat = real_df[real_df.Treatment == treat]
        bs_df_treat = bs_df[bs_df.Treatment == treat]

        # sorted bs track lengths
        bs_tracks = bs_df_treat.groupby('iter').cumulative_time.max().sort_values()
        sorted_vals = bs_tracks.values
        sorted_iters = bs_tracks.index.to_numpy()
        n_iters = len(sorted_iters)
        available = np.ones(n_iters, dtype=bool)

        # precompute per-iter lookups
        iter_time_sets = {}
        iter_time_arrays = {}
        for it, g in bs_df_treat.groupby('iter'):
            values = g['cumulative_time'].to_numpy()
            orig_index = g.index.to_numpy()
            iter_time_sets[it] = set(values.tolist())
            iter_time_arrays[it] = (values, orig_index)

        for cellid, cell_track in real_df_treat.groupby('CellID'):
            ## order track by time and get indices of consecutive runs
            cell_track, runs = utils.get_consecutive_transitions(cell_track)
            #get the times for this track and shift them to start at 10
            #which is the minimum time in the bs datasets
            cell_times = cell_track.real_time.to_numpy(copy=True)
            cell_times -= cell_times.min() - 10
            # get the start and end indices of each run
            run_start_end = [[cell_times[run[0]], cell_times[run[-1]]] for run in runs]
            #make a set to match with bs times
            target_times = set(x for run in run_start_end for x in run)
        
            # jump straight to the first viable candidate
            threshold = cell_track.time_elapsed.sum()
            start_idx = np.searchsorted(sorted_vals, threshold, side='left')
            # iterate to find a bs iter that has all the starts and ends of
            # the real track
            for r in range(num_replicates):
                first_greater = None
                chosen_pos = None
                for pos in range(start_idx, n_iters):
                    if not available[pos]:
                        continue
                    it = sorted_iters[pos]
                    if target_times.issubset(iter_time_sets[it]):
                        first_greater = it
                        chosen_pos = pos
                        break

                if first_greater is None:
                    raise ValueError(
                        f"No bootstrapped iter found containing all timepoints for CellID={cellid}"
                    )
                #get the bs iter that was chosen
                values, orig_index = iter_time_arrays[first_greater]
                #get all the indices between starts and ends of real runs
                iter_indices = []
                for start, end in run_start_end:
                    mask = (values >= start) & (values <= end)
                    iter_indices.extend(orig_index[mask].tolist())

                all_indices.extend(iter_indices)
                dataset_replicates.extend([r] * len(iter_indices))
                #remove the chosen index from future consideration
                available[chosen_pos] = False

    return pd.DataFrame({
        'replicate_id': dataset_replicates,
        'bs_indices': all_indices,
    })

############## BOOTSTRAP MANY TRAJECTORIES ##########
def get_bootstrapped_cgps_trajectories(
        rawtrans, #raw transitions from get_raw_cgps_trajectories
        interpolated_trans, #already-interpolated transitions from get_interpolated_cgps_trajectories
        whichpcs, #which two PCs to use in the cgps [x,y]
        config: Config,
        dbbssavedir: Path, #where to save the bootstrapped dataframes
        group_factor: str = 'Treatment', #column with factor to separate the data on
        ):

    wpc_str = utils.whichpc_string(whichpcs)
    ### get some settings from config
    if not dbbssavedir.exists():
        dbbssavedir.mkdir()
    nbins = config.db_params.nbins #how many bins in the x and y cgps axes
    ttot = config.db_params.ttot #set the total bootstrap time
    ntrans = config.db_params.ntrans #how many transitions to sample at each step
    bsiter = config.db_params.bsiter #number of times to bootstrap

    #make a bunch of lists that I will append things to as I go for each treatment
    bstrans = []
    bsframe_sep_full = []

    #bootstrap from raw trajectories
    with multiprocessing.Pool(processes=60) as pool:
        for m, mig in rawtrans.groupby(group_factor):            
            if ntrans == 1:
                combodf = mig.copy()
            else:
                combolist = []
                for cidc, cell in mig.groupby('CellID'):
                    cell, runs = utils.get_consecutive_transitions(cell)
                    for r in runs:
                        r = np.asarray(r)
                        n_windows = len(r) - ntrans + 1
                        if n_windows <= 0:
                            continue
                        # build (n_windows, ntrans) matrix of positions, then flatten
                        window_idx = r[np.arange(n_windows)[:, None] + np.arange(ntrans)[None, :]]
                        positions = window_idx.ravel()
                        combo = cell.iloc[positions]
                        combolist.append(combo)

                combodf = pd.concat(combolist)

            ## create and add multiindex for the number of transitions
            miarray = [
                # unique transition combo index
                np.repeat(range(int(len(combodf) / ntrans)), ntrans),
                # index of individual transitions within the combo
                np.tile(list(range(ntrans)), int(len(combodf) / ntrans))
            ]
            miindex = pd.MultiIndex.from_arrays(miarray, names=['transition_combination', 'transition_index'])
            combodf.index = miindex

            ## build a graph of the observed transitions to quickly
            ## walk through with random sampling
            dims = [x.split('from_')[-1] for x in combodf.columns if 'from_' in x]
            graph = build_graph(combodf, dims)
            ## use graph to bootstrap CGPS trajectories
            history, cum_time = batched_walk(graph, bsiter, ttot)
            ## convert transition indices to a dataframe with all
            ## bootstrapped interations
            migboot = reconstruct_trajectories(graph, history)
            #append the identity-only trajectory to the larger list of dataframes;
            bstrans.append(migboot)

        

            ###### now look up precomputed interpolated segments for the bootstrapped
            ###### trajectories, all replicates at once via a single merge
            print(f'Looking up interpolated trajectories for {m}')
            bsinttrans = interpolate_bootstrapped_trajectories(migboot, interpolated_trans)
            bsinttrans = bsinttrans.sort_values(by = ['iter','cumulative_time']).reset_index(drop=True)
            bsinttrans[group_factor] = m


            ###### now get transition rates
            #get list of tuples of arguments to pass to imap
            mapargs = [(it, nbins) for _, it in bsinttrans.groupby('iter')]
            #calculate bootstrapped transition rates
            print(f'Calculating bootstrapped CGPS transition rates for {m}')
            results = list(tqdm.tqdm(pool.imap(transition_count_wrapper, mapargs), total=bsiter))

            #combine and add other info
            migrate = pd.concat(results, ignore_index=True)
            migrate[group_factor] = m
            migrate['iter'] = list(itertools.chain.from_iterable([[k]*len(res) for k,res in enumerate(results)]))
            bsframe_sep_full.append(migrate)
        

    ####### pull everything together and save
    bstrans = pd.concat(bstrans, ignore_index=True)
    bstrans.to_csv(dbbssavedir.joinpath(f'{wpc_str}_bootstrapped_{ntrans}_transitions.csv'))
    bsframe_sep_full = pd.concat(bsframe_sep_full, ignore_index=True)
    bsframe_sep_full.to_csv(dbbssavedir.joinpath(f'{wpc_str}_bootstrapped_{ntrans}_transition_rates.csv'))
    print('Finished bootstrapping')

    ############# open average bootstrapped currents (saves its own csv) ###################
    get_avg_current_error(whichpcs, dbbssavedir, config, 'iter')

    return bstrans, bsframe_sep_full
    

############# open average bootstrapped currents ###################
def get_avg_current_error(
        whichpcs, #which two PCs to use in the cgps [x,y]
        dbbssavedir: Path, #where to save the bootstrapped dataframes
        config: Config,
        group_factor: str, #column with factor to separate the data on
        ):
    wpc_str = utils.whichpc_string(whichpcs)
    ### get some settings from config
    nbins = config.db_params.nbins #how many bins in the x and y cgps axes
    ntrans = config.db_params.ntrans #how many transitions to sample at each step
    ### open the data and fill sparse gaps with zeros to get real means
    bsframe_sep_full = load_and_fill_transition_counts(
        dbbssavedir.joinpath(f'{wpc_str}_bootstrapped_{ntrans}_transition_rates.csv'),
        nbins,
        group_factor,
    )

    #### estimate error in current field for this set of bootstrap realizations ######
    ####### this is for looking at data spread for the current field ############
    full_index = pd.MultiIndex.from_product(
        [range(1, nbins + 1), range(1, nbins + 1)],
        names=['x', 'y']
    )

    bsfield = []
    for m, mig in bsframe_sep_full.groupby('Treatment'):
        rows = []
        for (x, y), current in mig.groupby(['x', 'y']):
            js = np.column_stack([
                (current['x_plus_rate'].to_numpy() - current['x_minus_rate'].to_numpy()) / 2,
                (current['y_plus_rate'].to_numpy() - current['y_minus_rate'].to_numpy()) / 2,
            ])
            if js.shape[0] < 2:
                # not enough samples to estimate covariance/error at this bin
                evals = np.array([0.0, 0.0])
                evecs = np.eye(2)
            else:
                js_centered = js - js.mean(axis=0)
                avgjs = np.cov(js_centered.T)
                evals, evecs = np.linalg.eigh(avgjs)
            rows.append({'x':x,
                        'y':y,
                        'eval1':evals[1],
                        'eval2':evals[0],
                        'evec1x':evecs[0,1],
                        'evec1y':evecs[1,1],
                        'evec2x':evecs[0,0],
                        'evec2y':evecs[1,0],
                        'Treatment':m,
                        })
        default_row = {
            'eval1': 0,
            'eval2': 0,
            'evec1x': 0,
            'evec1y': 1,
            'evec2x': 1,
            'evec2y': 0,
            'Treatment':m,
        }
        df_m = pd.DataFrame(rows).set_index(['x', 'y'])
        df_m = df_m.reindex(full_index)
        df_m = df_m.fillna(value=default_row).reset_index()
        bsfield.append(df_m)

    bsfield_sep = pd.concat(bsfield, ignore_index=True)
    bsfield_sep.to_csv(dbbssavedir.joinpath(f'{wpc_str}_bootstrapped_{ntrans}_transitions_average_currents.csv'))
    
    return bsfield_sep


########## calculate all the aers and cycling frequencies from the bootstrapped data
def get_aer_cf(
        transdf, #dataframe with transitions
        whichpcs, #which two PCs to use in the cgps [x,y]
        config: Config,
        group_factor: str, #column with factor to separate the data on
        ):
    
    ### get some settings from config
    savedir = config.common.savedir
    
    if any([w<0 for w in whichpcs]):
        pc_combos = config.common.pc_combos_sym #unique PC pairs
        origins = config.db_params.origins_sym #flux origins for this dataset and alignment
    else:
        pc_combos = config.common.pc_combos #unique PC pairs
        origins = config.db_params.origins #flux origins for this dataset and alignment
    origin = origins[pc_combos.index(whichpcs)]

    ## open the CGPS bins to get scaling
    datadir = savedir / 'shape_data'
    centers = pd.read_csv(datadir.joinpath('PC_bin_centers.csv'), index_col=0)
    #scaling of the bins in real units of whatever the CGPS axis parameters are
    xyscaling = [centers[f'PC{abs(wpc)}'].diff().mean() for wpc in whichpcs]

    #compute instantaneous aer/angular_velocity/pc_speed for every transition
    #at once -- each row's value only depends on its own from_/to_/time_elapsed
    #columns, so no per-run splitting or multiprocessing is needed here
    allaers = get_area_enclosing_rate(transdf, xyscaling, origin)
    ## keep only new columns and those needed for ID (including the identity
    ## columns, so this output can also be used as a lookup table via
    ## construct_bstrans_from_lookup)
    new_cols = [x for x in allaers.columns if x not in transdf.columns]
    id_cols = ['Treatment', 'cell', 'CellID', 'frame']
    keep_cols = list(dict.fromkeys(id_cols + [group_factor] + [x for x in allaers.columns if 'time' in x]))
    allaers = allaers[keep_cols + new_cols]

    lrrdf = get_linear_rates(allaers, group_factor)

    return allaers, lrrdf


def get_linear_rates(allaers, group_factor):
    """
    Fit average aer/angular_velocity/pc_speed etc. per (Treatment,
    group_factor) group.
    """
    ### collect args for average aer, etc. 
    lrmapargs = [(df.sort_values('cumulative_time').reset_index(drop = True),
                group_factor) for _, df in allaers.groupby(['Treatment',group_factor])]

    with multiprocessing.Pool(processes=60) as pool:
        lrresults = list(tqdm.tqdm(pool.imap(get_linear_rates_wrap, lrmapargs), total=len(lrmapargs)))
    return pd.DataFrame(lrresults)


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
    wpc_str = utils.whichpc_string(whichpcs)
    #get constants from config
    ntrans = config.db_params.ntrans #how many transitions to sample at each step

    ## get directories from config
    savedir = config.common.savedir
    dbdir = savedir / 'detailed_balance'
    dbbssavedir = dbdir / 'separatedatabs'

    justaers = pd.read_csv(dbdir.joinpath(f'{wpc_str}_raw_transition_aer_cf.csv'), index_col = 0)

    ########## measure gap frequency and duration
    allrunlengths, allrunlengthmeans, allgaplengths, allgaplengthmeans, allgapfrequencies = get_run_stats(
            justaers, #dataframe
            'CellID', #what is the identifier to group by as a str
            config, #frame rate of the data
            )
    print(f'Average track run length mean for real data is {np.mean(allrunlengthmeans)} and mean gap frequency is {np.mean(allgapfrequencies)})')

    #### get bs data with gaps
    #bstrans is identity-only (Treatment, cell, CellID, frame, iter); look it up
    #against justaers (the raw aer table used for the real-data run stats above)
    #to get the full aer/time columns, instead of reading a separately-saved
    #bootstrapped aer_cf csv
    bstrans = pd.read_csv(dbbssavedir.joinpath(f'{wpc_str}_bootstrapped_{ntrans}_transitions.csv'), index_col = 0)
    bsaers = construct_bstrans_from_lookup(bstrans, justaers)

    bs_with_gaps = bootstrap_runs(
        bsaers, #dataframe with bootstrap iterations (doesn't actually need aer)
        allrunlengths, #the sample of movies lengths in seconds
        allgaplengths, #the sample of non-movie gap lengths in seconds
        )
    ### save the gapped bootstrap ids
    bs_with_gaps.to_csv(dbbssavedir.joinpath(f'{wpc_str}_bootstrapped_{ntrans}_gap_ids.csv'))

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



    ### merged gaps with actual bootstrapped aers so we can get average aer with the gapped data
    aers_with_gaps = bs_with_gaps.merge(bsaers, on = ['iter','real_time'], how = 'left') 
    lrrdf = get_linear_rates(aers_with_gaps, 'iter')
    lrrdf.to_csv(dbbssavedir.joinpath(f'{wpc_str}_bootstrapped_{ntrans}_linear_rates_gaps.csv'))
