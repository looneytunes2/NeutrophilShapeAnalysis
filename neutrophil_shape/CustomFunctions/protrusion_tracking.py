# -*- coding: utf-8 -*-
"""
Protrusions are linked within each cell track (CellID) using the per-protrusion
metrics saved by shparam_mod (see shparam_mod.measure_protrusions_from_mesh):
- the angle between the 'real_direction' vectors (cell-centered, real space, so
  it does not depend on the alignment frame, which can change between frames),
- the distance between the 'position' vectors (cell-centered, real space
  orientation, um), which also accounts for how far out from the cell centroid
  it is. Using the cell-centered position keeps the cell's own movement out of
  the comparison,
- the change in 'area_um2'.
Each term is divided by its gate so it runs from 0 to 1, and a pair beyond any
gate cannot be linked.

Linking happens in stages, each an optimal one-to-one assignment (linear
assignment problem) that leaves protrusions unlinked rather than forcing a
match:
1. frame to frame: protrusions in consecutive frames are linked into track
   segments. A protrusion that is not linked to the previous frame starts a
   new segment; one that is not linked to the next frame ends its segment.
2. gap closing: segment ends are linked to later segment starts up to
   max_gap_frames missing frames apart (a protrusion that dipped below the
   detection threshold, or a gap in the cell's data), with the gates widened
   by the square root of the number of frames apart. The linked segments
   become one track.
3. merges: a segment end that is still unlinked and sits next to a protrusion
   of another track in the next frame is recorded as merging into that track.
4. splits: a segment start that is still unlinked and sits next to a
   protrusion of another track in the previous frame is recorded as splitting
   from that track.
Once the protrusions are linked, the 'real_direction' and 'real_position'
(the position in real space, including the cell's own movement) of each
protrusion are compared with those of the previous frame's protrusion in its
track to get its trajectory, see track_protrusions.

Merges and splits are greedy (each is matched to its best candidate on its
own), and unlike steps 1-2 several protrusions may merge into or split from the
same one. In both, the area of the combined protrusions (the two merging
protrusions, or the two daughters of a split) is what gets compared to the
area of the single one.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, replace
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TrackingParams:
    """
    Gates and weights for linking protrusions between frames.

    max_angle_deg : largest angle (degrees) between the real_direction vectors
        of two protrusions that can be linked.
    max_distance_um : largest distance (um) between the position vectors
        (relative to the cell centroid) of two protrusions that can be linked.
    max_area_ratio : largest ratio (either way) between the areas of two
        protrusions that can be linked.
    angle_weight, distance_weight, area_weight : relative weights of the three
        gated terms in the linking cost.
    max_gap_frames : most missing frames between two segments that can still be
        closed into one track. 0 turns gap closing off.
    detect_merges_and_splits : whether to look for tracks that merge into or
        split from other tracks.
    """
    max_angle_deg: float = 60.0
    max_distance_um: float = 5.0
    max_area_ratio: float = 3.0
    angle_weight: float = 1.0
    distance_weight: float = 1.0
    area_weight: float = 0.5
    max_gap_frames: int = 2
    detect_merges_and_splits: bool = True


_DIRECTION_COLS = ['real_direction_x', 'real_direction_y', 'real_direction_z']
_POSITION_COLS = ['position_x', 'position_y', 'position_z']
_REAL_POSITION_COLS = ['real_position_x', 'real_position_y', 'real_position_z']
_TRAJECTORY_COLS = (
    [f'direction_trajectory_{d}' for d in 'xyz']
    + [f'protrusion_trajectory_{d}' for d in 'xyz']
    + ['protrusion_distance_um'])


def _pair_cost(
        a_direction, a_position, a_area,  # (n,3), (n,3), (n,) of the earlier protrusions
        b_direction, b_position, b_area,  # (m,3), (m,3), (m,) of the later protrusions
        params: TrackingParams,
        ):
    """
    (n, m) linking cost between every a and every b. The cost of a feasible
    pair runs from 0 (identical) to 1 (at the edge of every gate); pairs
    outside any gate are infeasible and get np.inf.
    """
    cos = np.clip(np.asarray(a_direction) @ np.asarray(b_direction).T, -1, 1)
    angle_term = np.degrees(np.arccos(cos)) / params.max_angle_deg
    distance_term = (np.linalg.norm(np.asarray(a_position)[:, None, :] - np.asarray(b_position)[None, :, :], axis=2)
                     / params.max_distance_um)
    area_term = (np.abs(np.log(np.asarray(b_area, dtype=float)[None, :] / np.asarray(a_area, dtype=float)[:, None]))
                 / np.log(params.max_area_ratio))

    weights = np.array([params.angle_weight, params.distance_weight, params.area_weight])
    terms = np.stack([angle_term, distance_term, area_term])
    cost = np.tensordot(weights, terms, axes=1) / weights.sum()
    cost[(terms > 1).any(axis=0)] = np.inf
    return cost


def _scaled_gates(params: TrackingParams, factor: float):
    """params with every gate widened by factor (the area gate in log space)."""
    return replace(
        params,
        max_angle_deg=params.max_angle_deg * factor,
        max_distance_um=params.max_distance_um * factor,
        max_area_ratio=params.max_area_ratio ** factor,
        )


def linking_cost(
        prev: pd.DataFrame,  # protrusions in the earlier frame
        curr: pd.DataFrame,  # protrusions in the later frame
        params: TrackingParams,
        ):
    """Pairwise linking cost between the protrusions of two frames, see _pair_cost."""
    return _pair_cost(
        prev[_DIRECTION_COLS].values, prev[_POSITION_COLS].values, prev.area_um2.values,
        curr[_DIRECTION_COLS].values, curr[_POSITION_COLS].values, curr.area_um2.values,
        params)


def _solve_assignment(cost: np.ndarray):
    """
    One-to-one assignment of the rows of a cost matrix to its columns that
    minimizes the total cost, where inf entries cannot be assigned. Returns a
    list of (row, column, cost) for the assigned pairs; every other row or
    column is left unassigned.

    This is the standard linear assignment problem with dummy rows and columns:
    leaving a row or column unassigned costs 1, so a feasible pair (cost <= 1)
    is always preferred to leaving both of its members unassigned (cost 2).
    """
    n, m = cost.shape
    if n == 0 or m == 0:
        return []
    feasible = np.isfinite(cost)

    big = 1e6
    full = np.full((n + m, n + m), big)
    full[:n, :m] = np.where(feasible, cost, big)
    full[np.arange(n), m + np.arange(n)] = 1.0   # row i is left unassigned
    full[n + np.arange(m), np.arange(m)] = 1.0   # column j is left unassigned
    full[n:, m:] = 0.0                           # unused dummies pair off freely

    rows, cols = linear_sum_assignment(full)
    return [(i, j, cost[i, j]) for i, j in zip(rows, cols) if i < n and j < m and feasible[i, j]]


def match_protrusions(
        prev: pd.DataFrame,
        curr: pd.DataFrame,
        params: TrackingParams,
        ):
    """
    One-to-one assignment of protrusions in prev to protrusions in curr that
    minimizes the total linking cost. Returns a list of (prev position, curr
    position, cost) for the linked pairs; everything else is a death (in prev)
    or a birth (in curr).
    """
    return _solve_assignment(linking_cost(prev, curr, params))


def _track_cell(
        cell: pd.DataFrame,  # protrusions of one CellID, with a default (0..n-1) index
        time_interval: float,
        params: TrackingParams,
        ):
    """Track the protrusions of one cell, returns a dict of per-row arrays (see track_protrusions)."""
    n = len(cell)
    direction = cell[_DIRECTION_COLS].values
    position = cell[_POSITION_COLS].values
    real_position = cell[_REAL_POSITION_COLS].values
    area = cell.area_um2.values.astype(float)
    frame = np.rint((cell.time.values - cell.time.values.min()) / time_interval).astype(int)
    by_frame = {f: np.flatnonzero(frame == f) for f in np.unique(frame)}

    track = np.full(n, -1)
    prev = np.full(n, -1)   # row linked to in the previous frame(s), -1 if none
    succ = np.full(n, -1)   # row linked to in the next frame(s), -1 if none
    cost = np.full(n, np.nan)
    link = np.full(n, 'birth', dtype=object)
    parent_track = np.full(n, np.nan)
    merge_track = np.full(n, np.nan)

    def cost_between(a, b, params=params, a_area=None, b_area=None):
        """(len(a), len(b)) cost between rows a and rows b, optionally with other areas."""
        return _pair_cost(
            direction[a], position[a], area[a] if a_area is None else a_area,
            direction[b], position[b], area[b] if b_area is None else b_area,
            params)

    ### 1. frame to frame
    next_id = 0
    for f in sorted(by_frame):
        idx = by_frame[f]
        matched = set()
        if f - 1 in by_frame:
            pidx = by_frame[f - 1]
            for i, j, c in _solve_assignment(cost_between(pidx, idx)):
                track[idx[j]] = track[pidx[i]]
                prev[idx[j]] = pidx[i]
                succ[pidx[i]] = idx[j]
                cost[idx[j]] = c
                link[idx[j]] = 'linked'
                matched.add(j)
        for j in range(len(idx)):
            if j not in matched:
                next_id += 1
                track[idx[j]] = next_id

    ### 2. gap closing: segment ends to later segment starts
    if params.max_gap_frames > 0:
        ends = np.flatnonzero(succ == -1)
        starts = np.flatnonzero(prev == -1)
        frames_apart = frame[starts][None, :] - frame[ends][:, None]
        gap_cost = np.full((len(ends), len(starts)), np.inf)
        for k in range(2, params.max_gap_frames + 2):
            in_range = frames_apart == k
            if in_range.any():
                widened = cost_between(ends, starts, _scaled_gates(params, np.sqrt(k)))
                gap_cost[in_range] = widened[in_range]
        for i, j, c in _solve_assignment(gap_cost):
            e, s = ends[i], starts[j]
            track[track == track[s]] = track[e]
            prev[s] = e
            succ[e] = s
            cost[s] = c
            link[s] = 'gap_closed'

    if params.detect_merges_and_splits:
        ### 3. merges: an unlinked end next to a continuing protrusion of another track
        for e in np.flatnonzero(succ == -1):
            if frame[e] + 1 not in by_frame:
                continue
            cand = by_frame[frame[e] + 1]
            cand = cand[(track[cand] != track[e]) & (prev[cand] >= 0)]
            best_cost, best = np.inf, None
            for q in cand:
                #the merged protrusion should be about as big as e and the protrusion q continues
                c = cost_between([e], [q], a_area=np.array([area[e] + area[prev[q]]]))[0, 0]
                if c < best_cost:
                    best_cost, best = c, q
            if best is not None:
                merge_track[e] = track[best]

        ### 4. splits: an unlinked start next to a continuing protrusion of another track
        for s in np.flatnonzero(prev == -1):
            if frame[s] - 1 not in by_frame:
                continue
            cand = by_frame[frame[s] - 1]
            cand = cand[(track[cand] != track[s]) & (succ[cand] >= 0)]
            best_cost, best = np.inf, None
            for q in cand:
                #the two daughters (s and the protrusion q continues as) should add up to about q
                c = cost_between([q], [s], b_area=np.array([area[s] + area[succ[q]]]))[0, 0]
                if c < best_cost:
                    best_cost, best = c, q
            if best is not None:
                link[s] = 'split'
                cost[s] = best_cost
                parent_track[s] = track[best]

    ### trajectory of every protrusion linked to the previous frame's protrusion
    trajectory = np.full((n, len(_TRAJECTORY_COLS)), np.nan)
    linked = np.flatnonzero(link == 'linked')
    direction_change = direction[linked] - direction[prev[linked]]
    position_change = real_position[linked] - real_position[prev[linked]]
    distance = np.linalg.norm(position_change, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):   #no change has no direction
        trajectory[linked, 0:3] = direction_change / np.linalg.norm(direction_change, axis=1, keepdims=True)
        trajectory[linked, 3:6] = position_change / distance[:, None]
    trajectory[linked, 6] = distance

    ### number the tracks 1, 2, ... in order of first appearance
    order = np.lexsort((np.arange(n), frame))
    new_id = {old: i + 1 for i, old in enumerate(pd.unique(track[order]))}
    renumber = np.vectorize(lambda t: new_id[int(t)] if np.isfinite(t) else np.nan, otypes=[float])
    return {
        'track_id': np.array([new_id[t] for t in track]),
        'match_cost': cost,
        'link_type': link,
        'parent_track_id': renumber(parent_track),
        'merges_into_track_id': renumber(merge_track),
        **{name: trajectory[:, k] for k, name in enumerate(_TRAJECTORY_COLS)},
    }


def track_protrusions(
        protdf: pd.DataFrame,  # protrusion metrics with CellID, time, area_um2 and the real_direction/position/real_position columns
        time_interval: float,  # time between frames in seconds
        params: TrackingParams = None,
        ):
    """
    Link protrusions between the frames of each cell track, see the module
    docstring for the stages.

    Returns a copy of protdf (same index) with these columns added:
    - 'track_id': integer identifying the protrusion's track, unique within
      its CellID (use CellID and track_id together),
    - 'match_cost': linking cost (0-1) to the protrusion before it in its
      track, or, for a split, to its parent; NaN for the first protrusion of
      a track that did not split off,
    - 'link_type': how the protrusion is connected to the one before it:
      'linked' (previous frame), 'gap_closed' (after missing frames),
      'split' (first protrusion of a track that split off another track) or
      'birth' (first protrusion of a track, nothing before it),
    - 'parent_track_id': for 'split' rows, the track_id it split from (NaN
      otherwise),
    - 'merges_into_track_id': for the last protrusion of a track that merges
      into another, the track_id it merges into (NaN otherwise),
    and, for every protrusion linked to the previous frame's protrusion
    (link_type 'linked'; NaN for the others, including gap_closed, whose
    previous protrusion is more than one frame away):
    - 'direction_trajectory_x/y/z': unit vector of the change in real_direction
      from the previous frame's protrusion to this one, i.e. which way the
      direction the protrusion points in is turning,
    - 'protrusion_trajectory_x/y/z': unit vector of the change in real_position,
      i.e. which way the protrusion moved in real space (so it includes the
      cell's own movement),
    - 'protrusion_distance_um': the 3D distance between the two real_positions.
    A change of exactly zero has no direction and gives NaN for its vector.
    """
    if params is None:
        params = TrackingParams()

    df = protdf.reset_index(drop=True)
    columns = {
        'track_id': np.full(len(df), -1),
        'match_cost': np.full(len(df), np.nan),
        'link_type': np.full(len(df), 'birth', dtype=object),
        'parent_track_id': np.full(len(df), np.nan),
        'merges_into_track_id': np.full(len(df), np.nan),
        **{name: np.full(len(df), np.nan) for name in _TRAJECTORY_COLS},
    }
    for _, cell_idx in df.groupby('CellID').groups.items():
        cell_idx = np.asarray(cell_idx)
        result = _track_cell(df.loc[cell_idx].reset_index(drop=True), time_interval, params)
        for name, values in result.items():
            columns[name][cell_idx] = values

    out = protdf.copy()
    for name, values in columns.items():
        out[name] = values
    return out
