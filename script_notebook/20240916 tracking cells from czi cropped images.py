# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:46:32 2024

@author: Aaron
"""

import os
import pandas as pd
import numpy as np
import re
import scipy

from CustomFunctions.track_functions import tracking_track


import networkx as nx
from tqdm.auto import tqdm
from typing import Iterable, Any
import motile
from motile_toolbox.candidate_graph import graph_to_nx
from motile_toolbox.visualization.napari_utils import assign_tracklet_ids

# from CustomFunctions.metadata_funcs import get_sec
def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

############### define a bunch of functions for buildilng the motile graph and solving it
def _compute_node_frame_dict(cand_graph: nx.DiGraph) -> dict[int, list[Any]]:
    """Compute dictionary from time frames to node ids for candidate graph.

    Args:
        cand_graph (nx.DiGraph): A networkx graph

    Returns:
        dict[int, list[Any]]: A mapping from time frames to lists of node ids.
    """
    node_frame_dict: dict[int, list[Any]] = {}
    for node, data in cand_graph.nodes(data=True):
        t = data["t"]
        if t not in node_frame_dict:
            node_frame_dict[t] = []
        node_frame_dict[t].append(node)
    return node_frame_dict

def create_kdtree(cand_graph: nx.DiGraph, node_ids: Iterable[Any]) -> scipy.spatial.KDTree:
    positions = [[cand_graph.nodes[node]["x"], cand_graph.nodes[node]["y"],cand_graph.nodes[node]["z"]] for node in node_ids]
    return scipy.spatial.KDTree(positions)

def add_cand_edges(
    cand_graph: nx.DiGraph,
    max_edge_distance: float,
) -> None:
    """Add candidate edges to a candidate graph by connecting all nodes in adjacent
    frames that are closer than max_edge_distance. Also adds attributes to the edges.

    Args:
        cand_graph (nx.DiGraph): Candidate graph with only nodes populated. Will
            be modified in-place to add edges.
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes within this distance in adjacent frames will by connected
            with a candidate edge.
        node_frame_dict (dict[int, list[Any]] | None, optional): A mapping from frames
            to node ids. If not provided, it will be computed from cand_graph. Defaults
            to None.
    """
    print("Extracting candidate edges")
    node_frame_dict = _compute_node_frame_dict(cand_graph)

    frames = sorted(node_frame_dict.keys())
    prev_node_ids = node_frame_dict[frames[0]]
    prev_kdtree = create_kdtree(cand_graph, prev_node_ids)
    for frame in tqdm(frames):
        if frame + 1 not in node_frame_dict:
            continue
        next_node_ids = node_frame_dict[frame + 1]
        next_kdtree = create_kdtree(cand_graph, next_node_ids)

        matched_indices = prev_kdtree.query_ball_tree(next_kdtree, max_edge_distance)

        for prev_node_id, next_node_indices in zip(prev_node_ids, matched_indices):
            for next_node_index in next_node_indices:
                next_node_id = next_node_ids[next_node_index]
                cand_graph.add_edge(prev_node_id, next_node_id)

        prev_node_ids = next_node_ids
        prev_kdtree = next_kdtree


def solve_basic_optimization(cand_graph):
    """Set up and solve the network flow problem.

    Args:
        graph (nx.DiGraph): The candidate graph.

    Returns:
        nx.DiGraph: The networkx digraph with the selected solution tracks
    """
    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    solver = motile.Solver(cand_trackgraph)
    # solver.add_cost(
    #     motile.costs.NodeSelection(weight=-1.0, constant = 2, attribute="score")
    # )
    solver.add_cost(
        motile.costs.EdgeDistance(weight=2, constant=-20, position_attribute=("x", "y", "z"))
    )
    solver.add_cost(motile.costs.Appear(constant=2.0))
    solver.add_cost(motile.costs.Split(constant=20.0))
    solver.add_cost(motile.costs.Disappear(constant=2.0))

    solver.add_constraint(motile.constraints.MaxParents(2))
    solver.add_constraint(motile.constraints.MaxChildren(2))

    solver.solve(timeout=120)
    solution_graph = graph_to_nx(solver.get_selected_subgraph())
    return solution_graph






############### read in an example dataset from the cropped cells

direct = '//10.158.28.37/ExpansionHomesA/avlnas/LLS/crop_csvs/'

csvlist = [x for x in os.listdir(direct) if '20240805' in x][:5]



readcsvs = [pd.read_csv(direct + x) for x in csvlist]
#change some of the column names to actually readable stuff
colnames = readcsvs[0].columns.to_list()
newcolnames = dict([[c,c.split('::')[0]] for c in colnames])
timecols = [newcolnames[str(x)] for x in newcolnames if "Time" in x]
newcsvs = []
#make csvs into a pandas dataframe with useful stuff
for r in readcsvs:
    temp = r.rename(columns = newcolnames)
    #drop the units columns
    temp = temp.drop(0)
    temp['ImageCell'] = [re.findall('.*cell\d(?=-\d*)', temp.ImageDocumentName.iloc[0])[0]]*len(temp)
    temp['ImageNumber'] = [int(x.split('-')[-1]) for x in temp.ImageDocumentName]
    print(len(temp.columns))
    newcsvs.append(temp)
df = pd.concat(newcsvs)

### iterate through the cells that I followed and track cells through movies 
for i, im in df.groupby('ImageCell'):
    im['CombinedTime'] = [get_sec(x.split('T')[1][:8]) for x in im.ImageAcquisitionTime.to_list()]
    im = im.rename(columns = {'ID':'cell'})
    uni = list(im.CombinedTime.unique())
    im['t'] = [uni.index(x) for x in im.CombinedTime.to_list()]
    im = im.sort_values('CombinedTime').reset_index(drop = True)
    #add stage position to the image positions
    shifts = np.zeros((len(im.ImageNumber.unique()),3))
    for p in range(1,len(im.ImageNumber.unique())):
        ptemp = im[im.ImageNumber==p+1].copy()
        prev = im[im.ImageNumber==p].copy()
        shifts[p,2] = float(ptemp.iloc[0].ImageStageXPosition) - float(prev.iloc[0].ImageStageXPosition) 
        shifts[p,1] = float(ptemp.iloc[0].ImageStageYPosition) - float(prev.iloc[0].ImageStageYPosition)
        shifts[p,0] = float(ptemp.iloc[0].ImageFocusPosition) - float(prev.iloc[0].ImageFocusPosition)
                
    im['x'] = im.CenterX3.astype(float) + im.ImageStageXPosition0.astype(float)
    im['y'] = im.CenterY3.astype(float) + im.ImageStageYPosition0.astype(float)
    im['z'] = im.CenterZ3.astype(float) + im.ImageFocusPosition.astype(float)

    # result = tracking_track(im)


### build motile graph
cand_graph = nx.DiGraph()
print("Extracting nodes from segmentation")
count = 0
for t, chunk in tqdm(im.groupby('t')):
    for i, row in chunk.iterrows():
        attrs = {
            "t": t,
            "x": row.x,
            "y": row.y,
            "z": row.z,
            "id": row.cell,
            "image": row.ImageDocumentName,
        }
        assert count not in cand_graph.nodes
        cand_graph.add_node(count, **attrs)
        count = count + 1


# add the edges of the motile graph by distance
add_cand_edges(cand_graph, max_edge_distance=15)
#solve the motile graph
solved = solve_basic_optimization(cand_graph)
#assign labels to the tracks in the solved graph
newsolved, inter = assign_tracklet_ids(solved)
#turn the solved graph into a pandas dataframe
solveddf = pd.DataFrame.from_dict(dict(newsolved.nodes(data=True)), orient= 'index')






###############
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['blue','orange','green','red','purple','yellow','grey','black']
for i, nm in enumerate(im.ImageDocumentName.unique()[:-1]):
    temp = im[im.ImageDocumentName == nm].sort_values('t').reset_index(drop=True)
    lastemp = temp[temp.t.isin(temp.t.unique()[-5:])]
    nxt = im[im.ImageDocumentName == im.ImageDocumentName.unique()[i+1]].sort_values('t').reset_index(drop=True)
    nxtemp = nxt[nxt.t.isin(nxt.t.unique()[:5])]
    ax.scatter(temp.x, temp.y, temp.z, color = colors[int(2*i)])
    # ax.scatter(temp.x, temp.y, temp.z, color = colors[int(2*i)])
    # ax.scatter(temp.x, temp.y, temp.z, color = colors[int(2*i)])
    ax.scatter(lastemp.x, lastemp.y, lastemp.z, color = colors[int(2*i+1)])
    # ax.scatter(nxt.CenterX3.astype(float), nxt.CenterY3.astype(float), nxt.CenterZ3.astype(float), color = colors[int(2*i+1)])
ax.set_xlim3d(-100, 200)
ax.set_ylim3d(200,500)
ax.set_zlim3d(-100, 200)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['blue','orange','green','red','purple','yellow','grey','black']
ax.scatter(temp.CenterX3.astype(float), temp.CenterY3.astype(float), temp.CenterZ3.astype(float), color = colors[int(2*i)])
ax.scatter(nxt.CenterX3.astype(float), nxt.CenterY3.astype(float), nxt.CenterZ3.astype(float), color = colors[int(2*i+1)])


newshifts = np.zeros((len(im.ImageNumber.unique()), 3))
for n in range(len(im.ImageNumber.unique())-1):
    first = im[im.ImageNumber == n+1].copy()
    second = im[im.ImageNumber == n+2].copy()
    firstco = first[first.t == max(first.t.unique())-1][['CenterX3', 'CenterY3','CenterZ3']].astype(float).to_numpy()
    secondco = second[second.t == min(second.t.unique())][['CenterX3', 'CenterY3','CenterZ3']].astype(float).to_numpy()
    # for i, row in first[first.t == max(first.t.unique())].iterrows():
    firsttree = scipy.spatial.KDTree(firstco)
    dists, inds = firsttree.query(secondco)
    print(dists)
    newshifts[n+1,:] = secondco[np.argmin(dists)] - firstco[inds[np.argmin(dists)]]



############## building validation data
first = pd.read_csv('G:/All_Cell_Tracking_Info.csv', index_col=0)
second = pd.read_csv('G:/All_Cell_Tracking_Info2.csv', index_col=0)
vdf = pd.concat([first, second])
onecell = vdf[vdf.cell.str.contains('Subset-01') & vdf.cell.str.contains('20240708_488_EGFP-CAAX_640_SPY650-DNA_01perDMSO_cell1')]





############ rotate the parallelogram
######### https://stackoverflow.com/questions/70647030/what-am-i-doing-wrong-with-affine-transform-of-parallelogram-into-rectangle
origco = np.array([[0,0],[261,0],[0,145],[261,145]])
rotco = np.array([[4,1],[406,1],[324,186],[726,186]])
np.linalg.lstsq(origco, rotco)
def add_bias_term(matrix):
    return np.append(np.ones((matrix.shape[0], 1)), matrix, axis=1)
def add_interaction(matrix):
    inter = (matrix[:, 0] + matrix[:, 1]).reshape(matrix.shape[0], 1)
    return np.append(inter, matrix, axis=1)
x, _, _, _ = np.linalg.lstsq(add_bias_term(origco), rotco, rcond=None)
rotcofix = add_bias_term(origco) @ x


xyres = 0.145
zres = 0.4

origco = np.array([[0,0],[758*xyres,0],[0*xyres,250*zres],[758*xyres,250*zres]])
rotco = np.array([[3*xyres,1*xyres],[695*xyres,1*xyres],[658*xyres,379*xyres],[1349*xyres,379*xyres]])
lintran, _, _, _ = np.linalg.lstsq(add_bias_term(origco), rotco, rcond=None)
def coord_shear(co):
    #change coords
    cc = np.dot(add_bias_term(co[:,1:]), lintran)
    #swap x and y coords to be consisten with "reality"
    cc = np.insert(cc, 1, co[:,0],axis = 1)
    return cc




################# trying the coordinate translation

### iterate through the cells that I followed and track cells through movies 
for i, im in df.groupby('ImageCell'):
    im['CombinedTime'] = [get_sec(x.split('T')[1][:8]) for x in im.ImageAcquisitionTime.to_list()]
    im = im.rename(columns = {'ID':'cell'})
    uni = list(im.CombinedTime.unique())
    im['t'] = [uni.index(x) for x in im.CombinedTime.to_list()]
    im = im.sort_values('CombinedTime').reset_index(drop = True)
    #add stage position to the image positions
    shifts = np.zeros((len(im.ImageNumber.unique()),3))
    for p in range(1,len(im.ImageNumber.unique())):
        ptemp = im[im.ImageNumber==p+1].copy()
        prev = im[im.ImageNumber=p].copy()
        shifts[p,2] = float(ptemp.iloc[0].ImageStageXPosition) - float(prev.iloc[0].ImageStageXPosition) 
        shifts[p,1] = float(ptemp.iloc[0].ImageStageYPosition) - float(prev.iloc[0].ImageStageYPosition)
        shifts[p,0] = float(ptemp.iloc[0].ImageFocusPosition) - float(prev.iloc[0].ImageFocusPosition)
    transcoords = deskew(im[['CenterX3','CenterY3','CenterZ3']].astype(float).to_numpy())
    im['x'] = transcoords[:,0] #+ im.ImageStageXPosition0.astype(float)
    im['y'] = transcoords[:,1] #+ im.ImageStageYPosition0.astype(float)
    im['z'] = transcoords[:,2] #+ im.ImageFocusPosition.astype(float)




def coordtrans(co):
    x = co[:,2] + co[:,1]*np.cos(29.989)
    y = co[:,0]
    z = co[:,1]*np.sin(29.989)
    return np.column_stack((x,y,z))


scipy.spatial.distance.pdist(im[['CenterX3','CenterY3','CenterZ3']].astype(float).to_numpy()[20:22,:])

scipy.spatial.distance.pdist(transcoords[20:22,:])


def deskew(co):

    # Create a rotation object for a rotation about the x-axis
    rotation_x = scipy.spatial.transform.Rotation.from_euler('x', 60, degrees=True)
    xapp = rotation_x.apply(co)
    rotation_y = scipy.spatial.transform.Rotation.from_euler('y', 90, degrees=True)
    yapp = rotation_y.apply(xapp)
    return yapp


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['blue','orange','green','red','purple','yellow','grey','black']
for i, nm in enumerate(im.ImageDocumentName.unique()[:-1]):
    temp = im[im.ImageDocumentName == nm].sort_values('t').reset_index(drop=True)
    lastemp = temp[temp.t.isin(temp.t.unique()[-5:])]
    nxt = im[im.ImageDocumentName == im.ImageDocumentName.unique()[i+1]].sort_values('t').reset_index(drop=True)
    nxtemp = nxt[nxt.t.isin(nxt.t.unique()[:5])]
    ax.scatter(temp.x-shifts[:i+1,2].sum(), temp.y-shifts[:i+1,1].sum(), temp.z-shifts[:i+1,0].sum(), color = colors[int(2*i)])
    # ax.scatter(temp.x, temp.y, temp.z, color = colors[int(2*i)])
    # ax.scatter(temp.x, temp.y, temp.z, color = colors[int(2*i)])
    ax.scatter(lastemp.x-shifts[:i+1,2].sum(), lastemp.y-shifts[:i+1,1].sum(), lastemp.z-shifts[:i+1,0].sum(), color = colors[int(2*i+1)])
    # ax.scatter(nxt.CenterX3.astype(float), nxt.CenterY3.astype(float), nxt.CenterZ3.astype(float), color = colors[int(2*i+1)])
ax.set_xlim3d(-100, 200)
ax.set_ylim3d(200,500)
ax.set_zlim3d(-100, 200)