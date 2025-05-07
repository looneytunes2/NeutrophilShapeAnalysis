# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 11:36:29 2025

@author: Aaron
"""

import numpy as np
import pandas as pd
import os
from itertools import groupby
from operator import itemgetter
from matplotlib.animation import FuncAnimation 
import matplotlib.pyplot as plt
from scipy import interpolate
import pickle as pk

#time interval for movies
time_interval = 10
whichpcs = [1,7]

#get some directories
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
infodir = basedir+'processed_data/'
cellname = '20231116_488EGFP-CAAX_3mA_37C_1_cell_42'

#### open the dataframe with all of the PCs etc.
TotalFrame = pd.read_csv(datadir + 'Shape_Metrics_Galvanotaxis_Confocal_40x_37C_10s.csv', index_col=0)
centers = pd.read_csv(datadir + 'PC_bin_centers.csv', index_col=0)
nbins = len(centers)
    
#get all the position and trajectory info
df = []
for x in os.listdir(infodir):
    if cellname in x:
        df.append(pd.read_csv(infodir+x, index_col = 0))
df = pd.concat(df).sort_values('frame').reset_index(drop=True)
#### crop df to just the frames that you want
trimdf = df.iloc[72:121]
#combine this cropped dataframe with PCs
trimdf = trimdf.merge(TotalFrame[[x for x in TotalFrame.columns.to_list() if x not in trimdf.columns.to_list()]+['cell']],
                      left_on='cell', right_on='cell')
###get the actual PC values and the bins
# open confocal pca model
pca = pk.load(open(datadir+"pca.pkl",'rb')) 
# tranform the LLS data
trim_coeff = trimdf[[x for x in trimdf.columns.to_list() if 'shcoeffs' in x]]
trim_transform = pca.transform(trim_coeff)
# Dataframe of transformed variable
pc_names = [f"PC{c}" for c in range(1, 1 + 10)]
df_trans = pd.DataFrame(data=trim_transform, columns=pc_names, index = trim_coeff.index)

for w in whichpcs:
    pc = f'PC{w}'
    bin_edges = np.insert((centers[pc].iloc[1:]-centers[pc].diff().iloc[-1]/2).values,[0,nbins-1],[-np.inf,np.inf])
    # Aplly digitization
    trimdf[f"PC{w}bins"] = np.digitize(df_trans[f'PC{w}'], bin_edges)


# ############## explore different cells ####################
# #find the length of cell consecutive frames
# results = []
# for i, cells in TotalFrame.groupby('CellID'):
#     cells = cells.sort_values('frame').reset_index(drop = True)
#     runs = list()
#     #######https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
#     for k, g in groupby(enumerate(cells['frame']), lambda ix: ix[0] - ix[1]):
#         currentrun = list(map(itemgetter(1), g))
#         list.append(runs, currentrun)
#     maxrun = max([len(l) for l in runs])
#     actualrun = max(runs, key=len, default=[])
#     results.append([i, maxrun, actualrun])
# #find
# stdf = pd.DataFrame(results, columns = ['CellID','length_of_run','actual_run']).sort_values('length_of_run', ascending=False).reset_index(drop=True)
# stdf.head(30)

# #### select cell from list above
# row = stdf.loc[4]
# print(row.CellID)
# #get the data related to this run of this cell
# data = TotalFrame[(TotalFrame.CellID==row.CellID) & (TotalFrame.frame.isin(row.actual_run))].sort_values('frame').reset_index(drop=True)



#get the interpolated points just in case I haven't calculated them yet
bintraj = trimdf[[f'PC{whichpcs[0]}bins',f'PC{whichpcs[1]}bins']].to_numpy()
#find the indicies of the duplicates
duplicates = [i for i,w in enumerate(bintraj) if all(w==bintraj[i-1])]
#add a small number to the duplicates so they're not the same, but not meaningfully different
for d in duplicates:
    bintraj[d,:] = bintraj[d,:]+0.001

#interpolate based on path
tck, b = interpolate.splprep(bintraj.T, u=range(len(bintraj)),k=1, s=0)
interpoints = np.linspace(start=0, stop = len(bintraj), num = 20*len(bintraj), endpoint = False)
interlist = interpolate.splev(interpoints,tck)
interarray = np.array(interlist).T





############# make animated fig
fig, ax = plt.subplots(figsize = (5,5))

#add "grid lines" first 
for h in np.linspace(0.5, nbins+0.5, nbins+1):
    ax.axhline(h, linestyle='-', color='grey', alpha=0.3) # horizontal lines
    ax.axvline(h, linestyle='-', color='grey', alpha=0.3) # vertical lines
    
    
ax.set_aspect("equal")
ax.set_xlabel(f'PC{whichpcs[0]}', fontsize = 12)
ax.set_ylabel(f'PC{whichpcs[1]}', fontsize = 12)
ax.set_xticks(list(range(1,nbins+1)),[round(x,1) for x in centers[f'PC{whichpcs[0]}'].to_list()], fontsize = 9)
ax.set_yticks(list(range(1,nbins+1)),[round(x,1) for x in centers[f'PC{whichpcs[1]}'].to_list()], fontsize = 9)
ax.set_xlim(0,nbins+1)
ax.set_ylim(0,nbins+1)
# ax.set_title(mm, fontsize = 30)


# create a point in the axes
point, = ax.plot([],[], marker="o", color = '#eb4034', markersize = 6, zorder = 2)
line, = ax.plot([],[], lw = 2.5, zorder = 1)
# make function for updating point position
def animate(i, interarray):
    point.set_data([interarray[i,0]], [interarray[i,1]])
    line.set_data(interarray[:i+1, 0], interarray[:i+1, 1])
    return point, line

ani = FuncAnimation(fig, animate, blit=True, interval=0, 
                    frames=len(interarray), fargs = (interarray,))
# plt.show()


ani.save(__file__.split('.')[0] + '_' + cellname + '.mp4', writer='ffmpeg', fps=40)#, dpi = 300, bitrate = 5000)#, codec = 'libx264') #,extra_args=['-vcodec', 'libx264'])
