# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 15:03:44 2023

@author: Aaron
"""
from scipy.spatial import distance
import math
import pandas as pd

longdistmatrix = distance.pdist(traj)
shortdistmatrix = distance.squareform(longdistmatrix)
shortdistmatrix = np.array(shortdistmatrix)


t_int = 15 #sec

x1 = [2,4,6,8,5,7,3]
x2 = [3,5,1,6,7,3,5]
traj = np.vstack([x1,x2]).T
ttot = len(traj)*t_int
#interpolate based on path
tck, b = interpolate.splprep(traj.T)
#time between frames normalized between 0 and 1
int_int = 1/(len(traj)-1)
        
#measure the trajectory and interpolate evenly by distance
interlist = []
for t in range(len(traj)-1):
    print(t*int_int)
    di = distance.pdist([traj[t,:],traj[t+1,:]])[0]
    intt = round(di/0.1)
    interpoints = np.linspace(start=t*int_int, stop = t*int_int+int_int, num = intt, endpoint = False)
    x, y = interpolate.splev(interpoints,tck)
    x = [math.floor(i) for i in x]
    y = [math.floor(i) for i in y]
    interlist.append(np.stack([x,y,interpoints]).T)

#add last position
interlist.append(np.array([[x1[-1], x2[-1], 1]]))
#concatenate all
fulltr = pd.DataFrame(np.concatenate(interlist), columns=['x','y','t'])

#find all single move transitions
trans = []
prev = pd.Series([traj[0,0],traj[0,1],0], index=['x','y','t'])
for i, g in fulltr.diff().iterrows():
    #provide an escape if the interpolation is still not good enough and there
    #is a >1 jump in the trajectory
    if (abs(g.x)>=1) and (abs(g.y)>=1):
        print('hey')
        extra = np.linspace(fulltr.iloc[i-1].t,fulltr.iloc[i].t,30)
        ex, ey = interpolate.splev(extra,tck)
        ex = [math.floor(i) for i in ex]
        ey = [math.floor(i) for i in ey]
        exdf = pd.DataFrame(np.stack([ex,ey,extra]).T, columns=['x','y','t'])
        for h, j in exdf.diff().iterrows():
            if (abs(j.x)==1) or (abs(j.y)==1):
                cur = exdf.iloc[h]
                trans.append([prev.x, prev.y, cur.x, cur.y, cur.t-prev.t])
                prev = exdf.iloc[h].copy()
    #collect all of the 1 moves
    elif (abs(g.x)==1) or (abs(g.y)==1):
        cur = fulltr.iloc[i]
        trans.append([prev.x, prev.y, cur.x, cur.y, cur.t-prev.t])
        prev = fulltr.iloc[i].copy()
    #ignore timepoints that don't transition
    else:
        pass

#combine the data
alltrans = pd.DataFrame(trans, columns=['from_x', 'from_y', 'to_x', 'to_y', 'time_elapsed'])
#double check for bad 
any((abs(alltrans.from_x-alltrans.to_x) + abs(alltrans.from_y-alltrans.to_y))!=1)