# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:32:45 2022

@author: Aaron
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import os
import pandas as pd
import re
import math

# parameter boundaries
gridSize =  200
# range of persistence
qBound = [-1.5, 1.5]
# range of activity (micron/sec)
aBound = [0.0, 0.5]
# likelihood function
# parameter grid (excluding boundary values)
qGrid  = (np.array([np.linspace(qBound[0], qBound[1], gridSize+2)[1:-1]]*gridSize)).T
aGrid  = (np.array([np.linspace(aBound[0], aBound[1], gridSize+2)[1:-1]]*gridSize))
a2Grid = (np.array([np.linspace(aBound[0], aBound[1], gridSize+2)[1:-1]]*gridSize))**2

# algorithm parameters
pMin = 1.0*10**(-4)
Rq = 2
Ra = 2

# compute likelihood on parameter grid
def compLike(vp,v):
    return np.exp(-((v[0] - qGrid*vp[0])**2 + (v[1] - qGrid*vp[1])**2 + (v[2] - qGrid*vp[2])**2)/(2*a2Grid) - (3/2)*np.log(2*np.pi*a2Grid))

# compute new prior
def compNewPrior(oldPrior,like):
    # compute posterior distribution
    post = oldPrior*like
    post /= np.sum(post)

    # use posterior as a starting point to create new prior
    newPrior = post

    # introduce minimal probability
    mask = newPrior < pMin
    newPrior[mask] = pMin

    # apply boxcar filter
    ker = np.ones((2*Rq + 1, 2*Ra+1))/((2*Rq+1)*(2*Ra+1))

    newPrior = convolve2d(newPrior, ker, mode='same', boundary='symm')

    return newPrior

# compute sequence of posterior distributions for a sequence of measured velocities
def compPostSequ(uList):
    # initialize array for posterior distributions
    dist = np.empty((len(uList),gridSize,gridSize))

    # initialize flat prior
    dist[0].fill(1.0/(gridSize**2))

    # forward pass (create forward priors for all time steps)
    for i in np.arange(1,len(uList)):
        dist[i] = compNewPrior(dist[i-1], compLike(uList[i-1], uList[i]))

    # backward pass
    backwardPrior = np.ones((gridSize,gridSize))/(gridSize**2)
    for i in np.arange(1,len(uList))[::-1]:
        # re-compute likelihood
        like = compLike(uList[i-1], uList[i])

        # forward prior * likelihood * backward prior
        dist[i] = dist[i-1]*like*backwardPrior
        dist[i] /= np.sum(dist[i])

        # generate new backward prior for next iteration
        backwardPrior = compNewPrior(backwardPrior, compLike(uList[i-1], uList[i]))

    # drop initial flat prior before return
    return dist[1:]

# compute posterior mean values from a list of posterior distributions
def compPostMean(postSequ):
    qMean = [np.sum(post*qGrid) for post in postSequ]
    aMean = [np.sum(post*aGrid) for post in postSequ]

    return np.array([qMean,aMean])



# # parameter boundaries
# gridSize = 200
# # range of persistence
# qBound = [-1.5, 1.5]
# # range of activity (micron/sec)
# aBound = [0.0, 0.5]

# # algorithm parameters
# # pMin = 1.0*10**(-7)
# pMin = 1.0*10**(-5)
# Rq   = 1
# Ra   = 1


def get_pa(
    df,
    interval,
    ):
    
    
    
    dx = np.zeros([len(df)-1])
    dy = np.zeros([len(df)-1])
    dz = np.zeros([len(df)-1])
    
    #And then for the actual  inference, for a single time series, plus in the x, y, and x speed values
    for i, t in enumerate(np.arange(df.frame.min(),df.frame.max())):
       dx[i] = df[df.frame==t+1].x.values - \
                        df[df.frame==t].x.values
    
       dy[i] = df[df.frame==t+1].y.values - \
                        df[df.frame==t].y.values
    
       dz[i] = df[df.frame==t+1].z.values - \
                        df[df.frame==t].z.values
    
    # Now lets begin Bayesian analysis of cell track
    veloSequ1 = np.zeros([len(dx), 3])
    veloSequ1[:,0] = dx/interval # um/sec
    veloSequ1[:,1] = dy/interval # um/sec
    veloSequ1[:,2] = dz/interval # um/sec
            
    postSequ1 = compPostSequ(veloSequ1)
    meanPost1 = compPostMean(postSequ1)
    
    persistence = meanPost1[0]
    activity = meanPost1[1]
    speed = pd.Series(dx**2 + dy**2 + dz**2).apply(lambda x: math.sqrt(x))/interval
    
    return persistence, activity, speed.to_numpy()



def get_pa_drift(
    df,
    interval,
    ):
    
    
    
    dx = np.zeros([len(df)-1])
    dy = np.zeros([len(df)-1])
    dz = np.zeros([len(df)-1])
    
    #And then for the actual  inference, for a single time series, plus in the x, y, and x speed values
    for i, t in enumerate(np.arange(df.frame.min(),df.frame.max())):
       dx[i] = df[df.frame==t+1].x_adj.values - \
                        df[df.frame==t].x_adj.values
    
       dy[i] = df[df.frame==t+1].y_adj.values - \
                        df[df.frame==t].y_adj.values
    
       dz[i] = df[df.frame==t+1].z_adj.values - \
                        df[df.frame==t].z_adj.values
    
    # Now lets begin Bayesian analysis of cell track
    veloSequ1 = np.zeros([len(dx), 3])
    veloSequ1[:,0] = dx/interval # um/sec
    veloSequ1[:,1] = dy/interval # um/sec
    veloSequ1[:,2] = dz/interval # um/sec
            
    postSequ1 = compPostSequ(veloSequ1)
    meanPost1 = compPostMean(postSequ1)
    
    persistence = meanPost1[0]
    activity = meanPost1[1]
    speed = pd.Series(dx**2 + dy**2 + dz**2).apply(lambda x: math.sqrt(x))/interval
    
    return persistence, activity, speed.to_numpy()



def velocity_and_distance(df, #data frame with at x, y, and z positions
                        interval, #time interval between data points
                        signal_vector = np.array([0,0,0]), #vector of directional cue, must be a unit vector in XYZ format
                        ):
    #if signal vector provided use that, otherwise use endpoint of migration as "signal"
    if not (signal_vector == np.zeros((1,3))).all():
        totaldisplacement = df.iloc[-1][['x','y','z']].to_numpy() - df.iloc[0][['x','y','z']].to_numpy()
        signal_vector = totaldisplacement/np.linalg.norm(totaldisplacement)
    
    # get "velocity" of cell towards a signal
    # aka distance travelled in the direction of the signal over time
    sds = []
    isvs = []
    track_length = 0
    total_distance = []
    euclidean_distance = []
    for l in range(len(df)-1):
        #signal displacement
        totaldisplacement = df.iloc[l+1][['x','y','z']].to_numpy() - df.iloc[0][['x','y','z']].to_numpy()
        sd = np.dot(totaldisplacement, signal_vector)
        sds.append(sd/(interval*(l+1)))
        #get euclidean distance from start at each time point
        euclidean = math.sqrt(totaldisplacement[0]**2 + totaldisplacement[1]**2 + totaldisplacement[2]**2)
        euclidean_distance.append(euclidean)
        #instantaneous signal velocity
        instantdisplacement = df.iloc[l+1][['x','y','z']].to_numpy() - df.iloc[l][['x','y','z']].to_numpy()
        isv = np.dot(instantdisplacement, signal_vector)
        isvs.append(isv/(interval))
        #get instantaneous distance travelled to sum together
        instantdistance = math.sqrt(instantdisplacement[0]**2 + instantdisplacement[1]**2 + instantdisplacement[2]**2)
        #sum total distance travelled along cell path so far
        track_length = track_length + instantdistance
        total_distance.append(track_length)
        
    return isvs, sds, total_distance, euclidean_distance



def DA_3D(
        #get the 3D directional autocorrelation
        pos: np.array,  # x,y,z positions in shape (N,3)
        lag: float = 1, # time interval over which to take the DA
        ):
    # Ensure lag is a valid positive integer
    if lag < 1 or lag >= len(pos):
        raise ValueError("Lag must be a positive integer less than the length of the coordinates")

    traj = np.zeros((len(pos)-lag,3))
    traj[:,0] = pos[lag:,0] - pos[:-lag,0]
    traj[:,1] = pos[lag:,1] - pos[:-lag,1]
    traj[:,2] = pos[lag:,2] - pos[:-lag,2]
    # Normalize vectors to get unit direction vectors
    unitvecs = traj/np.linalg.norm(traj, axis = 1)[:, np.newaxis]
    # Calculate dot products of consecutive unit vectors with the given lag
    dot_products = np.sum(unitvecs[:-lag] * unitvecs[lag:], axis=1)
    # Create an array with NaN values and insert the dot products in the correct positions
    DA = np.full(len(pos), np.nan)
    DA[lag+1:] = dot_products
    
    return DA


