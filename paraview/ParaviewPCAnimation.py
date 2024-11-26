# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:15:45 2023

@author: Aaron
"""

from paraview.simple import *
import os 
import re
import numpy as np
import pandas as pd

#get some directories
curdir = os.path.dirname(os.path.abspath(__file__))

metrics = [x for x in os.listdir(curdir) if '.csv' in x][0]
marr = pd.read_csv(curdir+'/'+metrics, index_col = 0)
if 'speed' in marr.columns.to_list():
    diffs = marr.speed.diff()
    diffinds = diffs[diffs!=0].reset_index()
    cum_pos = 0
    
#get the mesh files from the folder
meshfl = [x for x in os.listdir(curdir) if '.vtp' in x]
#sort the list by the frame number
meshfl.sort(key=lambda x: float(x.split('.vtp')[0]))

# get animation scene and make it at least the number of frames that I have meshes
animationScene1 = GetAnimationScene()
animationScene1.NumberOfFrames = len(meshfl)

time = 0
interval = 1/len(meshfl)
for i, p in enumerate(meshfl):
    #get time point
    # time = marr.arbitrarytime.values[i]
    #open time point mesh
    reader = XMLPolyDataReader(FileName=curdir+'/' + p)
    obj = GetRepresentation(reader)
    obj.Opacity = 0.6



    if 'speed' in marr.columns.to_list():
        if i == diffinds['index'].max():
            pass
        elif i in diffinds['index'].to_list():
            curdifind = diffinds[diffinds['index']==i].index[0]
            #how many interpolated "timepoints" spend time in this bin position
            tpts = diffinds.iloc[curdifind+1]['index'] - i
            #get the fraction of the speed travelled per timepoint spent in this bin position
            speedint = marr.speed[i]/tpts
        #add the speed interval for this timepoint to the cumulative position
        cum_pos = cum_pos + speedint
        #change the x-position to the cumulative distance travelled so far
        obj.Position = [cum_pos, 0, 0]

    # get active source.
    acso = GetActiveSource()
    
     # get animation representation helper for 'a00vtp'
    rephelp = GetRepresentationAnimationHelper(acso)
    
    
    # get animation track
    # rephelpvistrack = GetAnimationTrack('Visibility', index=0, proxy=rephelp)
    rephelpvistrack = GetAnimationTrack('Visibility', proxy=rephelp)
    
    #make key frames
    keyframes = []
    #make inivisible at first, unless it's the first frame
    if time != 0:
        # make mesh visible at the appropriate time
        keyFrame0 = CompositeKeyFrame()
        keyFrame0.KeyTime = 0.0
        keyFrame0.KeyValues = [0.0]
        keyFrame0.Interpolation = 'Boolean'
        keyframes.append(keyFrame0)
        
        
    # make mesh visible at the appropriate time
    keyFrame1 = CompositeKeyFrame()
    keyFrame1.KeyTime = time
    keyFrame1.KeyValues = [1.0]
    keyFrame1.Interpolation = 'Boolean'
    keyframes.append(keyFrame1)
    
    # make the mesh invisible at the appropriate time except for the last frame
    if i != len(meshfl)-1:
        keyFrame2 = CompositeKeyFrame()
        keyFrame2.KeyTime = time + interval
        keyFrame2.KeyValues = [0.0]
        keyFrame2.Interpolation = 'Boolean'
        keyframes.append(keyFrame2)
        
    # initialize the animation track
    rephelpvistrack.KeyFrames = keyframes
    
    time = time + interval
    # Hide(acso)
    
# #change background to white
# paraview.simple._DisableFirstRenderCameraReset()
# LoadPalette(paletteName='WhiteBackground')


view = GetActiveView()
if not view:
    # When using the ParaView UI, the View will be present, not otherwise.
    view = CreateRenderView()
    
view.CameraViewUp = [0, -1, 0]
view.CameraFocalPoint = [0, 0, 0]
# view.CameraViewAngle = 180
view.CameraPosition = [0, 0, -300]
view.ViewSize = [500, 500]  
view.OrientationAxesVisibility = 1
view.UseColorPaletteForBackground = 0
view.Background = [84/255, 94/255, 135/255]

# save animation
SaveAnimation(curdir+'/PC_mesh_animation.avi', view, FrameRate=30)#, ImageResolution=[788, 364])


