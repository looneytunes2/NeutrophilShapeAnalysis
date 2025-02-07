# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:32:17 2025

@author: Aaron
"""

from paraview.simple import *
import os 
import re
import numpy as np
import pandas as pd
import vtk
from vtk.util import numpy_support
from itertools import groupby

#get some directories
curdir = os.path.dirname(os.path.abspath(__file__))


#get the mesh files from the folder
meshfl = [x for x in os.listdir(curdir) if '.vtp' in x]
#sort the list by the frame number
meshfl.sort(key=lambda x: float(re.findall('(?<=frame_)\d*', x)[0]))

#open the position and euler angle dataframe
df = pd.read_csv(curdir+'/cell_info.csv', index_col = 0)

#use average volume in the cell info and the volume of the first recon to
#establish scaling
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(curdir+'/' +meshfl[0])
reader.Update()
tempmesh = reader.GetOutput()
CellMassProperties = vtk.vtkMassProperties()
CellMassProperties.SetInputData(tempmesh)
vol = CellMassProperties.GetVolume()
#get average speed in that bin
avgvol = df.Cell_Volume.mean()
scaling = (vol/avgvol) ** (1/3)


distlist = []
for c in range(1,len(cum_pos)):
    distlist.append(np.sqrt(np.sum((cum_pos[c]-cum_pos[c-1])**2)))

#get the cumulative position but take in to account different coordinates
#between movies
if 'Subset' in df.cell.iloc[0]:
    df['movie'] = [d.split('-Subset')[0] for d in df.cell.to_list()]
    df['frame'] = [int(d.split('frame_')[-1]) for d in df.cell.to_list()]
    cum_pos = np.zeros((len(df),3))
    ind = 0
    for m, mov in df.groupby('movie'):
        mov = mov.sort_values('frame').reset_index(drop=True)

        #get displacements
        tempc = mov[['x_raw','y_raw','z_raw']].diff().values
        if m == df.movie.unique()[1]:
            print(mov[['x','y','z']])
        #replace gaps with zeros
        jumpind = mov.frame.diff()[mov.frame.diff()!=1].index.to_list()
        print(jumpind)
        tempc[jumpind,:] = np.zeros((len(jumpind),3))
        cum_pos[ind:ind+len(tempc),:] = tempc * scaling
        ind = ind + len(tempc)
    #actually do cumulative sum
    cum_pos = np.cumsum(cum_pos, axis = 0) 
else:
    cum_pos = df[['x','y','z']].diff().cumsum().values * scaling
    cum_pos[0,:] = [0,0,0]
       
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
    
    
    row = df.iloc[i]
    eulers = -row[['Euler_angles_X','Euler_angles_Y','Euler_angles_Z']].values
    obj.Orientation = list(eulers)
    #ACTUALLY MOVE THE CELL
    if i != 0:
        obj.Position = list(cum_pos[i])
           
                
        
    # get active source.
    SetActiveSource(reader)
    acso = GetActiveSource()
      # get animation representation helper for 'a00vtp'
    rephelp = GetRepresentationAnimationHelper(acso)
    # get animation track
    rephelpvistrackcell = GetAnimationTrack('Visibility', proxy=rephelp)
    
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
    rephelpvistrackcell.KeyFrames = keyframes
    
    time = time + interval
    
# #change background to white
# paraview.simple._DisableFirstRenderCameraReset()
# LoadPalette(paletteName='WhiteBackground')


view = GetActiveView()
if not view:
    # When using the ParaView UI, the View will be present, not otherwise.
    view = CreateRenderView()
    
view.CameraViewUp = [0, -1, 0]
# view.CameraFocalPoint = [0, 0, 0]
# view.CameraViewAngle = 180

view.CameraPosition = list(np.mean(cum_pos,axis =0)-[0, 0, -2000])
view.CameraFocalPoint = list(np.mean(cum_pos,axis =0))

view.ViewSize = [500, 500]  
view.OrientationAxesVisibility = 1
view.UseColorPaletteForBackground = 0
view.Background = [84/255, 94/255, 135/255]

# save animation
SaveAnimation(curdir+'/SHrecon_mesh_animation.mp4', view, ImageResolution=[1000, 1000], FrameRate=10)#, ImageResolution=[788, 364])


