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
import vtk
from vtk.util import numpy_support


#get some directories
curdir = os.path.dirname(os.path.abspath(__file__))


#get the mesh files from the folder
meshfl = [x for x in os.listdir(curdir) if '.vtp' in x]
#sort the list by the frame number
meshfl.sort(key=lambda x: float(re.findall('(?<=frame_)\d*', x)[0]))


metrics = [x for x in os.listdir(curdir) if '.csv' in x]
if len(metrics)>0:
    marr = pd.read_csv(curdir+'/'+metrics[0], index_col = 0)
    if 'speed' in marr.columns.to_list():
        ###scale the speed to the reconstruction size using average volume
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(curdir+'/' +meshfl[0])
        reader.Update()
        tempmesh = reader.GetOutput()
        CellMassProperties = vtk.vtkMassProperties()
        CellMassProperties.SetInputData(tempmesh)
        vol = CellMassProperties.GetVolume()
        bincol = [c for c in marr.columns.to_list() if 'Continuous_Angular_Bins' in c][0]
        binvals = np.sort(marr[bincol].unique())
        #get average speed in that bin
        avgvol = marr[marr[bincol]==binvals[0]].Cell_Volume.mean()
        scaling = (vol/avgvol) ** (1/3)
        cum_pos = 0
        
        
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
    # obj.Opacity = 0.6

    
    if 'cum_pos' in locals():
        if i == 0:
            sphere1 = Sphere(registrationName=f'Sphere')
            sphere1.Radius = avgvol/200
            sphere1.Center = [0.0, 0.0, 0.0]
            sphere1Display = GetRepresentation(sphere1)
            sphere1Display.DiffuseColor = [50/255, 127/255, 168/255]
            
        #get linear cycle bins
        bincol = [c for c in marr.columns.to_list() if 'Continuous_Angular_Bins' in c][0]
        binvals = np.sort(marr[bincol].unique())
        #get average speed in that bin
        speedint = marr[marr[bincol]==binvals[i]].speed.mean()
        #add the speed interval for this timepoint to the cumulative position based only on the speed
        cum_pos = cum_pos + (speedint * scaling)
        #### make line that helps track cell movement
        cell_mesh = servermanager.Fetch(reader)
        cell_coords = numpy_support.vtk_to_numpy(cell_mesh.GetPoints().GetData())
        centroid = cell_coords.mean(axis=0, keepdims=True)
        cell_coords -= centroid
        ##### get centroid, largest, and smallest x coordinates and length along x
        centx = cell_coords.mean(axis=0, keepdims=True)[0][0]
        backx = cell_coords[cell_coords[:,0] == np.min(cell_coords[:,0])][0][0]
        #amount to move the back to 0 x
        backmove = centx-backx

        
        #ACTUALLY MOVE THE CELL ADJUSTED FOR THE BACK AT ZERO
        obj.Position = [cum_pos+backmove, 0, 0]
        
        
        # create a new 'Cylinder'
        cylinder1 = Cylinder(registrationName=f'Cylinder{i}')
        # set active source
        SetActiveSource(cylinder1)
        # Properties modified on cylinder13
        cylinder1.Resolution = 100
        cylinder1.Height = cum_pos
        cylinder1.Radius = avgvol/300
        cylinder1.Center = [0.0, 0.0, 0.0]
        # show data in view
        cylinder1Display = GetRepresentation(cylinder1)
        # rotate cylinder to go along x-axis
        cylinder1Display.Orientation = [0.0, 0.0, 90.0]
        #after rotating move "bottom" of cylinder to origin
        cylinder1Display.Position = [cum_pos/2,0,0]
        #change the color to red
        cylinder1Display.DiffuseColor = [1.0, 0.3882, 0.2784]
        # get active source.
        acso = GetActiveSource()
        # get animation representation helper for 'a00vtp'
        rephelpcyl = GetRepresentationAnimationHelper(acso)
        # get animation track
        rephelpvistrackcyl = GetAnimationTrack('Visibility', proxy=rephelpcyl)
        
                        
                
        
    # get active source.
    SetActiveSource(reader)
    acso = GetActiveSource()
      # get animation representation helper for 'a00vtp'
    rephelp = GetRepresentationAnimationHelper(acso)
    # get animation track
    rephelpvistrackcell = GetAnimationTrack('Visibility', proxy=rephelp)
    opacityAnimationCue = GetAnimationTrack('Opacity', proxy=rephelp)
    #make key frames
    keyframes = []
    oframes = []
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
    else:
        okf3 = CompositeKeyFrame()
        okf3.KeyTime = 0
        okf3.KeyValues = [1.0]
        oframes.append(okf3)
        okf4 = CompositeKeyFrame()
        okf4.KeyTime = time + interval*len(meshfl)
        okf4.KeyValues = [0.6]
        oframes.append(okf4)

    #make the first frame visible again in the last frame
    if i ==0:
        keyFrame2 = CompositeKeyFrame()
        keyFrame2.KeyTime = time + interval*len(meshfl)
        keyFrame2.KeyValues = [1.0]
        keyFrame2.Interpolation = 'Boolean'
        keyframes.append(keyFrame2)
        print('end')
        
        okf1 = CompositeKeyFrame()
        okf1.KeyTime = 0
        okf1.KeyValues = [1.0]
        oframes.append(okf1)
        okf2 = CompositeKeyFrame()
        okf2.KeyTime = time + interval*len(meshfl)
        okf2.KeyValues = [0.6]
        oframes.append(okf2)

        
        
    # initialize the animation track
    rephelpvistrackcell.KeyFrames = keyframes
    if 'rephelpvistrackcyl' in locals():
        rephelpvistrackcyl.KeyFrames = keyframes
    if 'opacityAnimationCue' in locals():
        opacityAnimationCue.KeyFrames = oframes
    time = time + interval
    
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
if 'cum_pos' in locals():
    view.CameraPosition = [(cum_pos+backmove)/2, 0, -400]
    view.CameraFocalPoint = [(cum_pos+backmove)/2, 0, 0]
else:
    view.CameraPosition = [0, 0, -400]
view.ViewSize = [500, 500]  
view.OrientationAxesVisibility = 1
view.UseColorPaletteForBackground = 0
view.Background = [84/255, 94/255, 135/255]

# save animation
SaveAnimation(curdir+'/PC_mesh_animation julie version.mp4', view, ImageResolution=[1000, 1000], FrameRate=10)#, ImageResolution=[788, 364])


