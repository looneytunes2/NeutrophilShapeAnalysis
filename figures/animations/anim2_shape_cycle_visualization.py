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
from matplotlib import cm

#get some directories
curdir = os.path.dirname(os.path.abspath(__file__))

##### load all of the data to get average speeds
whichpcs = [1,7]
treatments = ['Random']
binrange = 20
#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
aerdir = basedir + 'random/'
meshdir = curdir + f'/PC{whichpcs[0]}-PC{whichpcs[1]}_Cycle_AllSHCoeff_Visualization/{treatments[0]}/'

FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
#restrict dataframe to only random experiments
TotalFrame = FullFrame[FullFrame.Treatment.isin(treatments)]
#load aer data
aerdf = pd.read_csv(aerdir + f'PC{whichpcs[0]}-PC{whichpcs[1]}_raw_transition_aer_cf.csv', index_col = 0)
TotalFrame = TotalFrame.merge(aerdf, on = 'cell')

#average degrees per second in each bin
avgcf = TotalFrame.angular_velocity.mean()
#average number of seconds spent in each bin
avgtime = binrange / avgcf

#get the average speeds based on the angular bins
angbins = pd.read_csv(meshdir+ 'linear_cycle_data.csv', index_col = 0)
TotalFrame = TotalFrame.merge(angbins, on = 'cell')
displacements = (TotalFrame.groupby(f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins').speed.mean()*avgtime).reset_index()
displacements = pd.concat((displacements, displacements.iloc[[0]]), ignore_index = True)

#get the mesh files from the folder
meshfl = [x for x in os.listdir(meshdir) if '.vtp' in x]
#sort the list by the frame number
meshfl.sort(key=lambda x: float(re.findall('(?<=frame_)\d*', x)[0]))
#add the first mesh to the end of the list
meshfl.append(meshfl[0])
        
# get animation scene and make it at least the number of frames that I have meshes
animationScene1 = GetAnimationScene()
animationScene1.NumberOfFrames = len(meshfl)



#add one to the number of meshes rendered to include the first again at the end
posnum = int(len(meshfl))

#define the colors to make the meshes
cmap = cm.get_cmap('twilight')
discrete_colors = cmap(np.linspace(0,1,posnum-1))
discrete_colors = np.vstack((discrete_colors, discrete_colors[0]))

#change background to white
paraview.simple._DisableFirstRenderCameraReset()
LoadPalette(paletteName='WhiteBackground')


view = GetActiveView()
if not view:
    # When using the ParaView UI, the View will be present, not otherwise.
    view = CreateRenderView()
    
view.CameraViewUp = [0, -1, -1]
view.CameraFocalPoint = [0, 0, 0]
view.CameraPosition = [0, 0, -150]
view.ViewSize = [4000, 4000]  
view.OrientationAxesVisibility = 0



############# SCALE BAR
slen = 5
sx = -1
sy = 10
# 10um line scalebar
line = Line(Point1=[sx, sy, 0], Point2=[sx+slen, sy, 0])
# Apply a Tube filter to give it thickness
tube = Tube(Input=line)
tube.Radius = 0.25  # Adjust thickness as needed
tube.NumberofSides = 20  # Makes it smoother
# Show the tube in the active view
tube_display = Show(tube)
#change color
tube_display.DiffuseColor = [0, 0, 0]

####### scale bar label
text_source = Text()
text_source.Text = '5 μm'  # Set the text content
# Create a text representation
text_display = GetRepresentation(text_source)
# Set the position in 3D space
text_display.WindowLocation = 'Any Location'
text_display.Position = [0.145,0.28]
text_display.FontSize = 120
text_display.Bold = 1

time = 0
cum_pos = 0
interval = 1/len(meshfl)
for i, p in enumerate(meshfl):
    #get time point
    # time = marr.arbitrarytime.values[i]
    #open time point mesh
    reader = XMLPolyDataReader(FileName=meshdir+'/' + p)
    obj = GetRepresentation(reader)
    # obj.Opacity = 0.6

    
    #ACTUALLY MOVE THE CELL ADJUSTED FOR THE BACK AT ZERO
    obj.Position = [cum_pos, 0, 0]
    cum_pos = cum_pos + displacements.speed.iloc[i]
    
    #change the color to the ambient color
    obj.AmbientColor = discrete_colors[i,:-1]
    obj.DiffuseColor = discrete_colors[i,:-1]
                
        
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



        
        
    # initialize the animation track
    rephelpvistrackcell.KeyFrames = keyframes
    if 'rephelpvistrackcyl' in locals():
        rephelpvistrackcyl.KeyFrames = keyframes
    if 'opacityAnimationCue' in locals():
        opacityAnimationCue.KeyFrames = oframes
    time = time + interval
    



#set the camera focus in the middle of the cell path
view.CameraPosition = [(cum_pos)/2, 0, -110]
view.CameraFocalPoint = [(cum_pos)/2, 0, 0]


# save animation
SaveAnimation(__file__.split('.')[0] + '.mp4', view, ImageResolution=[4000, 4000], FrameRate=2)#, ImageResolution=[788, 364])


