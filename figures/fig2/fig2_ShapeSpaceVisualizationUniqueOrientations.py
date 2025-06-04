# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:34:44 2025

@author: Aaron
"""

from paraview.simple import *
import os 
import re
import numpy as np

meshdir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/Data_and_Figs/PC_Meshes/'
meshfl = os.listdir(meshdir)

PCnum = 8
framenumber = 100
hspacing = 13.5*1.5
vspacing = 11*1.5
reconnum = 5

if PCnum%2 == 0:
    zval = PCnum/2*vspacing-vspacing/2
    zpos = np.linspace(-zval,zval,PCnum)
else:
    zval = (PCnum-1)/2*vspacing
    zpos = np.linspace(-zval,zval,PCnum)

xval = (reconnum-1)/2*hspacing
xpos = np.linspace(-xval,xval,reconnum)
binrange = list(range(1,reconnum+1))
xarr = np.stack((binrange,xpos))

perspectives = ['xy','xy','xz','xz','yz','xz','xy','xy']#,'xy','xy']

# good pink color #ffaaff
#aaaaff

view = GetActiveView()
if not view:
    # When using the ParaView UI, the View will be present, not otherwise.
    view = CreateRenderView()

for p in meshfl:
    PC = re.findall('(?<=PC)\d*', p)[0]
    reader = XMLPolyDataReader(FileName=meshdir + p)
    obj = GetRepresentation(reader)
    if perspectives[int(PC)-1] == 'xz':
        obj.Orientation = [-90,0,0]
    elif perspectives[int(PC)-1] == 'yz':
        obj.Orientation = [0,90,90]
    # obj.Opacity = 0.7
    binn = p.split('_')[-2]
    #it says zpos but array meshes in xy
    obj.Position = [xarr[1,np.where(xarr==float(binn))[1][0]],zpos[int(PC)-1],0]

#add perspective labels to each row
lower = 0.2
upper = 0.87
prange = np.arange(lower,upper,(upper-lower)/len(perspectives))
for i, per in enumerate(perspectives):
    text_source = Text()
    text_source.Text = per  # Set the text content
    # Create a text representation
    text_display = GetRepresentation(text_source)
    # Set the position in 3D space
    text_display.WindowLocation = 'Any Location'
    text_display.Position = [0.01,prange[-(i+1)]]
    text_display.FontSize = 150

#change background to white
paraview.simple._DisableFirstRenderCameraReset()
LoadPalette(paletteName='WhiteBackground')


    
view.CameraViewUp = [0, -1, -1]
view.CameraFocalPoint = [0, 0, 0]
# view.CameraViewAngle = 45
view.CameraPosition = [0,0,-375]
view.ViewSize = [4000, 5000]  
view.OrientationAxesVisibility = 0
   
Render()

WriteImage(__file__.split('.')[0]+'.png')

# def LoadMultipleFiles(FilePrefix, Low, High):
# 	#setup paraview connection
# 	from paraview.simple import *

# 	for i in range(Low,High+1):
# 		#load files named FilePrefix[Low].vtp, FilePrefix[Low+1].vtp, ..., FilePrefix[High].vtp
# 		reader = XMLPolyDataReader(FileName=FilePrefix + str(i) + '.vtk')
# 		Show(reader)
# 	Render()