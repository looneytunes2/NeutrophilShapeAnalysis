# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:11:34 2022

@author: Aaron
"""

from paraview.simple import *
import os 
import re
import numpy as np

datadir = re.findall('.*(?=Data_and_Figs)', __file__)[0]
meshdir = datadir + 'PC_Meshes/'
meshfl = os.listdir(meshdir)
savedir = re.findall(f'.*(?={os.path.basename(__file__)})', __file__)[0]
PCnum = 10
framenumber = 100
hspacing = 17*3.5
vspacing = 12*3.5
reconnum = 5

if PCnum%2 == 0:
    zval = PCnum/2*vspacing-vspacing/2
    zpos = np.linspace(zval,-zval,PCnum)
else:
    zval = (PCnum-1)/2*vspacing
    zpos = np.linspace(zval,-zval,PCnum)

xval = (reconnum-1)/2*hspacing
xpos = np.linspace(-xval,xval,reconnum)
binrange = list(range(1,reconnum+1))
xarr = np.stack((binrange,xpos))


# good pink color #ffaaff
#aaaaff

for p in meshfl:
    PC = re.findall('(?<=PC)\d*', p)[0]
    reader = XMLPolyDataReader(FileName=meshdir + p)
    obj = GetRepresentation(reader)
    # obj.Opacity = 0.7
    binn = p.split('_')[-2]
    obj.Position = [xarr[1,np.where(xarr==float(binn))[1][0]],0,zpos[int(PC)-1]]


#change background to white
paraview.simple._DisableFirstRenderCameraReset()
LoadPalette(paletteName='WhiteBackground')


view = GetActiveView()
if not view:
    # When using the ParaView UI, the View will be present, not otherwise.
    view = CreateRenderView()
    
view.CameraViewUp = [0, 0, 1]
view.CameraFocalPoint = [0, 0, 0]
# view.CameraViewAngle = 45
view.CameraPosition = [0,-600,0]
view.ViewSize = [4000, 6000]  
view.OrientationAxesVisibility = 0
   
Render()

WriteImage(savedir + 'ShapeSpaceVisualization.png')

# def LoadMultipleFiles(FilePrefix, Low, High):
# 	#setup paraview connection
# 	from paraview.simple import *

# 	for i in range(Low,High+1):
# 		#load files named FilePrefix[Low].vtp, FilePrefix[Low+1].vtp, ..., FilePrefix[High].vtp
# 		reader = XMLPolyDataReader(FileName=FilePrefix + str(i) + '.vtk')
# 		Show(reader)
# 	Render()