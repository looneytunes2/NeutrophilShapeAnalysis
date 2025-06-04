# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:34:44 2025

@author: Aaron
"""

from paraview.simple import *
import os 
import re
import numpy as np

curdir = os.path.dirname(__file__)
avg_recon = curdir + '/average_cell_mesh.vtp'

# good pink color #ffaaff
#aaaaff

view = GetActiveView()
if not view:
    # When using the ParaView UI, the View will be present, not otherwise.
    view = CreateRenderView()

#change background to white
paraview.simple._DisableFirstRenderCameraReset()
LoadPalette(paletteName='WhiteBackground')
#change camera stuff
view.CameraViewUp = [0, -1, -1]
view.CameraFocalPoint = [0, 0, 0]
view.CameraPosition = [0,0,-50]
view.ViewSize = [1000, 1000]  


#read one of the average PC shapes
reader = XMLPolyDataReader(FileName=avg_recon)
obj = GetRepresentation(reader)
# obj.Opacity = 0.7
orientations = [['xy',[0,0,0]],['xz',[-90,0,0]],['yz',[0,90,0]]]
for o in orientations:
    obj.Orientation = o[1]
    
    Render()
    
    WriteImage(__file__.split('.')[0] + f'_{o[0]}.png')



    

# def LoadMultipleFiles(FilePrefix, Low, High):
# 	#setup paraview connection
# 	from paraview.simple import *

# 	for i in range(Low,High+1):
# 		#load files named FilePrefix[Low].vtp, FilePrefix[Low+1].vtp, ..., FilePrefix[High].vtp
# 		reader = XMLPolyDataReader(FileName=FilePrefix + str(i) + '.vtk')
# 		Show(reader)
# 	Render()