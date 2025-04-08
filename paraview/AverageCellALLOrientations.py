# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:34:44 2025

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
view.CameraViewUp = [0, 1, 0]
view.CameraFocalPoint = [0, 0, 0]
view.CameraPosition = [0,0,250]
view.ViewSize = [500, 500]  


#read one of the average PC shapes
reader = XMLPolyDataReader(FileName=meshdir + 'Cell_PC1_3_Cell.vtp')
obj = GetRepresentation(reader)
# obj.Opacity = 0.7
orientations = [['xy',[0,0,0]],['xz',[-90,0,0]],['yz',[0,90,90]]]
for o in orientations:
    obj.Orientation = o[1]
        
    Render()
    
    WriteImage(savedir + f'AverageCellALLORientations_{o[0]}.png')



    

# def LoadMultipleFiles(FilePrefix, Low, High):
# 	#setup paraview connection
# 	from paraview.simple import *

# 	for i in range(Low,High+1):
# 		#load files named FilePrefix[Low].vtp, FilePrefix[Low+1].vtp, ..., FilePrefix[High].vtp
# 		reader = XMLPolyDataReader(FileName=FilePrefix + str(i) + '.vtk')
# 		Show(reader)
# 	Render()