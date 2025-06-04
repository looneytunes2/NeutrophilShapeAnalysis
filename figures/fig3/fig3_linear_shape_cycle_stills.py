# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:15:45 2023

@author: Aaron
"""

from paraview.simple import *
import os 
import re
import numpy as np
from matplotlib import cm


#get some directories
curdir = os.path.dirname(os.path.abspath(__file__))
meshdir = curdir+'/PC1-PC7_Cycle_AllSHCoeff_Visualization/Random/'
# curdir = 'C:/Users/Aaron/NeutrophilShapeAnalysis/figures/fig2/'

#get the mesh files from the folder
meshfl = [x for x in os.listdir(meshdir) if '.vtp' in x]
#sort the list by the frame number
meshfl.sort(key=lambda x: float(re.findall('(?<=frame_)\d*', x)[0]))

#add one to the number of meshes rendered to include the first again at the end
posnum = int(len(meshfl)+1)

#define the colors to make the meshes
cmap = cm.get_cmap('twilight')
discrete_colors = cmap(np.linspace(0,1,posnum))

#generate the positions of the meshes along the line
scale = 20
if posnum%2 == 0:
    xval = posnum/2*scale-scale/2
    pos = np.linspace(-xval,xval,posnum)
else:
    xval = (posnum-1)/2*scale
    pos = np.linspace(-xval,xval,posnum)


for i, p in enumerate(meshfl):
    #get time point
    # time = marr.arbitrarytime.values[i]
    #open time point mesh
    reader = XMLPolyDataReader(FileName= meshdir + p)
    obj = GetRepresentation(reader)
    obj.Position = [pos[i],0,0]
    obj.AmbientColor = discrete_colors[i,:-1]
    obj.DiffuseColor = discrete_colors[i,:-1]
    # obj.Specular = 0.5        # Increase specular reflection
    # obj.SpecularPower = 50.0
    #put the first index also in the last position
    if i==0:
        reader = XMLPolyDataReader(FileName= meshdir + p)
        obj = GetRepresentation(reader)
        obj.Position = [pos[-1],0,0]
        obj.AmbientColor = discrete_colors[-1,:-1]
        obj.DiffuseColor = discrete_colors[-1,:-1]
        # obj.Specular = 0.5        # Increase specular reflection
        # obj.SpecularPower = 50.0
#change background to white
paraview.simple._DisableFirstRenderCameraReset()
LoadPalette(paletteName='WhiteBackground')


view = GetActiveView()
if not view:
    # When using the ParaView UI, the View will be present, not otherwise.
    view = CreateRenderView()
    
view.CameraViewUp = [0, -1, -1]
view.CameraFocalPoint = [0, 0, 0]
view.CameraPosition = [0, 0, -scale*5]
view.ViewSize = [6000, 1500]  
view.OrientationAxesVisibility = 1
# view.UseColorPaletteForBackground = 0
# view.Background = [84/255, 94/255, 135/255]



SaveScreenshot(
    __file__.split('.')[0]+'.png',
    view,
)



# # save animation
# Render()

# WriteImage(__file__.split('.')[0]+'.png')
