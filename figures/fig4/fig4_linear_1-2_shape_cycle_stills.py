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
meshdir = curdir+'/PC1-PC2_Cycle_AllSHCoeff_Visualization/Random/'
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


        
#change background to white
paraview.simple._DisableFirstRenderCameraReset()
LoadPalette(paletteName='WhiteBackground')


view = GetActiveView()
if not view:
    # When using the ParaView UI, the View will be present, not otherwise.
    view = CreateRenderView()
    
cameradist = 100
view.CameraViewUp = [0, 1, 0]
view.CameraFocalPoint = [0, 0, 0]
view.CameraPosition = [0, 0, cameradist]
view.ViewSize = [4000, 4000]  
# view.OrientationAxesVisibility = 0


############# SCALE BAR
slen = 5
sx = 5
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




views = ['xy','yz']
for i, p in enumerate(meshfl):
    for v in views:
        #get time point
        # time = marr.arbitrarytime.values[i]
        #open time point mesh
        reader = XMLPolyDataReader(FileName= meshdir + p)
        obj = GetRepresentation(reader)
        # obj.Position = [pos[i],0,0]
        obj.AmbientColor = discrete_colors[i,:-1]
        obj.DiffuseColor = discrete_colors[i,:-1]
        
        if v == 'xy':
            view.CameraViewUp = [0, 1, 0]
            view.CameraPosition = [0, 0, cameradist]
        elif v == 'xz':
            view.CameraViewUp = [0, 0, 1]
            view.CameraPosition = [0, -cameradist, 0]
        elif v == 'yz':
            view.CameraViewUp = [0, 0, 1]
            view.CameraPosition = [cameradist, 0, 0]
        

        # obj.Specular = 0.5        # Increase specular reflection
        # obj.SpecularPower = 50.0
        #put the first index also in the last position
        # if i==0:
        #     reader = XMLPolyDataReader(FileName= meshdir + p)
        #     obj = GetRepresentation(reader)
        #     obj.Position = [pos[-1],0,0]
        #     obj.AmbientColor = discrete_colors[-1,:-1]
        #     obj.DiffuseColor = discrete_colors[-1,:-1]
        #     # obj.Specular = 0.5        # Increase specular reflection
        #     # obj.SpecularPower = 50.0
            
            
        
        # save animation
        Render()
        
        WriteImage(__file__.split('.')[0]+f'_{p}_{v}.png')
        
        Hide(reader)
    
    
    
    

    


    
    





# view.UseColorPaletteForBackground = 0
# view.Background = [84/255, 94/255, 135/255]



# SaveScreenshot(
#     __file__.split('.')[0]+'.png',
#     view,
# )



# # save animation
# Render()

# WriteImage(__file__.split('.')[0]+'.png')
