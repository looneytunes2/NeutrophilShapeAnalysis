# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:34:44 2025

@author: Aaron
"""

from paraview.simple import *

from pathlib import Path

avg_recon = Path(__file__).parent.joinpath('average_cell_mesh.vtp').as_posix()


view = GetActiveView()
if not view:
    # When using the ParaView UI, the View will be present, not otherwise.
    view = CreateRenderView()

#change background to white
paraview.simple._DisableFirstRenderCameraReset()
LoadPalette(paletteName='WhiteBackground')
#change camera stuff
cameradist = 50
view.CameraViewUp = [0, 1, 0]
view.CameraFocalPoint = [0, 0, 0]
view.CameraPosition = [0,0,cameradist]
view.ViewSize = [1000, 1000]  


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





#read one of the average PC shapes
reader = XMLPolyDataReader(FileName=avg_recon)
obj = GetRepresentation(reader)
# obj.Opacity = 0.7
views = ['xy','xz','yz']
for vn, v in enumerate(views):
    if vn>0:
        Hide(tube)
    
    if v == 'xy':
        view.CameraViewUp = [0, 1, 0]
        view.CameraPosition = [0, 0, cameradist]

    elif v == 'xz':
        view.CameraViewUp = [0, 0, 1]
        view.CameraPosition = [0, -cameradist, 0]

    elif v == 'yz':
        view.CameraViewUp = [0, 0, 1]
        view.CameraPosition = [cameradist, 0, 0]
    

    
    Render()
    
    WriteImage(__file__.split('.')[0] + f'_{v}.png')



Hide(reader)
Hide(tube)


##### make a little axes at a specific position
yellow = np.array([255, 224, 102])/255
red = np.array([222, 33, 71])/255
blue = np.array([54, 111, 209])/255
xax = Arrow()
yax = Arrow()
zax = Arrow()
for ar in [xax,yax,zax]:
    # ar.ShaftRadius = 0.3
    ar.ShaftResolution = 500
    # ar.TipLength = 0.15
    ar.TipRadius = 0.075
    ar.TipResolution = 500


xyzprops = {'Scale':[[10,10,10]]*3,
            'Color':[red, yellow, blue],
            'Orientation': [[0,0,0], [-90,-90,0], [0,-90,90]],
            'Position': [[0,0,0]]*3}

xax_display = Show(xax)
xax_display.Orientation = xyzprops['Orientation'][0]
xax_display.Scale = xyzprops['Scale'][0]
xax_display.DiffuseColor = xyzprops['Color'][0]
xax_display.AmbientColor = xyzprops['Color'][0]
yax_display = Show(yax)
yax_display.Orientation = xyzprops['Orientation'][1]
yax_display.Scale = xyzprops['Scale'][1]
yax_display.DiffuseColor = xyzprops['Color'][1]
yax_display.AmbientColor = xyzprops['Color'][1]
zax_display = Show(zax)
zax_display.Orientation = xyzprops['Orientation'][2]
zax_display.Scale = xyzprops['Scale'][2]
zax_display.DiffuseColor = xyzprops['Color'][2]
zax_display.AmbientColor = xyzprops['Color'][2]


#### move all the arrows to the correct position
xax_display.Position = xyzprops['Position'][0]
yax_display.Position = xyzprops['Position'][1]
zax_display.Position = xyzprops['Position'][2]


for v in views:
    
    if v == 'xy':
        ResetCamera()
        view.CameraFocalPoint = [0, 0, 0]
        view.CameraViewUp = [0, 1, 0]
        view.CameraPosition = [0, 0, cameradist]
        view.CameraParallelProjection = 1
        view.CameraParallelScale = 15
    elif v == 'xz':
        ResetCamera()
        view.CameraFocalPoint = [0, 0, 0]
        view.CameraViewUp = [0, 0, 1]
        view.CameraPosition = [0, -cameradist, 0]
        view.CameraParallelProjection = 1
        view.CameraParallelScale = 15
    elif v == 'yz':
        ResetCamera()
        view.CameraFocalPoint = [0, 0, 0]
        view.CameraViewUp = [0, 0, 1]
        view.CameraPosition = [cameradist, 0, 0]
        view.CameraParallelProjection = 1
        view.CameraParallelScale = 15
    

    # save animation
    Render()
    
    WriteImage(__file__.split('.')[0]+f'_{v}_axes.png')


