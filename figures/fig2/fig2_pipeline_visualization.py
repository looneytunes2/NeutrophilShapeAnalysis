# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:34:44 2025

@author: Aaron
"""

from paraview.simple import *
import numpy as np
import pandas as pd
import vtk
from scipy.spatial.transform import Rotation as R
import os


#get some directories
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
meshdir = basedir+'Meshes/'
infodir = basedir+'processed_data/'
widthpeaks = pd.read_csv(basedir+'Data_and_Figs/Closest_Width_Peaks_Galvanotaxis_Confocal_40x_37C_10s.csv', index_col = 0)
cellname = '20231116_488EGFP-CAAX_3mA_37C_1_cell_79_frame_147'
cellinfo = pd.read_csv(infodir+cellname+'_cell_info.csv')


# scale = 50
# meshfl = list(range(3))
# if len(meshfl)%2 == 0:
#     xval = len(meshfl)/2*scale-scale/2
#     pos = np.linspace(-xval,xval,len(meshfl))
# else:
#     xval = (len(meshfl)-1)/2*scale
#     pos = np.linspace(-xval,xval,len(meshfl))


view = GetActiveView()
if not view:
    # When using the ParaView UI, the View will be present, not otherwise.
    view = CreateRenderView()

#change background to white
cp = GetSettingsProxy('ColorPalette')
cp.Background = [1,1,1]

#change camera settings
view.CameraViewUp = [0, -1, -1]
view.CameraFocalPoint = [0, 0, 0]
# view.CameraViewAngle = 45
view.CameraPosition = [0,0,-70]
view.ViewSize = [5000, 5000]  
view.OrientationAxesVisibility = 0



############# SCALE BAR
slen = 5
sx = 3
sy = 11
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




############# get the rotations to undo them
### get euler angles
vec = np.array([cellinfo.Trajectory_X[0], cellinfo.Trajectory_Y[0], cellinfo.Trajectory_Z[0]])
#align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
xaxis = np.array([[1,0,0], [0,1,0], [0,0,1]]).astype('float64')
upnorm = np.cross(vec,[1,0,0])
sidenorm = np.cross(vec,upnorm)
current_vec = np.stack((vec, sidenorm, upnorm), axis = 0)
rotationthing = R.align_vectors(xaxis, current_vec)
#below is actual rotation matrix if needed
Euler_Angles = rotationthing[0].as_euler('xyz', degrees = True)
wideroll = widthpeaks[widthpeaks.cell == cellname].Closest_minimums.values[0]


### make rotation object to "unrotate the trajectory vector

second = R.from_euler('x', [wideroll],degrees=True)
unrotthing = second * rotationthing[0]
unroteuler = unrotthing.as_euler('xyz', degrees = True)[0]


##### make a little trajectory arrow at a specific position
# green = np.array([45, 181, 81])/255

trajar = Arrow()
trajar.ShaftRadius = 0.1
trajar.ShaftResolution = 500
trajar.TipLength = 0.3
trajar.TipRadius = 0.2
trajar.TipResolution = 500

trajar_display = Show(trajar)
trajar_display.Orientation = -Euler_Angles
trajar_display.Scale = [5,5,5]
#change arrow color
arcolor = np.zeros(3)
trajar_display.DiffuseColor = arcolor
trajar_display.AmbientColor = arcolor
#turn off shininess
trajar_display.Specular = 0.0
trajar_display.Position = [-5.5,5.5,0]



#### open the mesh
mreader = vtk.vtkXMLPolyDataReader()
mreader.SetFileName(meshdir + cellname + '_cell_mesh.vtp')
mreader.Update()
mesh = mreader.GetOutput()
# mesh = translate_to_origin(mesh)


################# undo ALL the rotations
transformation = vtk.vtkTransform()
#rotate the shape
transformation.RotateWXYZ(-Euler_Angles[0], 1, 0, 0)
transformation.RotateWXYZ(-Euler_Angles[2], 0, 0, 1)
transformation.RotateWXYZ(-wideroll, 1, 0, 0)
transformFilter = vtk.vtkTransformPolyDataFilter()
transformFilter.SetTransform(transformation)
transformFilter.SetInputData(mesh)
transformFilter.Update()
unrot = transformFilter.GetOutput()

unrotsource = TrivialProducer()
unrotsource.GetClientSideObject().SetOutput(unrot)
unrotobj = GetRepresentation(unrotsource)
# unrotobj.Position = [pos[0],0,0]
ColorBy(unrotobj, None)



# #get bounds of the fully un-rotated mesh
# bounds = unrot.GetBounds() #(xmin,xmax, ymin,ymax, zmin,zmax)

# #make a box that is the orientation of the lab
# box = Box()
# box.XLength = bounds[1]-bounds[0]  # Set width along the X-axis
# box.YLength = bounds[3]-bounds[2]   # Set height along the Y-axis
# box.ZLength = bounds[5]-bounds[4]   # Set depth along the Z-axis
# #center the box which is off by these coords for some reason
# box.Center = [0,-1,-2]


# box_display = Show(box)
# box_display.Representation = 'Wireframe'
# box_display.Opacity = 1.0
# # box_display.Position = [0,-1,-2]


##### make a little axes at a specific position
yellow = np.array([255, 224, 102])/255
red = np.array([222, 33, 71])/255
blue = np.array([54, 111, 209])/255
xax = Arrow()
yax = Arrow()
zax = Arrow()
for ar in [xax,yax,zax]:
    # ar.ShaftRadius = 0.3
    ar.ShaftResolution = 100
    # ar.TipLength = 0.15
    ar.TipRadius = 0.075
    ar.TipResolution = 100


xyzprops = {'Scale':[[5,5,5], [5,5,5], [5,5,5]],
            'Color':[red, yellow, blue],
            'Orientation': [[0,0,0], [-90,-90,0], [0,-90,90]],
            'Position': [[-8,13,0],[-8,13,0],[-8,13,0]]}

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

Render()
    
WriteImage(__file__.split('.')[0] + '_unrotated.png')
Hide(unrotsource)
Hide(tube)



############## undo the width rotation
transformation = vtk.vtkTransform()
#rotate the shape
transformation.RotateWXYZ(-wideroll, 1, 0, 0)
transformFilter = vtk.vtkTransformPolyDataFilter()
transformFilter.SetTransform(transformation)
transformFilter.SetInputData(mesh)
transformFilter.Update()
trajrot = transformFilter.GetOutput()
trajrotsource = TrivialProducer()
trajrotsource.GetClientSideObject().SetOutput(trajrot)
trajrotobj = GetRepresentation(trajrotsource)
ColorBy(trajrotobj, None)


#rotate the lab orientation box
# box_display.Orientation = [Euler_Angles[0], 0, Euler_Angles[2]]
xax_display.Orientation = [Euler_Angles[0], 0, Euler_Angles[2]]
yax_display.Orientation = [Euler_Angles[0]-90,-90,Euler_Angles[2]]
zax_display.Orientation = [Euler_Angles[0],-90,Euler_Angles[2]+90]

#align traj with x
trajar_display.Orientation = [0,0,0]
trajar_display.Position = [-3,7,0]

Render()
    
WriteImage(__file__.split('.')[0] + '_traj_rotated.png')
Hide(trajrotsource)
Hide(xax)
Hide(yax)
Hide(zax)
# Hide(box)




# ############# SCALE BAR
# # 5um line scalebar
# line = Line(Point1=[0, 25, 0], Point2=[5, 25, 0])
# # Apply a Tube filter to give it thickness
# tube = Tube(Input=line)
# tube.Radius = 0.5  # Adjust thickness as needed
# tube.NumberofSides = 20  # Makes it smoother
# # Show the tube in the active view
# tube_display = Show(tube)
# #change color
# tube_display.DiffuseColor = [0, 0, 0]

Render()
    
# WriteImage(__file__.split('.')[0] + '_unrotated.png')
# Hide(tube)


######### fully rotated mesh
fullrotsource = TrivialProducer()
fullrotsource.GetClientSideObject().SetOutput(mesh)
fullrotobj = GetRepresentation(fullrotsource)
ColorBy(fullrotobj, None)






# #rotate the lab orientation box
# transform = Transform(Input=box)
# transform.Transform.Rotate = first.as_euler('xyz', degrees = True)  # Rotation angles in degrees
# transform1 = Transform(Input=transform)
# transform1.Transform.Rotate = [wideroll,0,0]
# # Show the transformed box
# transform_display = Show(transform1)
# transform_display.Representation = 'Wireframe'
# transform_display.Opacity = 1.0




for i, art in enumerate([xax, yax, zax]):
    if i == 1:
        art = Transform(Input=art)
        art.Transform.Rotate = [-90,-90,0]
    elif i == 2:
        art = Transform(Input=art)
        art.Transform.Rotate = [0,-90,90]
    transform = Transform(Input=art)
    transform.Transform.Rotate = rotationthing[0].as_euler('xyz', degrees = True)  # Rotation angles in degrees
    transform1 = Transform(Input=transform)
    transform1.Transform.Rotate = [wideroll,0,0]

    transform_display = Show(transform1)
    transform_display.Scale = xyzprops['Scale'][i]
    transform_display.DiffuseColor = xyzprops['Color'][i]
    transform_display.AmbientColor = xyzprops['Color'][i]
    transform_display.Position = xyzprops['Position'][i]


#move trajectory arrow for the last time
trajar_display.Position = [-1,11,0]


Render()
    
WriteImage(__file__.split('.')[0] + '_fully_rotated.png')
Hide(fullrotsource)



######### SH RECON
shreader = XMLPolyDataReader(FileName=[os.path.dirname(__file__) + '/' + cellname + '_cell_mesh.vtp'])
shreader.PointArrayStatus = ['Normals']
shobj = Show(shreader, view, 'GeometryRepresentation')
# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
shobj.ScaleTransferFunction.Points = [-0.9984926581382751, 0.0, 0.5, 0.0, 0.9936378002166748, 1.0, 0.5, 0.0]
# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
shobj.OpacityTransferFunction.Points = [-0.9984926581382751, 0.0, 0.5, 0.0, 0.9936378002166748, 1.0, 0.5, 0.0]
# obj.Position = [pos[-1],0,0]
# ColorBy(shobj, None)


Render()
WriteImage(__file__.split('.')[0] + '_shrecon.png')
Hide(shreader)

