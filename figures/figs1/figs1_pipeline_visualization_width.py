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
from pathlib import Path


#get some directories
basedir = Path('E:/Aaron/Combined_37C_Confocal_PCA_s5/')
meshdir = basedir.joinpath('Meshes')
df = pd.read_csv(basedir.joinpath('Data_and_Figs','All_Data_with_CGPS_bins.csv'), index_col = 0)
cellname = '20231116_488EGFP-CAAX_3mA_37C_1_cell_79_frame_145'
cellinfo = df[df.cell == cellname].copy() #pd.read_csv(infodir.joinpath(cellname+'_cell_info.csv'), index_col = 0)




view = GetActiveView()
if not view:
    # When using the ParaView UI, the View will be present, not otherwise.
    view = CreateRenderView()

#change background to white
cp = GetSettingsProxy('ColorPalette')
cp.Background = [1,1,1]

#change camera settings
view.CameraViewUp = [0.01, 1, 0]
view.CameraFocalPoint = [0, 0, 0]
# view.CameraViewAngle = 45
view.CameraPosition = [0,0,70]
view.ViewSize = [5000, 5000]  
view.OrientationAxesVisibility = 0



############# SCALE BAR
slen = 5
sx = 4.5
sy = -11
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




### get trajectory and rotations to apply to it
vec = cellinfo[['Trajectory_X','Trajectory_Y','Trajectory_Z']].values[0]
Euler_Angles = cellinfo[[x for x in cellinfo.columns if 'Euler' in x]].values[0]#rotationthing[0].as_euler('xyz', degrees = True)
wideroll = cellinfo.Width_Rotation_Angle.values#widthpeaks[widthpeaks.cell == cellname].Closest_minimums.values[0]

## scipy rotations to apply for the trajectory arrow
euler_rotation = R.from_euler('xyz', Euler_Angles, degrees = True)
width_rotation = R.from_euler('x', [wideroll],degrees=True)

### orientation angles for the trajectory arrow
arrow_orient_y = np.rad2deg(np.arctan2(vec[2],np.sqrt(vec[0]**2 + vec[1]**2)))
arrow_orient_z = np.rad2deg(np.arctan2(vec[1],vec[0]))


##### make a little trajectory arrow at a specific position
# green = np.array([45, 181, 81])/255

trajar = Arrow()
trajar.ShaftRadius = 0.1
trajar.ShaftResolution = 1000
trajar.TipLength = 0.3
trajar.TipRadius = 0.2
trajar.TipResolution = 1000

trajar_display = Show(trajar)
trajar_display.Orientation = [0, -arrow_orient_y, arrow_orient_z]
trajar_display.Scale = [5,5,5]
#change arrow color
arcolor = [77/255, 130/255, 56/255]#[0]*3#np.zeros(3)
trajar_display.DiffuseColor = arcolor
trajar_display.AmbientColor = arcolor
#turn off shininess
trajar_display.Specular = 5.0
trajar_display.SpecularPower = 65
trajar_display.Position = [-6,5.5,0]



#### open the mesh
mreader = vtk.vtkXMLPolyDataReader()
mreader.SetFileName(meshdir.joinpath(cellname + '_cell_mesh.vtp').as_posix())
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


xyz_orient_pos = [-8,-14,0]
xyzprops = {'Scale':[[5,5,5], [5,5,5], [5,5,5]],
            'Color':[red, yellow, blue],
            'Orientation': [[0,0,0], [-90,-90,0], [0,-90,90]],
            'Position': [xyz_orient_pos]*3}

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

### rotate trajectory arrow to the first alignment
long_axis_traj_vec = euler_rotation.apply(vec)
### orientation angles for the trajectory arrow
arrow_orient_y = np.rad2deg(np.arctan2(long_axis_traj_vec[2],
                                       np.sqrt(long_axis_traj_vec[0]**2 +
                                               long_axis_traj_vec[1]**2)))
arrow_orient_z = np.rad2deg(np.arctan2(long_axis_traj_vec[1],
                                       long_axis_traj_vec[0]))


trajar_display.Orientation = [0, -arrow_orient_y, arrow_orient_z]
trajar_display.Position = [-4,-9,0]

Render()
    
WriteImage(__file__.split('.')[0] + '_first_rotated.png')
Hide(trajrotsource)
Hide(xax)
Hide(yax)
Hide(zax)
# Hide(box)





Render()
    
# WriteImage(__file__.split('.')[0] + '_unrotated.png')
# Hide(tube)


######### fully rotated mesh
fullrotsource = TrivialProducer()
fullrotsource.GetClientSideObject().SetOutput(mesh)
fullrotobj = GetRepresentation(fullrotsource)
ColorBy(fullrotobj, None)





for i, art in enumerate([xax, yax, zax]):
    if i == 1:
        art = Transform(Input=art)
        art.Transform.Rotate = [-90,-90,0]
    elif i == 2:
        art = Transform(Input=art)
        art.Transform.Rotate = [0,-90,90]
    transform = Transform(Input=art)
    transform.Transform.Rotate = Euler_Angles  # Rotation angles in degrees
    transform1 = Transform(Input=transform)
    transform1.Transform.Rotate = [wideroll,0,0]

    transform_display = Show(transform1)
    transform_display.Scale = xyzprops['Scale'][i]
    transform_display.DiffuseColor = xyzprops['Color'][i]
    transform_display.AmbientColor = xyzprops['Color'][i]
    transform_display.Position = xyzprops['Position'][i]


### rotate trajectory arrow to the first alignment
width_rot_traj_vec = width_rotation.apply(long_axis_traj_vec)[0]
### orientation angles for the trajectory arrow
arrow_orient_y = np.rad2deg(np.arctan2(width_rot_traj_vec[2],
                                       np.sqrt(width_rot_traj_vec[0]**2 +
                                               width_rot_traj_vec[1]**2)))
arrow_orient_z = np.rad2deg(np.arctan2(width_rot_traj_vec[1],
                                       width_rot_traj_vec[0]))


trajar_display.Orientation = [0, -arrow_orient_y, arrow_orient_z]
trajar_display.Position = [-4,-9,0]


Render()
    
WriteImage(__file__.split('.')[0] + '_fully_rotated.png')
Hide(fullrotsource)



######### SH RECON
shreader = XMLPolyDataReader(FileName=Path(__file__).parent.joinpath('pipeline_SH_recon_width.vtp').as_posix())
# shreader.PointArrayStatus = ['Normals']
shobj = Show(shreader, view, 'GeometryRepresentation')



Render()
WriteImage(__file__.split('.')[0] + '_shrecon.png')
Hide(shreader)

