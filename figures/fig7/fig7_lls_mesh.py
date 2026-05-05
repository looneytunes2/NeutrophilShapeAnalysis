
################### make a mesh animation movie from the meshes saved during
################### data processing


from paraview.simple import *
import os 
import numpy as np
import pandas as pd
import vtk
from scipy.spatial.transform import Rotation as R
from matplotlib import cm


#get some directories

datadir = Path('C:/Users/Aaron/NeutrophilShapeAnalysis/data/trajectory_lls/shape_data')
meshdir = Path('E:/Aaron/random_lls/meshes')
df = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col = 0)
cellname = '20240520_488_EGFP-CAAX_561_mysoin-mApple_37C_cell2-04-Subset-01_frame_29'
celldf = df[df.cell == cellname].copy()
meshfl = meshdir.joinpath(cellname+'_cell_mesh.vtp')



#### get rotation angles to reverse
Euler_Angles = celldf[[x for x in celldf.columns if 'Euler' in x]].values[0]
wideroll = celldf.Width_Rotation_Angle.values[0]



#### open the mesh
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(meshfl)
reader.Update()
mesh = reader.GetOutput()


source = TrivialProducer()
source.GetClientSideObject().SetOutput(mesh)
obj = GetRepresentation(source)
        
ColorBy(obj, None)
  



############# SCALE BAR
slen = 5
sx = 4
sy = 10
# 10um line scalebar
line = Line(Point1=[sx, sy, 0], Point2=[sx+slen, sy, 0])
# Apply a Tube filter to give it thickness
tube = Tube(Input=line)
tube.Radius = 0.2  # Adjust thickness as needed
tube.NumberofSides = 20  # Makes it smoother
# Show the tube in the active view
tube_display = Show(tube)
#change color
tube_display.DiffuseColor = [0, 0, 0]




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

arrow_pos = [-1,7.5,0]
xyzprops = {'Scale':[[3,3,3]]*3,
            'Color':[red, yellow, blue],
            'Orientation': [[0,0,0], [-90,-90,0], [0,-90,90]],
            'Position': [arrow_pos]*3}

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




#change background to white
paraview.simple._DisableFirstRenderCameraReset()
LoadPalette(paletteName='WhiteBackground')
# txtrack.KeyFrames = textkf


############ create all of the view stuff and scale it      
view = GetActiveView()
if not view:
    # When using the ParaView UI, the View will be present, not otherwise.
    view = CreateRenderView()
    
view.CameraViewUp = [0, -1, 0]
view.CameraPosition = [0,0,-50]


view.ViewSize = [5000, 5000]  
view.OrientationAxesVisibility = 0


Render()

WriteImage(__file__.split('.')[0]+'.png', ImageResolution=[5000, 5000])


