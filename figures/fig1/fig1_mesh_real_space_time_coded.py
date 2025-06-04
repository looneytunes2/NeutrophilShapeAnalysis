
################### make a mesh animation movie from the meshes saved during
################### data processing


from paraview.simple import *
import os 
import numpy as np
import pandas as pd
import vtk
from scipy.spatial.transform import Rotation as R
from matplotlib import cm


realspace = True
scope = 'confocal'
#get some directories
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
meshdir = basedir+'Meshes/'
infodir = basedir+'processed_data/'
widthpeaks = pd.read_csv(basedir+'Data_and_Figs/Closest_Width_Peaks_Galvanotaxis_Confocal_40x_37C_10s.csv', index_col = 0)
cellname = '20231116_488EGFP-CAAX_3mA_37C_2_cell_9'



#get all the position and trajectory info
dflist = []
for x in os.listdir(infodir):
    if cellname in x:
        dflist.append(pd.read_csv(infodir+x, index_col = 0))
df = pd.concat(dflist).sort_values('frame').reset_index(drop=True)
#limit df to frame window
df = df[df.frame.isin(list(range(66,92,3)))].reset_index(drop=True)
#get displacements and then cumulative position
#get displacements
tempc = df[['x_raw','y_raw','z_raw']].diff()
tempc.fillna(0, inplace = True)
cum_pos = np.cumsum(tempc.values, axis = 0)




#define the colors to make the meshes
cmap = cm.get_cmap('rainbow')
discrete_colors = cmap(np.linspace(0,1,len(df)))



for i, row in df.iterrows():
    
    meshfl = meshdir+row.cell+'_cell_mesh.vtp'
    if realspace:
        wideroll = widthpeaks[widthpeaks.cell == row.cell]
        if (os.path.exists(meshfl)) and (len(wideroll)>0):
            ### get euler angles
            vec = np.array([row.Trajectory_X, row.Trajectory_Y, row.Trajectory_Z])
            #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
            xaxis = np.array([[1,0,0], [0,1,0], [0,0,1]]).astype('float64')
            upnorm = np.cross(vec,[1,0,0])
            sidenorm = np.cross(vec,upnorm)
            current_vec = np.stack((vec, sidenorm, upnorm), axis = 0)
            rotationthing = R.align_vectors(xaxis, current_vec)
            #below is actual rotation matrix if needed
            Euler_Angles = rotationthing[0].as_euler('xyz', degrees = True)
            
            #### open the mesh
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(meshfl)
            reader.Update()
            mesh = reader.GetOutput()
            #### transform the mesh
            transformation = vtk.vtkTransform()
            #rotate the shape
            transformation.RotateWXYZ(-Euler_Angles[0], 1, 0, 0)
            transformation.RotateWXYZ(-Euler_Angles[2], 0, 0, 1)
            transformation.RotateWXYZ(-wideroll.Closest_minimums.values[0], 1, 0, 0)
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(transformation)
            transformFilter.SetInputData(mesh)
            transformFilter.Update()
            mesh = transformFilter.GetOutput()
            
            source = TrivialProducer()
            source.GetClientSideObject().SetOutput(mesh)
            obj = GetRepresentation(source)
                    
            ColorBy(obj, None)
            #ACTUALLY MOVE THE CELL ADJUSTED FOR THE BACK AT ZERO
            obj.Position = cum_pos[i]
            #change the visulization
            obj.Opacity = 0.4
            obj.AmbientColor = discrete_colors[i,:-1]
            obj.DiffuseColor = discrete_colors[i,:-1]   
            
    else:
        source = XMLPolyDataReader(FileName=meshfl)
        obj = GetRepresentation(source)
       


############# SCALE BAR
slen = 10
sx = 15
sy = 7.5
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
# view.CameraViewAngle = 180
avgpos = np.mean(cum_pos,axis = 0)
view.CameraPosition = [avgpos[0],avgpos[1],avgpos[2]+(avgpos[0]*avgpos[1]*1.5)]
view.CameraFocalPoint = avgpos

view.ViewSize = [5000, 5000]  
view.OrientationAxesVisibility = 0


Render()

WriteImage(__file__.split('.')[0]+'.png', ImageResolution=[5000, 5000])


