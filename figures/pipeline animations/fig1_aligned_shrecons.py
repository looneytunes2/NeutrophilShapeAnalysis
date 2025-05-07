
################### make a mesh animation movie from the meshes saved during
################### data processing


from paraview.simple import *
import os 
import re
import numpy as np
import pandas as pd
import vtk
from pathlib import Path
from scipy.spatial.transform import Rotation as R



realspace = False
scope = 'confocal'
#get some directories
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
meshdir = basedir+'Meshes/'
infodir = basedir+'processed_data/'
widthpeaks = pd.read_csv(basedir+'Data_and_Figs/Closest_Width_Peaks_Galvanotaxis_Confocal_40x_37C_10s.csv', index_col = 0)
cellname = '20231116_488EGFP-CAAX_3mA_37C_1_cell_42'



#get all the position and trajectory info
df = []
for x in os.listdir(infodir):
    if cellname in x:
        df.append(pd.read_csv(infodir+x, index_col = 0))
df = pd.concat(df).sort_values('frame').reset_index(drop=True)
#frame gaps
diffs = df.frame.diff()
gaps = diffs[diffs>1]
trimdf = df.iloc[72:121].reset_index(drop=True)#int(gaps.iloc[0])]
#get displacements and then cumulative position
#get displacements
tempc = trimdf[['x_raw','y_raw','z_raw']].diff().values
if np.any(np.isnan(tempc)):
    naninds = np.unique(np.where(np.isnan(tempc))[0])
    tempc[naninds] = [0,0,0]*len(naninds)
#replace gaps with zeros
cum_pos = np.cumsum(tempc, axis = 0) 


#open the position and euler angle dataframe
extradf = pd.read_csv(basedir+'Data_and_Figs/Shape_Metrics_Galvanotaxis_Confocal_40x_37C_10s.csv', index_col = 0)

#use average volume in the cell info and the volume of the first recon to
#establish scaling
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(os.path.dirname(__file__)+'\\shrecon_meshes\\frame_1_mesh.vtp')
reader.Update()
tempmesh = reader.GetOutput()
CellMassProperties = vtk.vtkMassProperties()
CellMassProperties.SetInputData(tempmesh)
vol = CellMassProperties.GetVolume()
#get average speed in that bin
avgvol = extradf.Cell_Volume.mean()
scaling = (vol/avgvol) ** (1/3)


        
# get animation scene and make it at least the number of frames that I have meshes
animationScene1 = GetAnimationScene()
animationScene1.NumberOfFrames = len(trimdf)

time = 0
interval = 1/len(trimdf)
for i, row in trimdf.iterrows():
    meshfl = os.path.dirname(__file__)+f'\\shrecon_meshes\\frame_{int(row.frame+1)}_mesh.vtp'
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
                    
            
            #ACTUALLY MOVE THE CELL ADJUSTED FOR THE BACK AT ZERO
            obj.Position = cum_pos[i]
            obj.Scale = [1/scaling,1/scaling,1/scaling]
    else:
        source = XMLPolyDataReader(FileName=meshfl)
        obj = GetRepresentation(source)
        obj.Scale = [1/scaling,1/scaling,1/scaling]
        
    # get active source.
    SetActiveSource(source)
    acso = GetActiveSource()
    # get animation representation helper for 'a00vtp'
    rephelp = GetRepresentationAnimationHelper(acso)
    # get animation track
    rephelpvistrackcell = GetAnimationTrack('Visibility', proxy=rephelp)
    
    #make key frames
    keyframes = []
    #make inivisible at first, unless it's the first frame
    if time != 0:
        # make mesh visible at the appropriate time
        keyFrame0 = CompositeKeyFrame()
        keyFrame0.KeyTime = 0.0
        keyFrame0.KeyValues = [0.0]
        keyFrame0.Interpolation = 'Boolean'
        keyframes.append(keyFrame0)
        
        
    # make mesh visible at the appropriate time
    keyFrame1 = CompositeKeyFrame()
    keyFrame1.KeyTime = time
    keyFrame1.KeyValues = [1.0]
    keyFrame1.Interpolation = 'Boolean'
    keyframes.append(keyFrame1)
    
    # make the mesh invisible at the appropriate time except for the last frame
    if i != len(meshfl)-1:
        keyFrame2 = CompositeKeyFrame()
        keyFrame2.KeyTime = time + interval
        keyFrame2.KeyValues = [0.0]
        keyFrame2.Interpolation = 'Boolean'
        keyframes.append(keyFrame2)
        
    # initialize the animation track
    rephelpvistrackcell.KeyFrames = keyframes

    
    time = time + interval
    


view = GetActiveView()
if not view:
    # When using the ParaView UI, the View will be present, not otherwise.
    view = CreateRenderView()
    
view.CameraViewUp = [0, -1, 0]
# view.CameraViewAngle = 180
if realspace:
    avgpos = np.mean(cum_pos,axis = 0)
    view.CameraPosition = [avgpos[0],avgpos[1],avgpos[2]-100]
    view.CameraFocalPoint = avgpos
else:
    view.CameraPosition = [0,0,-100]
    view.CameraFocalPoint = [0,0,0]

view.ViewSize = [200, 200]  
view.OrientationAxesVisibility = 1
view.UseColorPaletteForBackground = 0
view.Background = [84/255, 94/255, 135/255]

# save animation
SaveAnimation(__file__.split('.')[0]+'.mp4', view, ImageResolution=[1000, 1000], FrameRate=2)#, ImageResolution=[788, 364])


