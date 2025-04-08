


################### make a mesh animation movie from the meshes made specifically
################### for animations, such as SH reconstruction videos



from paraview.simple import *
import os 
import re
import numpy as np
import pandas as pd
import vtk
from scipy.spatial.transform import Rotation as R
from vtk.util import numpy_support
from itertools import groupby


scope = 'LLS'
realspace = True #bool

#get some directories
basedir = 'E:/Aaron/random_lls/'
cellname = '20240524_488_EGFP-CAAX_640_actin-halotag_cell1_01'
savedir = basedir+f'singlecells/{cellname}/shrecon_meshes/'
  
#get the mesh files from the folder
meshfl = [x for x in os.listdir(savedir) if '.vtp' in x]
if scope == 'LLS':
    # Function to extract sorting keys
    def extract_lls_sort_key(filename):
        subset_match = re.search(r'(\d+)-Subset', filename)
        frame_match = re.search(r'frame_(\d+)', filename)

        subset_num = int(subset_match.group(1)) if subset_match else 0
        frame_num = int(frame_match.group(1)) if frame_match else 0

        return (subset_num, frame_num)
    #sort the list by the complicated function
    meshfl.sort(key=lambda x: extract_lls_sort_key(x))
elif scope == 'confocal':
    #sort the list by the frame number
    meshfl.sort(key=lambda x: float(re.findall('(?<=frame_)\d*', x)[0]))
    
#if you want meshes in their real positions then get the data to do that
if realspace:
    infodir = basedir+'processed_data/'
    widthpeaks = pd.read_csv(basedir+'Data_and_Figs/Closest_Width_Peaks.csv', index_col = 0)
    if scope == 'LLS':
        df = []
        for x in os.listdir(infodir):
            if cellname in x:
                df.append(pd.read_csv(infodir+x, index_col = 0))
        # Sort the dataframe based on extracted numbers
        df = pd.concat(df).reset_index(drop=True)
        #sort the list by the frame number
        df = df.sort_values('cell',key=lambda x: extract_lls_sort_key(x)).reset_index(drop=True)
        
        #get displacements and then cumulative position
        df['movie'] = [d.split('-Subset')[0] for d in df.cell.to_list()]
        df['frame'] = [int(d.split('frame_')[-1]) for d in df.cell.to_list()]
        cum_pos = np.zeros((len(df),3))
        ind = 0
        for m, mov in df.groupby('movie'):
            mov = mov.sort_values('frame').reset_index(drop=True)
            #get displacements
            tempc = mov[['x_raw','y_raw','z_raw']].diff().values
            #replace gaps with zeros
            jumpind = mov.frame.diff()[mov.frame.diff()!=1].index.to_list()
            tempc[jumpind,:] = np.zeros((len(jumpind),3))
            cum_pos[ind:ind+len(tempc),:] = tempc
            ind = ind + len(tempc)
        #actually do cumulative sum
        cum_pos = np.cumsum(cum_pos, axis = 0) 
    
    elif scope == 'confocal':
        #get all the position and trajectory info
        df = []
        for x in os.listdir(infodir):
            if cellname in x:
                df.append(pd.read_csv(infodir+x, index_col = 0))
        df = pd.concat(df).sort_values('frame').reset_index(drop=True)
        #get displacements and then cumulative position
        #get displacements
        tempc = df[['x_raw','y_raw','z_raw']].diff().values
        #replace gaps with zeros
        jumpind = df.frame.diff()[df.frame.diff()!=1].index.to_list()
        tempc[jumpind,:] = np.zeros((len(jumpind),3))
        cum_pos = np.cumsum(tempc, axis = 0) 





    #open the position and euler angle dataframe
    extradf = pd.read_csv(basedir+'Data_and_Figs/Shape_Metrics.csv', index_col = 0)
    
    #use average volume in the cell info and the volume of the first recon to
    #establish scaling
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(savedir+meshfl[0])
    reader.Update()
    tempmesh = reader.GetOutput()
    CellMassProperties = vtk.vtkMassProperties()
    CellMassProperties.SetInputData(tempmesh)
    vol = CellMassProperties.GetVolume()
    #get average speed in that bin
    avgvol = df.Cell_Volume.mean()
    scaling = (vol/avgvol) ** (1/3)
    


        
# get animation scene and make it at least the number of frames that I have meshes
animationScene1 = GetAnimationScene()
animationScene1.NumberOfFrames = len(meshfl)

time = 0
interval = 1/len(meshfl)
for i, m in enumerate(meshfl):
    meshfl = savedir+m
    if realspace:
        row = df[df.cell == m.split('_cell_mesh.vtp')[0]]
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
    else:
        source = XMLPolyDataReader(FileName=meshdir + p)
        obj = GetRepresentation(source)
            
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
avgpos = np.mean(cum_pos,axis = 0)
view.CameraPosition = [avgpos[0],avgpos[1],avgpos[2]-400]
view.CameraFocalPoint = avgpos


view.ViewSize = [500, 500]  
view.OrientationAxesVisibility = 1
view.UseColorPaletteForBackground = 0
view.Background = [84/255, 94/255, 135/255]

# save animation
SaveAnimation(savedir+'/mesh_animation.mp4', view, ImageResolution=[1000, 1000], FrameRate=10)#, ImageResolution=[788, 364])





