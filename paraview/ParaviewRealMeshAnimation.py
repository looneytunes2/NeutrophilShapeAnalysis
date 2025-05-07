
################### make a mesh animation movie from the meshes saved during
################### data processing


from paraview.simple import *
import os 
import re
import numpy as np
import pandas as pd
import vtk
from scipy.spatial.transform import Rotation as R



realspace = True
scope = 'lls'
#get some directories
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_Apply/'
meshdir = basedir+'Meshes/'
infodir = basedir+'processed_data/'
widthpeaks = pd.read_csv(basedir+'Data_and_Figs/Closest_Width_Peaks_random_lls.csv', index_col = 0)
cellname = '20240527_488_EGFP-CAAX_640_SPY650-DNA_cell2_01'
savedir = basedir+'singlecells/'+cellname
if not os.path.exists(savedir):
    os.makedirs(savedir)


def filename_match_llscellid(
        cellid, #CellID of cell in question
        lst, #list of file names
        ):
    movie = '_'.join(cellid.split('_')[:-1])
    cellinmovie = cellid.split('_')[-1]
    filematches = []
    for l in lst:
        if movie in l:
            if re.search(r'\d+', l.split('Subset-')[-1])[0] == cellinmovie:
                filematches.append(l)
    return filematches


def format_seconds(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02}:{secs:02}"

if realspace:
    if scope == 'lls':
        #get some directories
        df = []
        lsslist = filename_match_llscellid(cellname, os.listdir(infodir))
        for x in lsslist:
            df.append(pd.read_csv(infodir+x, index_col = 0))
        # Sort the dataframe based on extracted numbers
        df = pd.concat(df).sort_values('time').reset_index(drop=True)
        
        #get displacements and then cumulative position
        df['movie'] = [d.split('-Subset')[0] for d in df.cell.to_list()]

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



############ create all of the view stuff and scale it      
view = GetActiveView()
if not view:
    # When using the ParaView UI, the View will be present, not otherwise.
    view = CreateRenderView()
    
view.CameraViewUp = [0, -1, 0]
# view.CameraViewAngle = 180
avgpos = np.mean(cum_pos,axis = 0)
view.CameraPosition = [avgpos[0],avgpos[1],avgpos[2]-(avgpos[0]*avgpos[1]*3)]
view.CameraFocalPoint = avgpos

view.ViewSize = [500, 500]  
view.OrientationAxesVisibility = 1
view.UseColorPaletteForBackground = 0
view.Background = [84/255, 94/255, 135/255]


        
# get animation scene and make it at least the number of frames that I have meshes
animationScene1 = GetAnimationScene()
animationScene1.NumberOfFrames = len(df)
# animationScene1.GoToFirst()


# #### create the time text object to animate
# dummy = Wavelet()
# annotation = PythonAnnotation(Input=dummy)
# annotationDisplay = Show(annotation, view)
# annotationDisplay.FontSize = 24
# annotationDisplay.WindowLocation = 'Upper Left Corner'
# annotation.Expression = format_seconds(0)

# #### add the real time in minutes and seconds
# txtrack = GetAnimationTrack("Expression", index=0, proxy=annotation)
# # txtrack.AnimatedElement = 'Expression'
# textkf = []

time = 0
interval = 1/len(df)
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
                    
            
            #ACTUALLY MOVE THE CELL ADJUSTED FOR THE BACK AT ZERO
            obj.Position = cum_pos[i]
    else:
        source = XMLPolyDataReader(FileName=meshfl)
        obj = GetRepresentation(source)
        
    # get active source.
    SetActiveSource(source)
    acso = GetActiveSource()
    # get animation representation helper for 'a00vtp'
    rephelp = GetRepresentationAnimationHelper(acso)
    # get animation track
    rephelpvistrackcell = GetAnimationTrack('Visibility', proxy=rephelp)
    
    ##make text source and track
    txtsource = Text()
    txtsource.Text = format_seconds(row.time)
    txtobj = GetRepresentation(txtsource)
    # get animation representation helper for 'a00vtp'
    rephelptext = GetRepresentationAnimationHelper(txtsource)
    # get animation track
    textvistrack = GetAnimationTrack('Visibility', proxy=rephelptext)
    
    
    
    #make key frames
    keyframes = []
    keytextframes = []
    #make inivisible at first, unless it's the first frame
    if time != 0:
        # make mesh visible at the appropriate time
        keyFrame0 = CompositeKeyFrame()
        keyFrame0.KeyTime = 0.0
        keyFrame0.KeyValues = [0.0]
        keyFrame0.Interpolation = 'Boolean'
        keyframes.append(keyFrame0)
        
        kft0 = CompositeKeyFrame()
        kft0.KeyTime = 0.0
        kft0.KeyValues = [0.0]
        kft0.Interpolation = 'Boolean'
        keytextframes.append(kft0)
        
    # make mesh visible at the appropriate time
    keyFrame1 = CompositeKeyFrame()
    keyFrame1.KeyTime = time
    keyFrame1.KeyValues = [1.0]
    keyFrame1.Interpolation = 'Boolean'
    keyframes.append(keyFrame1)
    
    kft1 = CompositeKeyFrame()
    kft1.KeyTime = time
    kft1.KeyValues = [1.0]
    kft1.Interpolation = 'Boolean'
    keytextframes.append(kft1)
    
    # make the mesh invisible at the appropriate time except for the last frame
    if i != len(meshfl)-1:
        keyFrame2 = CompositeKeyFrame()
        keyFrame2.KeyTime = time + interval
        keyFrame2.KeyValues = [0.0]
        keyFrame2.Interpolation = 'Boolean'
        keyframes.append(keyFrame2)
        
        kft2 = CompositeKeyFrame()
        kft2.KeyTime = time + interval
        kft2.KeyValues = [0.0]
        kft2.Interpolation = 'Boolean'
        keytextframes.append(kft2)
        
    # initialize the animation track
    rephelpvistrackcell.KeyFrames = keyframes

    textvistrack.KeyFrames = keytextframes
    


    # print(format_seconds(row.time))
    # annotation.Expression = format_seconds(row.time)
    # animationScene1.GoToNext()
    # #change the time annotation
    # kf = CompositeKeyFrame()
    # kf.KeyTime = time
    # kf.KeyValues = [format_seconds(row.time)]
    # textkf.append(kf)

    time = time + interval
    

# txtrack.KeyFrames = textkf

# save animation
SaveAnimation(savedir+'/mesh_animation.mp4', view, ImageResolution=[1000, 1000], FrameRate=10)#, ImageResolution=[788, 364])


