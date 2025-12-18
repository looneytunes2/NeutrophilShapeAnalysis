
################### make a mesh animation movie from the meshes saved during
################### data processing


from paraview.simple import *
import os 
import re
import numpy as np
import pandas as pd
import vtk
from scipy.spatial.transform import Rotation as R
from pathlib import Path


realspace = False
scope = 'confocal'
#get some directories
basedir = Path('E:/Aaron/Combined_37C_Confocal_PCA_s5')
meshdir = basedir.joinpath('Meshes')
infodir = basedir.joinpath('processed_data')
datadir = basedir.joinpath('Data_and_Figs')
widthpeaks = pd.read_csv(datadir.joinpath('Closest_Width_Peaks_Galvanotaxis_Confocal_40x_37C_10s.csv'), index_col = 0)
cellname = '20231116_488EGFP-CAAX_3mA_37C_1_cell_116'
savedir = Path('C:/Users/Aaron/Desktop/').joinpath(cellname)#basedir+'singlecells/'+cellname
if not savedir.exists():
    savedir.mkdir(parents = True)


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


def get_frame_int(strr):
    if isinstance(strr,Path):
        return int(re.findall(r'(?<=frame_)\d*', strr.name)[0])
    elif isinstance(strr, str):
        return int(re.findall(r'(?<=frame_)\d*', strr)[0])

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
        df = pd.read_csv(datadir+'All_Data_with_CGPS_bins.csv', index_col=0)
        #narrow df down to cell of interest
        df = df[df.CellID == cellname]
        df = df.sort_values('frame').reset_index(drop=True)
        #get displacements and then cumulative position
        #get displacements
        tempc = df[['x_raw','y_raw','z_raw']].diff().values
        #replace gaps with zeros
        jumpind = df.frame.diff()[df.frame.diff()!=1].index.to_list()
        tempc[jumpind,:] = np.zeros((len(jumpind),3))
        cum_pos = np.cumsum(tempc, axis = 0) 
else:
    #get all the position and trajectory info
    df = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col=0)
    #narrow df down to cell of interest
    df = df[df.CellID == cellname]


############ create all of the view stuff and scale it      
view = GetActiveView()
if not view:
    # When using the ParaView UI, the View will be present, not otherwise.
    view = CreateRenderView()
    
view.CameraViewUp = [0, -1, 0]
# view.CameraViewAngle = 180
avgpos = np.mean(cum_pos,axis = 0) if realspace else np.array([0,0,0])
view.CameraPosition = [avgpos[0],avgpos[1],avgpos[2]-(avgpos[0]*avgpos[1]*3)] if realspace else [0,0,-65]
view.CameraFocalPoint = avgpos

view.ViewSize = [500, 500]  
view.OrientationAxesVisibility = 1
view.UseColorPaletteForBackground = 0
view.Background = [84/255, 94/255, 135/255]



meshfl = [str(x) for x in meshdir.glob(f'*{cellname}*') if x.name.split('_cell_mesh')[0] in df.cell.to_list()]
meshfl = sorted(meshfl, key = get_frame_int)

reader = XMLPolyDataReader(FileName=meshfl)

display = Show(reader, view)
display.Representation = "Surface"
ColorBy(display, None)

animationScene = GetAnimationScene()
animationScene.UpdateAnimationUsingDataTimeSteps()



# ##make text source and track
# txtsource = Text()
# txtsource.Text = format_seconds(df.time.iloc[0])
# Show(txtsource)

# # get animation track
# texttrack = GetAnimationTrack('Text', proxy=txtsource)

# # Create animation keyframes
# keyframes = []
# for i, time in enumerate(df.time):
#     kf = StringKeyFrame()
#     kf.Interpolation = 'None'       # one position per step
#     kf.KeyTime = i
#     kf.KeyValues = time
#     keyframes.append(kf)
# ## add keyframes to track
# texttrack.KeyFrames = keyframes

if realspace:
    transform = Transform(Input = reader)
    transform.Transform = 'Transform'
    
    
    # Create an animation track for the rotation
    rotationTrackX = GetAnimationTrack('Transform', index=0, scene=animationScene)  # X rotation
    rotationTrackY = GetAnimationTrack('Transform', index=1, scene=animationScene)  # Y rotation
    rotationTrackZ = GetAnimationTrack('Transform', index=2, scene=animationScene)  # Z rotation

    for t, (x, y, z) in enumerate(df[['Euler_angles_X','Euler_angles_Y','Euler_angles_Z']].values):
        kx = DoubleKeyFrame()
        kx.KeyTime = t
        kx.KeyValue = x
        rotationTrackX.KeyFrames.append(kx)
        
        ky = DoubleKeyFrame()
        ky.KeyTime = t
        ky.KeyValue = y
        rotationTrackY.KeyFrames.append(ky)
        
        kz = DoubleKeyFrame()
        kz.KeyTime = t
        kz.KeyValue = z
        rotationTrackZ.KeyFrames.append(kz)
        
        
        
    # num_frames = len(meshfl)
    # assert len(positions) == num_frames, "positions array must match number of meshes!"
    
    # # Create animation keyframes
    # keyframes = []
    # for i, pos in enumerate(cum_pos):
    #     kf = CompositeKeyFrame()
    #     kf.Interpolation = 'None'       # one position per step
    #     kf.KeyTime = i
    #     kf.KeyValues = pos
    #     keyframes.append(kf)
    
    # # Hook keyframes to Transform1.Translate
    # ca = GetAnimationTrack('Translate', index=0, proxy=transform)
    # ca.KeyFrames = keyframes

# ---------------------------------------------------------
# Save MP4 animation
# ---------------------------------------------------------
# SaveAnimation("output.mp4", view, FrameRate=30, Compression=True)
SaveAnimation(str(savedir)+'/mesh_animation.mp4', view, ImageResolution=[1000, 1000], FrameRate=3)#, ImageResolution=[788, 364])

        
# # get animation scene and make it at least the number of frames that I have meshes
# animationScene1 = GetAnimationScene()
# animationScene1.NumberOfFrames = len(df)
# # animationScene1.GoToFirst()



# time = 0
# interval = 1/len(df)
# for i, row in df.iterrows():
#     ## get the mesh file
#     meshfl = meshdir+row.cell+'_cell_mesh.vtp'
    
#     #### make keyframes first and then apply only if they
    
#     if os.path.exists(meshfl):
#         if realspace:
#             wideroll = widthpeaks[widthpeaks.cell == row.cell]
#             if len(wideroll)>0:
#                 ### get euler angles
#                 vec = np.array([row.Trajectory_X, row.Trajectory_Y, row.Trajectory_Z])
#                 #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
#                 xaxis = np.array([[1,0,0], [0,1,0], [0,0,1]]).astype('float64')
#                 upnorm = np.cross(vec,[1,0,0])
#                 sidenorm = np.cross(vec,upnorm)
#                 current_vec = np.stack((vec, sidenorm, upnorm), axis = 0)
#                 rotationthing = R.align_vectors(xaxis, current_vec)
#                 #below is actual rotation matrix if needed
#                 Euler_Angles = rotationthing[0].as_euler('xyz', degrees = True)
                
#                 #### open the mesh
#                 reader = vtk.vtkXMLPolyDataReader()
#                 reader.SetFileName(meshfl)
#                 reader.Update()
#                 mesh = reader.GetOutput()
#                 #### transform the mesh
#                 transformation = vtk.vtkTransform()
#                 #rotate the shape
#                 transformation.RotateWXYZ(-Euler_Angles[0], 1, 0, 0)
#                 transformation.RotateWXYZ(-Euler_Angles[2], 0, 0, 1)
#                 transformation.RotateWXYZ(-wideroll.Closest_minimums.values[0], 1, 0, 0)
#                 transformFilter = vtk.vtkTransformPolyDataFilter()
#                 transformFilter.SetTransform(transformation)
#                 transformFilter.SetInputData(mesh)
#                 transformFilter.Update()
#                 mesh = transformFilter.GetOutput()
                
#                 source = TrivialProducer()
#                 source.GetClientSideObject().SetOutput(mesh)
#                 obj = GetRepresentation(source)
                        
                
#                 #ACTUALLY MOVE THE CELL ADJUSTED FOR THE BACK AT ZERO
#                 obj.Position = cum_pos[i]
#         else:
#             source = XMLPolyDataReader(FileName=meshfl)
#             obj = GetRepresentation(source)
        
#     # get active source.
#     SetActiveSource(source)
#     acso = GetActiveSource()
#     # get animation representation helper for 'a00vtp'
#     rephelp = GetRepresentationAnimationHelper(acso)
#     # get animation track
#     rephelpvistrackcell = GetAnimationTrack('Visibility', proxy=rephelp)
    
#     ##make text source and track
#     txtsource = Text()
#     txtsource.Text = format_seconds(row.time)
#     txtobj = GetRepresentation(txtsource)
#     # get animation representation helper for 'a00vtp'
#     rephelptext = GetRepresentationAnimationHelper(txtsource)
#     # get animation track
#     textvistrack = GetAnimationTrack('Visibility', proxy=rephelptext)
    
    
    
#     #make key frames
#     keyframes = []
#     keytextframes = []
#     #make inivisible at first, unless it's the first frame
#     if time != 0:
#         # make mesh visible at the appropriate time
#         keyFrame0 = CompositeKeyFrame()
#         keyFrame0.KeyTime = 0.0
#         keyFrame0.KeyValues = [0.0]
#         keyFrame0.Interpolation = 'Boolean'
#         keyframes.append(keyFrame0)
        
#         kft0 = CompositeKeyFrame()
#         kft0.KeyTime = 0.0
#         kft0.KeyValues = [0.0]
#         kft0.Interpolation = 'Boolean'
#         keytextframes.append(kft0)
        
#     # make mesh visible at the appropriate time
#     keyFrame1 = CompositeKeyFrame()
#     keyFrame1.KeyTime = time
#     keyFrame1.KeyValues = [1.0]
#     keyFrame1.Interpolation = 'Boolean'
#     keyframes.append(keyFrame1)
    
#     kft1 = CompositeKeyFrame()
#     kft1.KeyTime = time
#     kft1.KeyValues = [1.0]
#     kft1.Interpolation = 'Boolean'
#     keytextframes.append(kft1)
    
#     # make the mesh invisible at the appropriate time except for the last frame
#     if i != len(meshfl)-1:
#         keyFrame2 = CompositeKeyFrame()
#         keyFrame2.KeyTime = time + interval
#         keyFrame2.KeyValues = [0.0]
#         keyFrame2.Interpolation = 'Boolean'
#         keyframes.append(keyFrame2)
        
#         kft2 = CompositeKeyFrame()
#         kft2.KeyTime = time + interval
#         kft2.KeyValues = [0.0]
#         kft2.Interpolation = 'Boolean'
#         keytextframes.append(kft2)
        
#     # initialize the animation track
#     rephelpvistrackcell.KeyFrames = keyframes

#     textvistrack.KeyFrames = keytextframes
    


#     # print(format_seconds(row.time))
#     # annotation.Expression = format_seconds(row.time)
#     # animationScene1.GoToNext()
#     # #change the time annotation
#     # kf = CompositeKeyFrame()
#     # kf.KeyTime = time
#     # kf.KeyValues = [format_seconds(row.time)]
#     # textkf.append(kf)

#     time = time + interval
    

# # txtrack.KeyFrames = textkf

# # save animation
# SaveAnimation(savedir+'/mesh_animation.mp4', view, ImageResolution=[1000, 1000], FrameRate=10)#, ImageResolution=[788, 364])


