# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:35:37 2023

@author: Aaron
"""

# trace generated using paraview version 5.11.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# get active source.
a00vtp = GetActiveSource()

 # get animation representation helper for 'a00vtp'
a00vtpRepresentationAnimationHelper = GetRepresentationAnimationHelper(a00vtp)


# get animation track
a00vtpRepresentationAnimationHelperVisibilityTrack = GetAnimationTrack('Visibility', index=0, proxy=a00vtpRepresentationAnimationHelper)

# create a new key frame
keyFrame10057 = CompositeKeyFrame()

# create a new key frame
keyFrame10058 = CompositeKeyFrame()
keyFrame10058.KeyTime = 1.0

# initialize the animation track
a00vtpRepresentationAnimationHelperVisibilityTrack.KeyFrames = [keyFrame10057, keyFrame10058]

# Properties modified on keyFrame10057
keyFrame10057.KeyValues = [1.0]

# Properties modified on keyFrame10058
keyFrame10058.KeyTime = 0.5

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(791, 364)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [14.424932667480695, 151.91246350791556, 0.5580325850811229]
renderView1.CameraFocalPoint = [-0.06103515624999902, -0.6728899478912346, -0.2303299903869645]
renderView1.CameraViewUp = [0.03590826056535411, -0.008572192846851059, 0.9993183248259621]
renderView1.CameraParallelScale = 12.640113750059736

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).


# save animation
SaveAnimation('C:/Users/Aaron/Documents/Python Scripts/temp/test.avi', renderView1, ImageResolution=[788, 364],
    FrameWindow=[0, 9])


curdir = 'D:/Aaron/Data/Chemotaxis/Data_and_Figs/contours/loop_4-6_8-6_8-3_4-3_4-6/'


#get some directories
curdir = os.path.dirname(os.path.abspath(__file__))
savedir = re.findall('.*(?=Data_and_Figs)', __file__)[0]


#get the mesh files from the folder
meshfl = [x for x in os.listdir(curdir) if '.py' not in x]


# get animation scene and make it at least the number of frames that I have meshes
animationScene1 = GetAnimationScene()
animationScene1.NumberOfFrames = len(meshfl)


prev = 0
for i, p in enumerate(meshfl):
    reader = XMLPolyDataReader(FileName=curdir + p)
    obj = GetRepresentation(reader)
    # obj.Opacity = 0.7
    time = float(p.replace('.vtp',''))

    # get active source.
    acso = GetActiveSource()
    
     # get animation representation helper for 'a00vtp'
    rephelp = GetRepresentationAnimationHelper(acso)
    
    
    # get animation track
    # rephelpvistrack = GetAnimationTrack('Visibility', index=0, proxy=rephelp)
    rephelpvistrack = GetAnimationTrack('Visibility', proxy=rephelp)
    
    # make mesh visible at the appropriate time
    keyFrame1 = CompositeKeyFrame()
    keyFrame1.KeyTime = time
    keyFrame1.KeyValues = [1.0]
    
    # make the mesh invisible at the appropriate time
    keyFrame2 = CompositeKeyFrame()
    keyFrame2.KeyTime = 2*time - prev
    keyFrame2.KeyValues = [0.0]

    # initialize the animation track
    rephelpvistrack.KeyFrames = [keyFrame1, keyFrame2]
    
    prev = time.copy()
    
#change background to white
paraview.simple._DisableFirstRenderCameraReset()
LoadPalette(paletteName='WhiteBackground')


view = GetActiveView()
if not view:
    # When using the ParaView UI, the View will be present, not otherwise.
    view = CreateRenderView()
    
view.CameraViewUp = [0, 0, 1]
view.CameraFocalPoint = [0, 0, 0]
view.CameraViewAngle = 45
view.CameraPosition = [0,-100,0]
view.ViewSize = [500, 500]  
view.OrientationAxesVisibility = 0
   
# save animation
SaveAnimation('C:/Users/Aaron/Documents/Python Scripts/temp/test.avi', view, ImageResolution=[788, 364])


