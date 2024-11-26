# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:11:34 2022

@author: Aaron
"""

from paraview.simple import *
import os 
import re
import numpy as np


moviename = 'ShapeSpaceAni.tif'
datadir = re.findall('.*(?=Data_and_Figs)', __file__)[0]
meshdir = datadir + 'PC_Meshes/'
meshfl = os.listdir(meshdir)
savedir = re.findall(f'.*(?={os.path.basename(__file__)})', __file__)[0]
PCnum = 10
framenumber = 100
hspacing = 18
vspacing = 13
reconnum = 5
sigmax = 2

if PCnum%2 == 0:
    zval = PCnum/2*vspacing-vspacing/2
    zpos = np.linspace(zval,-zval,PCnum)
else:
    zval = (PCnum-1)/2*vspacing
    zpos = np.linspace(zval,-zval,PCnum)

xval = (reconnum-1)/2*hspacing
xpos = np.linspace(-xval,xval,reconnum)
sigrange = np.linspace(-sigmax,sigmax,reconnum)
xarr = np.stack((sigrange,xpos))




view = GetActiveView()
if not view:
    # When using the ParaView UI, the View will be present, not otherwise.
    view = CreateRenderView()
    
view.CameraViewUp = [0, 0, 1]
view.CameraFocalPoint = [0, 0, 0]
view.CameraViewAngle = 45
view.CameraPosition = [0,-175,0]
view.ViewSize = [4000, 6000]  
view.OrientationAxesVisibility = 0
   

# Create an animation scene
scene = GetAnimationScene()
#set the number of frames
scene.NumberOfFrames = framenumber
# scene.Duration = framenumber
scene.ViewModules = [view]



############ Create all the keyframes just once ############
# Create 2 keyframes for the StartTheta track
# create a key frame
keyf0 = CompositeKeyFrame()
keyf0.KeyTime = 0.0
keyf0.KeyValues = [0.0]
keyf0.Interpolation = 'Ramp'

#back to center
keyf1 = CompositeKeyFrame()
keyf1.Interpolation = 'Ramp'
keyf1.KeyTime = 0.2
keyf1.KeyValues= [-90]

#to the left
keyf2 = CompositeKeyFrame()
keyf2.Interpolation = 'Ramp'
keyf2.KeyTime = 0.4
keyf2.KeyValues= [0]

#back to center
keyf3 = CompositeKeyFrame()
keyf3.Interpolation = 'Ramp'
keyf3.KeyTime = 0.6
keyf3.KeyValues= [90]

#back to center
keyf4 = CompositeKeyFrame()
keyf4.KeyTime = 0.8
keyf4.KeyValues= [0]



# good pink color #ffaaff
#aaaaff

for count, p in enumerate(meshfl):
    PC = re.findall('(?<=PC)\d*', p)[0]
    if p.endswith('Cell.vtp'):
        reader = XMLPolyDataReader(FileName=meshdir + p)
        t = Transform(reader)
        Hide(reader)
        Show(t)
        Hide3DWidgets(proxy=t.Transform)
        obj = GetRepresentation(t)
        obj.Opacity = 0.8
        
        #move to correct map location
        sig = re.findall(f'PC{PC}_(.*)_', p)[0]
        obj.Position = [xarr[1,np.where(xarr==float(sig))[1][0]],0,zpos[int(PC)-1]]        
        
        #do rotation animation
        at = GetAnimationTrack('Rotation', index=2, proxy=t.Transform)

        # initialize the animation track
        at.TimeMode = 'Normalized'
        at.StartTime = 0.0
        at.EndTime = 1.0
        at.Enabled = 1
        at.KeyFrames = [keyf0, keyf1, keyf2, keyf3, keyf4]
        
    if p.endswith('Nuc.vtp'):
        reader = XMLPolyDataReader(FileName=meshdir + p)
        t = Transform(reader)
        Hide(reader)
        Show(t)
        Hide3DWidgets(proxy=t.Transform)
        obj = GetRepresentation(t)
        obj.DiffuseColor = [170/255, 170/255, 1]
        
        #move to correct map location
        sig = re.findall(f'PC{PC}_(.*)_', p)[0]
        obj.Position = [xarr[1,np.where(xarr==float(sig))[1][0]],0,zpos[int(PC)-1]]
    
        #do rotation animation
        at = GetAnimationTrack('Rotation', index=2, proxy=t.Transform)

        # initialize the animation track
        at.TimeMode = 'Normalized'
        at.StartTime = 0.0
        at.EndTime = 1.0
        at.Enabled = 1
        at.KeyFrames = [keyf0, keyf1, keyf2, keyf3, keyf4]
  




# #get sources so I can rotate them for each frame
# so = GetSources()
# for count, s in enumerate(list(so.values())):
#     exec(f'Trans{count}' + ' = Transform(s)')
#     Hide(s)
#     Show(globals()[f'Trans{count}'])
#     Hide3DWidgets(proxy=globals()[f'Trans{count}'].Transform)
#     #create new cue for each loop
#     exec(f'Track{count}' + f" = GetAnimationTrack('Rotation', index=2, proxy=Trans{count}.Transform)")

#     # initialize the animation track
#     globals()[f'Track{count}'].TimeMode = 'Normalized'
#     globals()[f'Track{count}'].StartTime = 0.0
#     globals()[f'Track{count}'].EndTime = 1.0
#     globals()[f'Track{count}'].Enabled = 1
#     globals()[f'Track{count}'].KeyFrames = [keyf0, keyf1, keyf2, keyf3, keyf4]
    

# scene.Play()
SaveAnimation(savedir+'Animation/' + moviename, view, scene)


