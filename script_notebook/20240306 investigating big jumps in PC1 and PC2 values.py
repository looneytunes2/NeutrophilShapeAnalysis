# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:22:51 2024

@author: Aaron
"""

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.transforms
import seaborn as sns
import math
from math import sqrt, factorial
import re
from itertools import groupby
import scipy
import random
from decimal import Decimal
from operator import itemgetter
import multiprocessing
from CustomFunctions import PCvisualization
from scipy.spatial.transform import Rotation as R
import vtk



########## find a cell of interest ###########
savedir = 'D:/Aaron/Data/Combined_Confocal_PCA/'
infrsavedir = savedir + 'Inframe_Videos/'

TotalFrame = pd.read_csv(savedir + 'Shape_Metrics_transitionPCbins.csv', index_col=0)

NewTotalFrame = pd.read_csv('D:/Aaron/Data/Combined_Confocal_PCA_newrotation/Shape_Metrics_transitionPCbins.csv', index_col=0)

#find the length of cell consecutive frames
results = []
for i, cells in TotalFrame.groupby('CellID'):
    cells = cells.sort_values('frame').reset_index(drop = True)
    runs = list()
    #######https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
    for k, g in groupby(enumerate(cells['frame']), lambda ix: ix[0] - ix[1]):
        currentrun = list(map(itemgetter(1), g))
        list.append(runs, currentrun)
    maxrun = max([len(l) for l in runs])
    actualrun = max(runs, key=len, default=[])
    results.append([i, maxrun, actualrun])
#find
stdf = pd.DataFrame(results, columns = ['CellID','length_of_run','actual_run']).sort_values('length_of_run', ascending=False).reset_index(drop=True)
stdf.CellID.head(30)






#select cell from list above
row = stdf.loc[5]
print(row.CellID)
#get the data related to this run of this cell
data = TotalFrame[(TotalFrame.CellID==row.CellID) & (TotalFrame.frame.isin(row.actual_run))].sort_values('frame').reset_index(drop=True)



#NEW ROTATION DATA
newdata = NewTotalFrame[(NewTotalFrame.CellID==row.CellID) & (NewTotalFrame.frame.isin(row.actual_run))].sort_values('frame').reset_index(drop=True)




####### compare the PC values AND width rotation angles
fig, axes = plt.subplots(8,1)
axes[0].plot(data.PC1)
axes[0].plot(newdata.PC1)
axes[1].plot(data.PC2)
axes[1].plot(newdata.PC2)
axes[2].plot(data.Width_Rotation_Angle.diff())
axes[2].plot(newdata.Width_Rotation_Angle.diff())
axes[3].plot(data.Euler_angles_X.diff())
axes[3].plot(newdata.Euler_angles_X.diff())
axes[4].plot(data.Euler_angles_Z.diff())
axes[4].plot(newdata.Euler_angles_Z.diff())
axes[5].plot(data.Turn_Angle)
axes[5].plot(newdata.Turn_Angle)
axes[6].plot(data.Trajectory_X)
axes[6].plot(newdata.Trajectory_X)
axes[7].plot(data.Trajectory_Z)
axes[7].plot(newdata.Trajectory_Z)
plt.show()





vec = data.iloc[25][['Trajectory_X','Trajectory_Y','Trajectory_Z']].to_numpy().astype(float).copy()
#align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
xaxis = np.array([[0,0,0],[1,0,0], [5,0,0]]).astype('float64')
current_vec = np.stack(([0,0,0],vec), axis = 0)
current_vec = np.concatenate((current_vec,[5*vec]), axis = 0)
rotationthing = R.align_vectors(xaxis, current_vec)
#below is actual rotation matrix if needed
#rot_mat = rotationthing[0].as_matrix()
rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)
Euler_Angles = np.array([rotthing_euler[0], rotthing_euler[1], rotthing_euler[2]])
Euler_Angles



############## NEW WAY WITH DEFINING MORE VECTORS
lessmove = []
for x in range(len(newdata)):
    vec = newdata.iloc[x][['Trajectory_X','Trajectory_Y','Trajectory_Z']].to_numpy().astype(float).copy()
    #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    xaxis = np.array([[1,0,0],[0,1,0], [0,0,1]]).astype('float64')
    upnorm = np.cross(vec,[1,0,0])
    # if upnorm[2]<0:
    #     upnorm = upnorm.copy() * -1
    sidenorm = np.cross(vec,upnorm)
    current_vec = np.stack((vec, sidenorm, upnorm), axis = 0)
    # current_vec = np.concatenate((current_vec,[5*vec]), axis = 0)
    rotationthing = R.align_vectors(xaxis, current_vec)
    #below is actual rotation matrix if needed
    #rot_mat = rotationthing[0].as_matrix()
    rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)
    Euler_Angles = np.array([rotthing_euler[0], rotthing_euler[1], rotthing_euler[2]])
    lessmove.append(Euler_Angles)
    
plt.plot(lessmove)




######################### OLD WAY WITH ONLY DEFINING THE X AXIS VECTOR
lessmove = []
for x in range(len(newdata)):
    vec = newdata.iloc[x][['Trajectory_X','Trajectory_Y','Trajectory_Z']].to_numpy().astype(float).copy()
    #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    xaxis = np.array([[0,0,0],[1,0,0], [5,0,0]]).astype('float64')
    current_vec = np.stack(([0,0,0],vec), axis = 0)
    current_vec = np.concatenate((current_vec,[5*vec]), axis = 0)
    rotationthing = R.align_vectors(xaxis, current_vec)
    #below is actual rotation matrix if needed
    #rot_mat = rotationthing[0].as_matrix()
    rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)
    Euler_Angles = np.array([rotthing_euler[0], rotthing_euler[1], rotthing_euler[2]])
    lessmove.append(Euler_Angles)
    
plt.plot(lessmove)








############ test whether inverted euler angles acutally results in reflected mesh orientation
row = stdf.loc[5]
#NEW ROTATION DATA
newdata = NewTotalFrame[(NewTotalFrame.CellID==row.CellID) & (NewTotalFrame.frame.isin(row.actual_run))].sort_values('frame').reset_index(drop=True)
########## find a cell of interest ###########
meshdir = 'D:/Aaron/Data/CK666/Processed_Data/Meshes/'

firstframe = newdata.iloc[108]
secondframe = newdata.iloc[109]

# Read all the data from the file
mesh = vtk.vtkXMLPolyDataReader()
mesh.SetFileName(meshdir + firstframe.cell + '_cell_mesh.vtp')
mesh.Update()
mesh = mesh.GetOutput()
#save cell mesh
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName('C:/Users/Aaron/Desktop/firstframerotation.vtp')
writer.SetInputData(mesh)
writer.Write()
    

########### UNDO THE ALIGNMENT FROM MY PIPELINE
transformation = vtk.vtkTransform()
#rotate the shape
transformation.RotateWXYZ(-firstframe.Euler_angles_X, 1, 0, 0)
transformation.RotateWXYZ(-firstframe.Euler_angles_Z, 0, 0, 1)
#set scale to actual image scale
transformFilter = vtk.vtkTransformPolyDataFilter()
transformFilter.SetTransform(transformation)
transformFilter.SetInputData(mesh)
transformFilter.Update()
unrotmesh = transformFilter.GetOutput()


############### DO ROTATION OF SAME MESH FOR THE NEXT FRAME
transformation = vtk.vtkTransform()
#rotate the shape
transformation.RotateWXYZ(secondframe.Euler_angles_Z, 0, 0, 1)
transformation.RotateWXYZ(secondframe.Euler_angles_X, 1, 0, 0)
transformFilter = vtk.vtkTransformPolyDataFilter()
transformFilter.SetTransform(transformation)
transformFilter.SetInputData(unrotmesh)
transformFilter.Update()
newrotmesh = transformFilter.GetOutput()


#save cell mesh
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName('C:/Users/Aaron/Desktop/secondframerotation.vtp')
writer.SetInputData(newrotmesh)
writer.Write()
    





#################### change the rotation of selected mesh to new rotation
meshlist = [x for x in os.listdir('C:/Users/Aaron/Documents/Python Scripts/temp/Meshes') if '.vtp' in x]
meshlist.sort(key=lambda x: float(re.findall('(?<=frame_)\d*', x)[0]))
for m in meshlist:
    fr = newdata[newdata.cell == m.split('_cell_mesh')[0]]
    ############# calculate the new fixed euler angles
    vec = fr[['Trajectory_X','Trajectory_Y','Trajectory_Z']].to_numpy()[0].astype(float).copy()
    #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    xaxis = np.array([[1,0,0],[0,1,0], [0,0,1]]).astype('float64')
    upnorm = np.cross(vec,[1,0,0])
    # if upnorm[2]<0:
    #     upnorm = upnorm.copy() * -1
    sidenorm = np.cross(vec,upnorm)
    current_vec = np.stack((vec, sidenorm, upnorm), axis = 0)
    # current_vec = np.concatenate((current_vec,[5*vec]), axis = 0)
    rotationthing = R.align_vectors(xaxis, current_vec)
    #below is actual rotation matrix if needed
    #rot_mat = rotationthing[0].as_matrix()
    rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)
    Euler_Angles = np.array([rotthing_euler[0], rotthing_euler[1], rotthing_euler[2]])

    
    ############################ reverse old rotaiton and apply the new one
    # Read all the data from the file
    mesh = vtk.vtkXMLPolyDataReader()
    mesh.SetFileName('C:/Users/Aaron/Documents/Python Scripts/temp/Meshes/' + m)
    mesh.Update()
    mesh = mesh.GetOutput()
    # #save cell mesh
    # writer = vtk.vtkXMLPolyDataWriter()
    # writer.SetFileName('C:/Users/Aaron/Desktop/firstframerotation.vtp')
    # writer.SetInputData(mesh)
    # writer.Write()
        
    
    ########### UNDO THE ALIGNMENT FROM MY PIPELINE
    transformation = vtk.vtkTransform()
    #rotate the shape
    transformation.RotateWXYZ(-fr.Euler_angles_X, 1, 0, 0)
    transformation.RotateWXYZ(-fr.Euler_angles_Z, 0, 0, 1)
    #set scale to actual image scale
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transformation)
    transformFilter.SetInputData(mesh)
    transformFilter.Update()
    unrotmesh = transformFilter.GetOutput()
    
    
    ############### DO ROTATION OF SAME MESH FOR THE NEXT FRAME
    transformation = vtk.vtkTransform()
    #rotate the shape
    transformation.RotateWXYZ(Euler_Angles[2], 0, 0, 1)
    transformation.RotateWXYZ(Euler_Angles[0], 1, 0, 0)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transformation)
    transformFilter.SetInputData(unrotmesh)
    transformFilter.Update()
    newrotmesh = transformFilter.GetOutput()
    
    
    #save cell mesh
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName('C:/Users/Aaron/Documents/Python Scripts/temp/New rotation Meshes/' + m)
    writer.SetInputData(newrotmesh)
    writer.Write()
