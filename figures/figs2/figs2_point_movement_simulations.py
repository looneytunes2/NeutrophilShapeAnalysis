# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:49:33 2025

@author: Aaron
"""


import os
import re
import numpy as np
import vtk
from vtk.util import numpy_support
# from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from CustomFunctions import linear_cycle_utils, utils

time_interval = 10
origin = [9, 9]
whichpcs = [1,7]
binrange = 20
direction = 'clockwise'
zerostart = 'left'
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
shapefold = os.getcwd()+f'\\PC{whichpcs[0]}-PC{whichpcs[1]}_Cycle_AllSHCoeff_Visualization\\Random\\'
if not os.path.exists(shapefold):
    os.makedirs(shapefold)

#open actual data
df = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
TotalFrame = df[df.Treatment=='Random'].copy()

# shapefold = 'C:/Users/Aaron/NeutrophilShapeAnalysis/figures/figs2/PC1-PC7_Cycle_AllSHCoeff_Visualization/Random'

cycleshapes = [x for x in os.listdir(shapefold) if '.vtp' in x]
cycleshapes.sort(key=lambda x: float(re.findall('(?<=frame_)\d*', x)[0]))


############## if the average shapes for this shape cycle don't exist, make them
if len(cycleshapes)<1:
    

    ############### linearize PC cycle #############
    angframe = linear_cycle_utils.linearize_cycle_continuous(
                TotalFrame, 
                centers,
                origin, 
                whichpcs,
                zerostart,
                direction,)
    
    angframe =  linear_cycle_utils.bin_angular_coord(
            angframe,
            whichpcs,
            binrange,
            )


    for t, treat in angframe.groupby('Treatment'):
        linear_cycle_utils.animate_linear_cycle_shcoeffs(
                                angframe,
                                os.getcwd(),
                                t,
                                whichpcs,
                                binrange,
                                lmax = 10,
                                smooth = False
                                )

    #now get the files of the shapes
    cycleshapes = [x for x in os.listdir(shapefold) if '.vtp' in x]
    cycleshapes.sort(key=lambda x: float(re.findall('(?<=frame_)\d*', x)[0]))

###### open the linearized data if you didn't just create it
if 'angframe' not in locals():
    angframe = pd.read_csv(shapefold + '/linear_cycle_data.csv', index_col = 0)
    TotalFrame = pd.merge(TotalFrame,angframe, how = 'left', on = 'cell')


########### also open AER/CF data
aerdf = pd.read_csv(basedir + f'random/PC{whichpcs[0]}-PC{whichpcs[1]}_raw_transition_aer_cf.csv', index_col = 0)
TotalFrame = pd.merge(TotalFrame, aerdf, how = 'left', on = 'cell')

#average degrees per second in each bin
avgcf = TotalFrame.angular_velocity.mean()
#average number of seconds spent in each bin
avgtime = binrange / avgcf

#### get the volume of the first mesh for scaling
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(shapefold+'/'+cycleshapes[0])
reader.Update()
mesh = reader.GetOutput()
CellMassProperties = vtk.vtkMassProperties()
CellMassProperties.SetInputData(mesh)
shvolume = CellMassProperties.GetVolume()
# vol_ratio = (TotalFrame.Cell_Volume.mean()/shvolume)**(1/3)
bincol = [c for c in TotalFrame.columns.to_list() if 'Continuous_Angular_Bins' in c][0]
binvals = np.sort(TotalFrame[bincol].unique())
#get average speed in that bin
avgvol = TotalFrame[TotalFrame[bincol]==binvals[0]].Cell_Volume.mean()
scaling = (avgvol/shvolume) ** (1/3)



lens = []
coordlist = []
for c in cycleshapes:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(shapefold+'/'+c)
    reader.Update()
    mesh = reader.GetOutput()
    ### scale mesh to roughly microns based on the ratio of the real average
    ### volume to the volume of the first fram shape (roughly the average)
    # transformation = vtk.vtkTransform()
    # #set scale to actual image scale
    # transformation.Scale(scaling, scaling, scaling)
    # transformFilter = vtk.vtkTransformPolyDataFilter()
    # transformFilter.SetTransform(transformation)
    # transformFilter.SetInputData(mesh)
    # transformFilter.Update()
    # mesh = transformFilter.GetOutput()
    
    #get cell major, minor, and mini axes using the segmented image
    cell_coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    centroid = cell_coords.mean(axis=0, keepdims=True)
    cell_coords -= centroid
    ##### get centroid, largest, and smallest x coordinates
    coordlist.append([cell_coords.mean(axis=0, keepdims=True)[0],
                        cell_coords[cell_coords[:,0] == np.max(cell_coords[:,0])][0],
                        cell_coords[cell_coords[:,0] == np.min(cell_coords[:,0])][0]])
    #Get length of the cell projected along the x axis
    lens.append(np.max(cell_coords[:,0]) - np.min(cell_coords[:,0]))
#make numpy array from centroid, front, and back coords
larr = np.array(coordlist)


##### colors
line_colors = plt.cm.Dark2.colors[:3]


# ############## what do front and back do when centroid moves as it actually moves
# avgbinspeeds = TotalFrame.groupby(f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins').speed.mean().values
# #combine speeds with times
# avgbindis = avgbinspeeds*avgtime
# cumbindis = np.cumsum(avgbindis)#*16
# ########### moving the back
# blarr = larr.copy()
# for i,l in enumerate(blarr):
#     #start by aligning the back at 0
#     blarr[i,:,0] = l[:,0] - l[-1,0]
#     #then add displacement to each
#     blarr[i,:,0] = blarr[i,:,0] + cumbindis[i]
# ############ moving the centroid
# clarr = larr.copy()
# for i,l in enumerate(clarr):
#     clarr[i,:,0] = l[:,0] + cumbindis[i] - l[0,0]
# #shift back to 0 start
# clarr[:,:,0] = clarr[:,:,0] - clarr[0,-1,0]
# ############ moving the front
# flarr = larr.copy()
# for i,l in enumerate(flarr):
#     flarr[i,:,0] = l[:,0] + cumbindis[i] - l[1,0]
# #shift back to 0 start
# flarr[:,:,0] = flarr[:,:,0] - flarr[0,-1,0]




# #plot
# fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey = True)
# label = ['Back','Center','Front']
# for i, l in enumerate([blarr, clarr, flarr]): 
#     for j, y in enumerate(l[:,:,0].T):
#         sns.lineplot(x = np.array(range(len(y)))*(360/len(y)), y = y, color = line_colors[j], lw=2.5, ci='none',ax = axes[i], label = label[j])
#         axes[i].set_title(f'{label[i]} Displacement', fontsize = 20)
#         if i != 0:
#             axes[i].get_legend().remove()
#         else:
#             axes[i].set_ylabel('X Position (μm)', fontsize = 20)
#         if i == 1:
#             axes[i].set_xlabel('Angular Cycle Position (°)', fontsize = 20)
#         axes[i].spines['top'].set_visible(False)
#         axes[i].spines['right'].set_visible(False)
# plt.tight_layout()


# plt.savefig(__file__.split('.')[0] + '_actual_displacements.png', dpi = 500, bbox_inches='tight')




############## what do the positions of the front and rear look like with realistic centroid movement
avgbinspeeds = TotalFrame.groupby(f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins').speed.mean().values
#combine speeds with times
avgbindis = avgbinspeeds*avgtime
cumbindis = np.cumsum(avgbindis)#*16
############ moving the centroid
clarr = larr.copy()
for i,l in enumerate(clarr):
    clarr[i,:,0] = l[:,0] + cumbindis[i] - l[0,0]
#shift back to 0 start
clarr[:,:,0] = clarr[:,:,0] - clarr[0,-1,0]

label = ['Back','Center','Front']
fig, ax = plt.subplots(1,1, figsize=(5, 5))
for j, y in enumerate(clarr[:,:,0].T):
    sns.lineplot(x = np.arange(0,360,binrange), y = y, color = line_colors[j], lw=2.5, ci='none', ax = ax, label = label[j])


#make x ticks intervals of 60
ax.set_xticks(np.arange(0,420,60))
ax.set_xlim(-20,360)

# ax.set_title('Back Position Fixed', fontsize = 20)
ax.set_ylabel('X Position (μm)', fontsize = 20)
ax.set_xlabel('Angular Cycle Position (°)', fontsize = 20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend().set_visible(False)

plt.tight_layout()
    
plt.savefig(__file__.split('.')[0] + '_shrecon_positions_real_centroid_displacement.png', dpi = 500, bbox_inches='tight')




############################
reallengths = TotalFrame.groupby(f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins')[['LengthAlongTrajectoryFront','LengthAlongTrajectoryRear']].mean().values
cdis = np.hstack((np.zeros((1,len(reallengths))).T,reallengths))
#move all the coords up to starting at zero
cdis = cdis+abs(cdis[0,-1])
#add the cumulative displacements
cdis = cdis + cumbindis[:,np.newaxis]


label = ['Back','Center','Front']
fig, ax = plt.subplots(1,1, figsize=(5, 5))
for j, y in enumerate(cdis[:,:].T):
    sns.lineplot(x = np.arange(0,360,binrange), y = y, color = line_colors[j], lw=2.5, ci='none', ax = ax, label = label[j])


#make x ticks intervals of 60
ax.set_xticks(np.arange(0,420,60))
ax.set_xlim(-20,360)

ax.set_title('Average Translocation', fontsize = 20)
ax.set_ylabel('X Position (μm)', fontsize = 20)
ax.set_xlabel('Angular Cycle Position (°)', fontsize = 20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend().set_visible(False)

plt.tight_layout()
    
plt.savefig(__file__.split('.')[0] + '_meshavg_positions_real_centroid_displacement.png', dpi = 500, bbox_inches='tight')


# ############ displacements based on constant approximate average speed and volume scaling
# avgspeed = TotalFrame.speed.mean()
# displacements = np.array(range(len(larr)))*avgspeed

# ########### moving the back
# blarr = larr.copy()
# for i,l in enumerate(blarr):
#     #start by aligning the back at 0
#     blarr[i,:,0] = l[:,0] - l[-1,0]
#     #then add displacement to each
#     blarr[i,:,0] = blarr[i,:,0] + displacements[i]
# ############ moving the centroid
# clarr = larr.copy()
# for i,l in enumerate(clarr):
#     clarr[i,:,0] = l[:,0] + displacements[i] - l[0,0]
# #shift back to 0 start
# clarr[:,:,0] = clarr[:,:,0] - clarr[0,-1,0]
# ############ moving the front
# flarr = larr.copy()
# for i,l in enumerate(flarr):
#     flarr[i,:,0] = l[:,0] + displacements[i] - l[1,0]
# #shift back to 0 start
# flarr[:,:,0] = flarr[:,:,0] - flarr[0,-1,0]


# ##### colors
# line_colors = plt.cm.Dark2.colors[:3]


# #plot
# fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey = True)
# label = ['Back','Center','Front']
# for i, l in enumerate([blarr, clarr, flarr]): 
#     for j, y in enumerate(l[:,:,0].T):
#         sns.lineplot(x = np.array(range(len(y)))*(360/len(y)), y = y, color = line_colors[j], lw=2.5, ci='none',ax = axes[i], label = label[j])
#         axes[i].set_title(f'{label[i]} Displacement', fontsize = 20)
#         if i != 0:
#             axes[i].get_legend().remove()
#         else:
#             axes[i].set_ylabel('X Position (μm)', fontsize = 20)
#         if i == 1:
#             axes[i].set_xlabel('Angular Cycle Position (°)', fontsize = 20)
#         axes[i].spines['top'].set_visible(False)
#         axes[i].spines['right'].set_visible(False)
# plt.tight_layout()


# plt.savefig(__file__.split('.')[0] + '_constant_displacements.png', dpi = 500, bbox_inches='tight')




########## BACK MOVEMENT BUT SUBTRACT DISPLACEMENT
########### moving the back
blarrnodis = larr.copy()
for i,l in enumerate(blarrnodis):
    #start by aligning the back at 0
    blarrnodis[i,:,0] = l[:,0] - l[-1,0]
    #then add displacement to each
    blarrnodis[i,:,0] = blarrnodis[i,:,0]

label = ['Back','Center','Front']
fig, ax = plt.subplots(1,1, figsize=(5, 5))
for j, y in enumerate(blarrnodis[:,:,0].T):
    sns.lineplot(x = np.arange(0,360,binrange), y = y, color = line_colors[j], lw=2.5, ci='none', ax = ax, label = label[j])


#make x ticks intervals of 60
ax.set_xticks(np.arange(0,420,60))
ax.set_xlim(-20,360)

ax.set_title('Back Position Fixed', fontsize = 20)
ax.set_ylabel('X Position (μm)', fontsize = 20)
ax.set_xlabel('Angular Cycle Position (°)', fontsize = 20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend().set_visible(False)

plt.tight_layout()
    
plt.savefig(__file__.split('.')[0] + '_back_no_displacement.png', dpi = 500, bbox_inches='tight')






TotalFrame.groupby(f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins').LengthAlongTrajectoryRear.mean()












