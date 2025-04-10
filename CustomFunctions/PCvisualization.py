# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:55:43 2023

@author: Aaron
"""

######### do contour integrals for all migration modes ################
import vtk
from aicsshparam import shtools
import os
import numpy as np
from scipy import interpolate
from scipy.spatial import distance
import re
from vtk.util import numpy_support as vtknp
import math
import operator
from functools import reduce
import matplotlib.pyplot as plt
import pandas as pd
from CustomFunctions import shtools_mod

def save_mesh(mesh, savedir):
    #delete file if it already exists
    if os.path.exists(savedir):
        os.remove(savedir)
    #save mesh
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(savedir)
    writer.SetInputData(mesh)
    writer.Write()
    return

def mesh_from_bins(binpos,
                   whichpcs,
                   avgpcs,
                   PC1bins, #list of bin centers for PC1
                   PC2bins,#list of bin centers for PC2
                   pca,
                   savedir,
                   lmax,
                   binsorPCs: str = 'PCs',
                   ):
    temppcs = avgpcs.copy()

    #exchange the values of the appropriate PCs in temppcs with the desired PC values
    for w in range(len(whichpcs)):
        temppcs[whichpcs[w]-1] = PC1bins[int(binpos[w]-1)]
        
    #inverse pca transform
    coeffs = pca.inverse_transform(temppcs)
    #get mesh from coeffs
    mesh, _ = shtools.get_reconstruction_from_coeffs(coeffs.reshape(2,lmax+1,lmax+1))

    #save mesh
    save_mesh(mesh, savedir)
    return


def mesh_from_PCs(avgpcs, #average value for all PCs generated with the pca
                    whichpcs, #which PC number is being reconstructed
                    PCs, #list of PCs, [0] is the first PC, [1] is the second
                    pca, #actual pca file
                    lmax,):
    temppcs = avgpcs.copy()
    #transform PCs back from bins into PCs and put them in the proper location
    #in the average PC array
    for w in range(len(whichpcs)):
        temppcs[whichpcs[w]-1] = PCs[w]
    #inverse pca transform
    coeffs = pca.inverse_transform(temppcs)
    #get mesh from coeffs
    mesh, _ = shtools.get_reconstruction_from_coeffs(coeffs.reshape(2,lmax+1,lmax+1))
    return mesh

def animate_PCs(avgpcs, #average value for all PCs generated with the pca
                whichpcs, #which PC number is being reconstructed
                PCs, #list of PCs, [0] is the first PC, [1] is the second
                pca, #actual pca file
                savedir,
                lmax,
                rotations = [],):
    #generate mesh from PCs given
    mesh = mesh_from_PCs(avgpcs,whichpcs, PCs, pca, lmax,)
    #rotate if needed
    if any(rotations):
        #rotate  mesh
        #worked from https://kitware.github.io/vtk-examples/site/Python/PolyData/AlignTwoPolyDatas/
        #rotate around z axis
        transformation = vtk.vtkTransform()
        #rotate the shape
        transformation.RotateWXYZ(rotations[0], 1, 0, 0)
        transformation.RotateWXYZ(rotations[2], 0, 0, 1)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(mesh)
        transformFilter.Update()
        mesh = transformFilter.GetOutput()
        #rotate the mesh around x by a specific angle
        if len(rotations)>3:
            #rotate around x by the widest y axis
            transformation = vtk.vtkTransform()
            #rotate the shape
            transformation.RotateWXYZ(rotations[3], 1, 0, 0)
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(transformation)
            transformFilter.SetInputData(mesh)
            transformFilter.Update()
            mesh = transformFilter.GetOutput()
    #save mesh
    save_mesh(mesh, savedir)
    return


def interpolate_transitions_by_distance(traj,
                                        interpfreq: float = 0.1, #distance interval
                                        ):
    

    #ensure float for duplicates adjustment
    traj = traj.astype(float)
    
    #get total distance to use for intervals
    totaldist = []
    for i, y in enumerate(traj[:-1]):
        totaldist.append(distance.pdist([traj[i,:],traj[i+1,:]])[0])
    totaldist = sum(totaldist)

    #remove duplicates
    #generate the tiny amount to add incase there's consecutive duplicates
    tinyadd = 0
    duplicates = [i for i,w in enumerate(traj) if all(w==traj[i-1])]
    for d in duplicates:
        tinyadd = tinyadd + 0.0001
        traj[d,:] = traj[d,:]+tinyadd
    
    #interpolate 
    tck, b = interpolate.splprep(traj.T, k=1, s=0)

    #measure the trajectory and interpolate evenly by distance
    interlist = []
    rollingstart = 0
    for t in range(len(traj)-1):
        di = distance.pdist([traj[t,:],traj[t+1,:]])[0]
        intt = round(di/interpfreq)
        #### distances for duplicates are low so make sure to include at least one point 
        #### for that frame even through things don't move much
        if intt<1:
            intt = 1
        interpoints = np.linspace(start=rollingstart/totaldist, stop = (rollingstart+di)/totaldist, num = intt, endpoint = False)
        x, y = interpolate.splev(interpoints,tck)
        interlist.append(np.stack([np.array([t]*len(x)),interpoints,x,y]).T)
        rollingstart = rollingstart + di

    return np.concatenate(interlist)


def interpolate_transitions_by_time(traj,
                                    ppt: int = 5, #points per time
                                    ):
    #ensure float for duplicates adjustment
    traj = traj.astype(float)
    #get interval 
    int_int = 1/(len(traj)-1)
    #generate the tiny amount to add incase there's consecutive duplicates
    tinyadd = 0
    duplicates = [i for i,w in enumerate(traj) if all(w==traj[i-1])]
    for d in duplicates:
        tinyadd = tinyadd + 0.0001
        traj[d,:] = traj[d,:]+tinyadd
    
    #interpolate 
    tck, b = interpolate.splprep(traj.T, k=1, s=0)
    
    #measure the trajectory and interpolate evenly by distance
    interlist = []
    for t in range(len(traj)-1):
        interpoints = np.linspace(start=t*int_int, stop = t*int_int+int_int, num = ppt, endpoint = False)
        x, y = interpolate.splev(interpoints,tck)
        interlist.append(np.stack([interpoints,x,y]).T)

    return np.concatenate(interlist)


def interpolate_contour_shapes(vertices,
                               avgpcs,
                               whichpcs,
                               pca,
                               PC1bins,
                               PC2bins,
                               savedir,
                               lmax,
                               TotalFrame,
                               metrics: list = [], #calculate average metrics at positions along the trajectory
                               interpfreq: float = 0.1 #interpolation frequency
                               ):
    
    #exchange the values of the appropriate PCs in temppcs with the desired PC values
    newverts = vertices.copy()
    for i, v in enumerate(newverts):
        newverts[i,0] = PC1bins[int(v[0]-1)]
        newverts[i,1] = PC2bins[int(v[1]-1)]
    
    #extend the corners to make a full loop
    newverts = np.concatenate((newverts,np.array([newverts[0,:]])))
    
    #interpolate the trajectory in the transition space
    interarray = interpolate_transitions_by_distance(newverts,interpfreq)
    
    #create the name of the current mesh
    digitlist = re.findall(r'\d', np.array2string(vertices))
    dash = ['-'.join(digitlist[x:x+2]) for x in list(range(0,len(digitlist),2))]
    underscore = '_'.join(dash)
    loopname = 'loop_'+underscore


    #make the directory to put the meshes in
    longsave = savedir+'contours/'+loopname+'/'
    if not os.path.exists(longsave):
        os.makedirs(longsave)
    
    #prepare the dataframe stuff if metrics along trajectory are to be calculated
    TotalFrame = TotalFrame.sort_values(['CellID','frame'])
    metricsarray = np.zeros((interarray.shape[0],interarray.shape[1]+len(metrics)))
    #get actual meshes along interpolated trajectory and/or calculate metrics in those positions
    for count, i in enumerate(interarray):
        mesh = mesh_from_PCs(avgpcs,
                          whichpcs,
                          i[2:],
                          pca,
                          lmax)
        save_mesh(mesh, longsave+str(format(i[1],'.5f'))+'.vtp')
        
        #get some average stats for the bins in the contour
        if len(metrics)>0:
            current =  TotalFrame[(TotalFrame['PC1bins'] == round(i[2])) & (TotalFrame['PC2bins'] == round(i[3]))]
            if current.empty:
                metricsarray[count,:] = np.concatenate((i ,[0]*len(metrics)))
            elif len(current)==1:
                metricsarray[count,:] = np.concatenate((i ,current[metrics].iloc[0,:].values.tolist()))
            else:
                metricsarray[count,:] = np.concatenate((i ,current[metrics].mean().values.tolist()))
    
    metricsframe = pd.DataFrame(metricsarray, columns=['original coord number','arbitrarytime','PC1bin','PC2bin']+metrics)
    metricsframe.to_csv(longsave+'interarray.csv')
    # np.save(longsave+'interarray.npy', metricsarray)    

    return interarray, loopname, metricsframe



########### function for finding 2D contour of 3D mesh ##############
def find_plane_mesh_intersection(mesh, proj, use_vtk_for_intersection=True):

    # Find axis orthogonal to the projection of interest
    axis = [a for a in [0, 1, 2] if a not in proj][0]

    # Get all mesh points
    points = vtknp.vtk_to_numpy(mesh.GetPoints().GetData())

    if not np.abs(points[:, axis]).sum():
        raise Exception("Only zeros found in the plane axis.")

    if use_vtk_for_intersection:

        mid = np.mean(points[:, axis])
        '''Set the plane a little off center to avoid undefined intersections.
        Without this the code hangs when the mesh has any edge aligned with the
        projection plane. Also add a little of noisy to the coordinates to
        help with the same problem.'''
        mid += 0.75
        offset = 0.1 * np.ptp(points, axis=0).max()

        # Create a vtkPlaneSource
        plane = vtk.vtkPlaneSource()
        plane.SetXResolution(4)
        plane.SetYResolution(4)
        if axis == 0:
            plane.SetOrigin(
                mid, points[:, 1].min() - offset, points[:, 2].min() - offset
            )
            plane.SetPoint1(
                mid, points[:, 1].min() - offset, points[:, 2].max() + offset
            )
            plane.SetPoint2(
                mid, points[:, 1].max() + offset, points[:, 2].min() - offset
            )
        if axis == 1:
            plane.SetOrigin(
                points[:, 0].min() - offset, mid, points[:, 2].min() - offset
            )
            plane.SetPoint1(
                points[:, 0].min() - offset, mid, points[:, 2].max() + offset
            )
            plane.SetPoint2(
                points[:, 0].max() + offset, mid, points[:, 2].min() - offset
            )
        if axis == 2:
            plane.SetOrigin(
                points[:, 0].min() - offset, points[:, 1].min() - offset, mid
            )
            plane.SetPoint1(
                points[:, 0].min() - offset, points[:, 1].max() + offset, mid
            )
            plane.SetPoint2(
                points[:, 0].max() + offset, points[:, 1].min() - offset, mid
            )
        plane.Update()
        plane = plane.GetOutput()

        # Trangulate the plane
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(plane)
        triangulate.Update()
        plane = triangulate.GetOutput()

        # Calculate intersection
        intersection = vtk.vtkIntersectionPolyDataFilter()
        intersection.SetInputData(0, mesh)
        intersection.SetInputData(1, plane)
        intersection.Update()
        intersection = intersection.GetOutput()

        # Get coordinates of intersecting points
        points = vtknp.vtk_to_numpy(intersection.GetPoints().GetData())
        coords = points[:, proj]

    else:
        
        valids = np.where((points[:,axis] > -2.5)&(points[:,axis] < 2.5))
        coords = points[valids[0]][:,proj]

    # Sorting points clockwise
    # This has been discussed here:
    # https://stackoverflow.com/questions/51074984/sorting-according-to-clockwise-point-coordinates/51075469
    # but seems not to be very efficient. Better version is proposed here:
    # https://stackoverflow.com/questions/57566806/how-to-arrange-the-huge-list-of-2d-coordinates-in-a-clokwise-direction-in-python
    center = tuple(
        map(
            operator.truediv,
            reduce(lambda x, y: map(operator.add, x, y), coords),
            [len(coords)] * 2,
        )
    )
    coords = sorted(
        coords,
        key=lambda coord: (
            -135
            - math.degrees(
                math.atan2(*tuple(map(operator.sub, coord, center))[::-1])
            )
        )
        % 360,
    )

    # Store sorted coordinates
    # points[:, proj] = coords
    return np.array(coords)

def PC2D_contour_for_axes(meshdir,
                          proj,
                          PCkey,
                          binkey,):
    ############# GET THE CONTOURS OF THE MOST EXTREME PCs ##############
    #create some empty variables
    contours = []
    xmaxs = []
    ymaxs = []
    xmins = []
    ymins = []
    
    for p in PCkey:
        for b in binkey:
            #find specific mesh
            me = [x for x in os.listdir(meshdir) if f'PC{p}_' in x and f'_{b}_' in x][0]
            #read the mesh file
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(meshdir + me)
            reader.Update()
            mesh = reader.GetOutput()
            #get 2D contour
            coords = find_plane_mesh_intersection(mesh, proj)
            #store in dict
            contours.append(np.vstack((coords, coords[0,:])))
            #record max coords for this particular structure for axis normalization later
            xmaxs.append(math.ceil(coords[:,0].max()))
            ymaxs.append(math.ceil(coords[:,1].max()))
            xmins.append(math.floor(coords[:,0].min()))
            ymins.append(math.floor(coords[:,1].min()))
    
    xmax = max(xmaxs)
    ymax = max(ymaxs)
    xmin = min(xmins)
    ymin = min(ymins)
    
    totalmax = max(abs(np.array([xmin,xmax,ymin,ymax])))
    
    return contours, totalmax
            


def get_contours_for_axes(meshdir,
                proj,
                PCkey,
                binkey,
                graphnum: int = 1,):
    
    gridshape = (5, 5+(4*(graphnum-1)))
    
    ax1 = plt.subplot2grid(gridshape, (3, 0))
    ax2 = plt.subplot2grid(gridshape, (0, 0))
    ax3 = plt.subplot2grid(gridshape, (4, 4))
    ax4 = plt.subplot2grid(gridshape, (4, 1))
    
    graphaxes = []
    for g in range(graphnum):
        graphaxes.append(plt.subplot2grid(shape=gridshape, loc=(0, 1+(4*g)), colspan=4, rowspan=4))
    
    contours, minmax = PC2D_contour_for_axes(meshdir,
                                            proj,
                                            PCkey,
                                            binkey,)
    axes = [ax1, ax2, ax3, ax4]
    
    for i, a in enumerate(axes):
        a.plot(contours[i][:,0], contours[i][:,1])
        a.set_xlim(-minmax,minmax)
        a.set_ylim(-minmax,minmax)
        a.axis("off")
        
        

    return graphaxes, axes


    
# Function to extract sorting keys
def extract_lls_sort_key(filename):
    subset_match = re.search(r'(\d+)-Subset', filename)
    frame_match = re.search(r'frame_(\d+)', filename)

    subset_num = int(subset_match.group(1)) if subset_match else 0
    frame_num = int(frame_match.group(1)) if frame_match else 0

    return (subset_num, frame_num)

######## get SH reconstructions of a cell from all time points 
######## specifically for LLS data
def shcoeff_recon_mesh_timelapse_realspace(
        basedir,
        cellname,
        savedir,
        lmax = 10,
        ):
    #get some directories
    df = pd.read_csv(basedir + 'Data_and_Figs/Shape_Metrics.csv')
    if 'Subset' in df.cell.iloc[0]:
        df['CellID'] = [re.split(r'(-\d+-Subset)',x)[0]+ '_'+ re.findall(r'Subset-(\d+)', x)[0] for x in df.cell.to_list()]
    else:
        df['CellID'] = [x.split('_frame')[0] for x in df.cell.to_list()]
    cell = df[df.CellID == cellname]
    specificdir = savedir+'/shrecon_meshes/'
    if not os.path.exists(specificdir):
        os.makedirs(specificdir)


    # Sort the list based on extracted numbers
    if 'Subset' in df.cell.iloc[0]:
        cellsorted = cell.sort_values(by="cell", key=lambda col: col.map(extract_lls_sort_key)).reset_index(drop=True)
    else:
        cell['frame'] = [int(x.split('frame_')[-1]) for x in cell.cell.to_list()]
        cellsorted = cell.sort_values('frame').reset_index(drop=True)
        
    poz = []
    for i, row in cellsorted.iterrows():
        c = row.cell
        coeffs = row[[x for x in cell.columns.to_list() if 'shco' in x]].values
        mesh, _ = shtools_mod.get_even_reconstruction_from_coeffs(np.reshape(coeffs, (2,lmax+1,lmax+1)), lmax)
        save_mesh(mesh, specificdir + f'frame_{int(row.frame)}_mesh.vtp')
        poz.append(pd.read_csv(basedir+'processed_data/'+c+'_cell_info.csv', index_col = 0))
    #add position info
    pozdf = pd.concat(poz)
    merged = cellsorted.merge(pozdf, left_on='cell',right_on='cell')
    merged.to_csv(specificdir+'cell_info.csv')



