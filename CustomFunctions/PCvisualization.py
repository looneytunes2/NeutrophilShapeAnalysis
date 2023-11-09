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
                   pc1int,
                   pc2int,
                   pca,
                   savedir,
                   lmax,
                   ):
    temppcs = avgpcs.copy()
    #transform PCs back from bins into PCs and put them in the proper location
    #in the average PC array
    temppcs[whichpcs[0]-1] = pc1int(binpos[1])
    temppcs[whichpcs[1]-1] = pc2int(binpos[2])
    #inverse pca transform
    coeffs = pca.inverse_transform(temppcs)
    #get mesh from coeffs
    mesh, _ = shtools.get_reconstruction_from_coeffs(coeffs.reshape(2,lmax+1,lmax+1))

    #save mesh
    save_mesh(mesh, savedir+str(round(binpos[0],3))+'.vtp')
    return


def mesh_from_PCs(avgpcs, #average value for all PCs generated with the pca
                    whichpcs, #which PC number is being reconstructed
                    PCs, #list of PCs, [0] is the first PC, [1] is the second
                    pca, #actual pca file
                    savedir,
                    lmax,):
    temppcs = avgpcs.copy()
    #transform PCs back from bins into PCs and put them in the proper location
    #in the average PC array
    temppcs[whichpcs[0]-1] = PCs[0]
    temppcs[whichpcs[1]-1] = PCs[1]
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
    mesh = mesh_from_PCs(avgpcs,whichpcs, PCs, pca, savedir, lmax,)
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
                            interpfreq: float = 0.1):
    
    #get interval 
    int_int = 1/(len(traj)-1)
    
    duplicates = [i for i,w in enumerate(traj) if all(w==traj[i-1])]
    for d in duplicates:
        traj[d,:] = traj[d,:]+0.001
    
    #interpolate 
    tck, b = interpolate.splprep(traj.T, k=1, s=0)


    #measure the trajectory and interpolate evenly by distance
    interlist = []
    for t in range(len(traj)-1):
        di = distance.pdist([traj[t,:],traj[t+1,:]])[0]
        intt = round(di/interpfreq)
        interpoints = np.linspace(start=t*int_int, stop = t*int_int+int_int, num = intt, endpoint = False)
        x, y = interpolate.splev(interpoints,tck)
        interlist.append(np.stack([interpoints,x,y]).T)


    return np.concatenate(interlist)


def interpolate_transitions_by_time(traj,
                                    ppt: int = 5, #points per time
                                    ):
    #get interval 
    int_int = 1/(len(traj)-1)
    
    duplicates = [i for i,w in enumerate(traj) if all(w==traj[i-1])]
    for d in duplicates:
        traj[d,:] = traj[d,:]+0.001
    
    #interpolate 
    tck, b = interpolate.splprep(traj.T, k=1, s=0)
    
    #measure the trajectory and interpolate evenly by distance
    interlist = []
    for t in range(len(traj)-1):
        interpoints = np.linspace(start=t*int_int, stop = t*int_int+int_int, num = ppt, endpoint = False)
        x, y = interpolate.splev(interpoints,tck)
        interlist.append(np.stack([interpoints,x,y]).T)

    return np.concatenate(interlist)


def interpolate_contour_shapes(fourcorners,
                               avgpcs,
                               whichpcs,
                               pca,
                               PC1bins,
                               PC2bins,
                               savedir,
                               lmax,
                               interpfreq: float = 0.1):
    
    
    #extend the corners to make a full loop
    fourcorners = np.concatenate((fourcorners,np.array([fourcorners[0,:]])))
    
    #interpolate the trajectory in the transition space
    interarray = interpolate_transitions_by_distance(fourcorners,interpfreq)


    #get interpolator for PCbins back into PCs
    pc1int = interpolate.interp1d(list(range(len(PC1bins))), PC1bins)
    pc2int = interpolate.interp1d(list(range(len(PC2bins))), PC2bins)

    
    #create the name of the current mesh
    digitlist = re.findall(r'\d', np.array2string(fourcorners))
    dash = ['-'.join(digitlist[x:x+2]) for x in list(range(0,len(digitlist),2))]
    underscore = '_'.join(dash)
    loopname = 'loop_'+underscore


    #make the directory to put the meshes in
    longsave = savedir+'contours/'+loopname+'/'
    if not os.path.exists(longsave):
        os.makedirs(longsave)
    
    

    for i in interarray:
        mesh_from_bins(
                        i,
                        whichpcs,
                        avgpcs,
                        pc1int,
                        pc2int,
                        pca,
                        longsave,
                        lmax)

    return interarray, loopname



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