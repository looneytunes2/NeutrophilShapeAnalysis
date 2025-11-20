# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:12:09 2025

@author: Aaron
"""

from vtk.util import numpy_support as vtknp
import numpy as np
import vtk
import math
import operator
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from vtkmodules.vtkFiltersCore import (
    vtkCleanPolyData,
    vtkTriangleFilter
)


meshdir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/Data_and_Figs/PC_Meshes/'


reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(meshdir + 'Cell_PC7_5_Cell.vtp')
reader.Update()
mesh = reader.GetOutput()

tri1 = vtkTriangleFilter()
tri1.SetInputData(mesh)
clean1 = vtkCleanPolyData()
clean1.SetInputConnection(tri1.GetOutputPort())
clean1.Update()
mesh = clean1.GetOutput()



##### FROM ALLEN INSTITUTE
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








coords = find_plane_mesh_intersection(mesh, [0,2], use_vtk_for_intersection=True)
#center coords on zero
coords = coords - np.mean(coords,axis = 0)
#shift the coords up so I can project onto the xaxis
xmin, ymin = np.min(coords,axis = 0)
coords[:,0] = coords[:,0] + abs(xmin) + 1
coords[:,1] = coords[:,1] + abs(ymin) + 1


# Create a figure and axis
fig, ax = plt.subplots()

# Create a Polygon patch
polygon = patches.Polygon(coords, closed=True, edgecolor='0.4', facecolor='0.8', linewidth=3)

# Add the polygon to the plot
ax.add_patch(polygon)


#add the "x" axis points
#colors
point_colors = plt.cm.Dark2.colors[:3]
# point_colors = point_colors*2
#points
proj_points = np.array([[np.mean(coords,axis = 0)[0],0], #centroid on x axis
                        [np.max(coords,axis=0)[0],0], #front on x axis
                        [np.min(coords,axis=0)[0],0], #rear on x axis
                        np.mean(coords,axis=0), #centroid
                        [coords[np.argmax(coords[:,0])][0],coords[np.argmax(coords[:,0])][1]], #front
                        [coords[np.argmin(coords[:,0])][0],coords[np.argmin(coords[:,0])][1]] #rear
                        ])
ax.scatter(proj_points[3:,0], proj_points[3:,1],s = 150, color = point_colors, edgecolors = 'black', zorder=10)



### add the points projected onto the x axis
proj_cent = patches.Circle((proj_points[0,0], 0), 0.25, facecolor=point_colors[0],
                        edgecolor = 'black', clip_on=False, zorder = 0)
ax.add_patch(proj_cent)
proj_front = patches.Circle((proj_points[1,0], 0), 0.25, facecolor=point_colors[1],
                        edgecolor = 'black', clip_on=False, zorder = 0)
ax.add_patch(proj_front)
proj_rear = patches.Circle((proj_points[2,0], 0), 0.25, facecolor=point_colors[2],
                        edgecolor = 'black', clip_on=False, zorder = 0)
ax.add_patch(proj_rear)



#arrow properties
arrowdict = dict(facecolor=point_colors[0], arrowstyle="simple, tail_width = 0.5, head_length=1.25, head_width=1.25", linewidth=1)
#centroid arrow
ax.annotate("", xy=(np.mean(coords,axis = 0)[0],0),xytext = np.mean(coords,axis=0),
            arrowprops=arrowdict, zorder = 5)
#rear arrow
arrowdict = dict(facecolor=point_colors[2], arrowstyle="simple, tail_width = 0.5, head_length=1.25, head_width=1.25", linewidth=1)
ax.annotate("", xy=(coords[np.argmin(coords[:,0])][0], 0),xytext =(coords[np.argmin(coords[:,0])][0],coords[np.argmin(coords[:,0])][1]),
            arrowprops=arrowdict, zorder = 5)
#front arrow
arrowdict = dict(facecolor=point_colors[1], arrowstyle="simple, tail_width = 0.5,  head_length=1.25, head_width=1.25", linewidth=1)
ax.annotate("", xy=(coords[np.argmax(coords[:,0])][0], 0),xytext =(coords[np.argmax(coords[:,0])][0],coords[np.argmax(coords[:,0])][1]),
            arrowprops=arrowdict, zorder = 5)





#Big trajectory arrow
bigarrowdict = dict(facecolor='black', arrowstyle="simple, head_length=1.25, head_width=1.25", linewidth=3)
ax.annotate("", xy=(coords[np.argmax(coords[:,0])][0], 9.5),xytext =(coords[np.argmax(coords[:,0])][0]-4, 9.5),
            arrowprops=bigarrowdict)
#trajectory label
#xaxis text
ax.text(coords[np.argmax(coords[:,0])][0]-4, 10.25, 'trajectory', va='center', ha='left', fontsize=20)


#### set ax labels
ax.set_xlabel('x-axis (μm)', fontsize = 20)
ax.set_ylabel('z-axis (μm)', fontsize = 20)


# Set limits and aspect ratio
# ax.set_xlim(-0.15, 17)
ax.set_ylim(0, 10.5)
ax.set_aspect('equal')  # Ensures equal scaling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# Show the plot
plt.tight_layout()








#save
plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')
