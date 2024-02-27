# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 18:58:23 2023

@author: Aaron
"""

import vtk
import numpy as np
from vtk.util import numpy_support
import random



#create inner sphere
sphereSource = vtk.vtkSphereSource()
sphereSource.SetCenter(0.0, 0.0, 0.0)
sphereSource.SetRadius(nisos[0]/2)
# Make the surface smooth.
sphereSource.SetPhiResolution(100)
sphereSource.SetThetaResolution(100)
sphereSource.Update()
spherepoly = sphereSource.GetOutput()





# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import (
    vtkIdList,
    vtkPoints
)
from vtkmodules.vtkCommonDataModel import (
    VTK_POLYHEDRON,
    vtkUnstructuredGrid
)
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkDataSetMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)



colors = vtkNamedColors()

# create polyhedron (cube)
# The point Ids are: [0, 1, 2, 3, 4, 5, 6, 7]

points = vtkPoints()
points.InsertNextPoint(-5.0, -2.0, -3.0)
points.InsertNextPoint(5.0, -2.0, -3.0)
points.InsertNextPoint(5.0, 2.0, -3.0)
points.InsertNextPoint(-5.0, 2.0, -3.0)
points.InsertNextPoint(-5.0, -2.0, 3.0)
points.InsertNextPoint(5.0, -2.0, 3.0)
points.InsertNextPoint(5.0, 2.0, 3.0)
points.InsertNextPoint(-5.0, 2.0, 3.0)

# These are the point ids corresponding to each face.
faces = [[0, 3, 2, 1], [0, 4, 7, 3], [4, 5, 6, 7], [5, 1, 2, 6], [0, 1, 5, 4], [2, 3, 7, 6]]
faceId = vtkIdList()
faceId.InsertNextId(6)  # Six faces make up the cell.
for face in faces:
    faceId.InsertNextId(len(face))  # The number of points in the face.
    [faceId.InsertNextId(i) for i in face]

ugrid = vtkUnstructuredGrid()
ugrid.SetPoints(points)
ugrid.InsertNextCell(VTK_POLYHEDRON, faceId)

# Here we write out the cube.
writer = vtkXMLUnstructuredGridWriter()
writer.SetInputData(ugrid)
writer.SetFileName('C:/Users/Aaron/Desktop/polyhedron.vtu')
writer.SetDataModeToAscii()
writer.Update()


def mkVtkIdList(it):
    """
    Makes a vtkIdList from a Python iterable. I'm kinda surprised that
     this is necessary, since I assumed that this kind of thing would
     have been built into the wrapper and happen transparently, but it
     seems not.

    :param it: A python iterable.
    :return: A vtkIdList
    """
    vil = vtk.vtkIdList()
    for i in it:
        vil.InsertNextId(int(i))
    return vil


 # x = array of 8 3-tuples of float representing the vertices of a cube:
x = [(-5.0, -2.0, -3.0), (5.0, -2.0, -3.0), (5.0, 2.0, -3.0), (-5.0, 2.0, -3.0),
     (-5.0, -2.0, 3.0), (5.0, -2.0, 3.0), (5.0, 2.0, 3.0), (-5.0, 2.0, 3.0)]

# pts = array of 6 4-tuples of vtkIdType (int) representing the faces
#     of the cube in terms of the above vertices
pts = [(0, 3, 2, 1), (0, 4, 7, 3), (4, 5, 6, 7),
       (5, 1, 2, 6), (0, 1, 5, 4), (2, 3, 7, 6)]

# We'll create the building blocks of polydata including data attributes.
cube = vtk.vtkPolyData()
points = vtk.vtkPoints()
polys = vtk.vtkCellArray()
scalars = vtk.vtkFloatArray()

# Load the point, cell, and data attributes.
for i, xi in enumerate(x):
    points.InsertPoint(i, xi)
for pt in pts:
    polys.InsertNextCell(mkVtkIdList(pt))
for i, _ in enumerate(x):
    scalars.InsertTuple1(i, i)

# We now assign the pieces to the vtkPolyData.
cube.SetPoints(points)
cube.SetPolys(polys)
cube.GetPointData().SetScalars(scalars)


# Here we write out the cube.
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName('C:/Users/Aaron/Desktop/polyhedron.vtp')
writer.SetInputData(rotcube)
writer.Write()





#rotate the cube randomly
transformation = vtk.vtkTransform()
#rotate the shape
transformation.RotateWXYZ(random.randint(0,360), 0, 0, 1)
transformation.RotateWXYZ(random.randint(0,360), 1, 0, 0)
transformation.RotateWXYZ(random.randint(0,360), 0, 1, 0)
#set scale to actual image scale
transformFilter = vtk.vtkTransformPolyDataFilter()
transformFilter.SetTransform(transformation)
transformFilter.SetInputData(cube)
transformFilter.Update()
rotcube = transformFilter.GetOutput()





#Get physical properties of cell
CellMassProperties = vtk.vtkMassProperties()
CellMassProperties.SetInputData(cell_mesh)


cell_coords = numpy_support.vtk_to_numpy(rotcube.GetPoints().GetData())
cov = np.cov(cell_coords.T)
cell_evals, cell_evecs = np.linalg.eig(cov)
cell_sort_indices = np.argsort(cell_evals)[::-1]
rotationthing = R.align_vectors(np.array([[0,0,1],[0,1,0],[1,0,0]]), cell_evecs.T)
cell_coords = rotationthing[0].apply(cell_coords)



np.max(cell_coords[:,2])-np.min(cell_coords[:,2]),

np.max(cell_coords[:,1])-np.min(cell_coords[:,1]),

np.max(cell_coords[:,0])-np.min(cell_coords[:,0]),



np.array(np.where(ci>0))[::-1]

