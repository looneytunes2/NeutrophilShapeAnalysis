# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:43:06 2024

@author: Aaron
"""

from scipy import ndimage, signal


curdir = 'C:/Users/Aaron/Documents/Python Scripts/temp/Meshes/'
meshfl = [x for x in os.listdir(curdir) if '.vtp' in x]
meshfl.sort(key=lambda x: float(re.findall('(?<=frame_)\d*', x)[0]))
allangles = []
ALLANGLES = []
for m in meshfl:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(curdir + m)
    reader.Update()
    mesh = reader.GetOutput()
    
    #Get physical properties of cell
    CellMassProperties = vtk.vtkMassProperties()
    CellMassProperties.SetInputData(mesh)
    #rotate around the x axis until you find the widest distance in y
    angles = np.arange(0,360,0.5)
    widths = np.empty(len(angles))
    for i, a in enumerate(angles):
        
        transformation = vtk.vtkTransform()
        #rotate the shape
        transformation.RotateWXYZ(a, 1, 0, 0)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(mesh)
        transformFilter.Update()
        rotatedmesh = transformFilter.GetOutput()
        
        

        
        #store the volume in the positive y direction
        widths[i] = measure_volume_half(rotatedmesh, 'y')/CellMassProperties.GetVolume()
    
    #get the rotation angle with the most heavy negative y direction bias
    widestangle = angles[np.where(widths==widths.max())[0][0]]
    allangles.append(widestangle)
    ALLANGLES.append(widths)
    
    
    

curdir = 'C:/Users/Aaron/Documents/Python Scripts/temp/Meshes/'
meshfl = [x for x in os.listdir(curdir) if '.vtp' in x]
meshfl.sort(key=lambda x: float(re.findall('(?<=frame_)\d*', x)[0]))
allmaxwidths = []
ALLWIDTHS = []
for m in meshfl:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(curdir + m)
    reader.Update()
    mesh = reader.GetOutput()
    
    #rotate around the x axis until you find the widest distance in y
    angles = np.arange(0,360,0.5)
    widths = np.empty(len(angles))
    for i, a in enumerate(angles):
        
        transformation = vtk.vtkTransform()
        #rotate the shape
        transformation.RotateWXYZ(a, 1, 0, 0)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(mesh)
        transformFilter.Update()
        rotatedmesh = transformFilter.GetOutput()
        coords = numpy_support.vtk_to_numpy(rotatedmesh.GetPoints().GetData())
        #store the average of the negative y coordinates
        widths[i] = coords[np.where(coords[:,1]<0)][:,1].mean()
    
    #get the rotation angle with the most heavy negative y direction bias
    widestangle = angles[np.where(widths==widths.min())[0][0]]
    allmaxwidths.append(widestangle)
    ALLWIDTHS.append(widths)
    
    
diffs 
    
adjwidths = np.array(allmaxwidths)
adjwidths[adjwidths>180] -= 360
    
###############3 new method!



curdir = 'C:/Users/Aaron/Documents/Python Scripts/temp/Meshes/'
meshfl = [x for x in os.listdir(curdir) if '.vtp' in x]
meshfl.sort(key=lambda x: float(re.findall('(?<=frame_)\d*', x)[0]))
allmaxwidths = []
ALLWIDTHS = []
for m in meshfl:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(curdir + m)
    reader.Update()
    mesh = reader.GetOutput()
    
    #rotate around the x axis until you find the widest distance in y
    angles = np.arange(0,360,0.5)
    widths = np.empty(len(angles))
    for i, a in enumerate(angles):
        
        transformation = vtk.vtkTransform()
        #rotate the shape
        transformation.RotateWXYZ(a, 1, 0, 0)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(mesh)
        transformFilter.Update()
        rotatedmesh = transformFilter.GetOutput()
        coords = numpy_support.vtk_to_numpy(rotatedmesh.GetPoints().GetData())
        #store the average of the negative y coordinates
        widths[i] = coords[np.where(coords[:,1]<0)][:,1].mean()
    
    #get the rotation angle with the most heavy negative y direction bias
    both = np.concatenate((widths, widths))
    gau = gaussian_filter1d(both,20)
    peaks, properties = signal.find_peaks(abs(both),prominence=0.13, width=70)
    angpeaks = np.concatenate((angles,angles))[peaks]
    tangpeaks = angpeaks.copy()
    tangpeaks[tangpeaks>180] -= 360
    widestangle = angpeaks[np.argmin(abs(tangpeaks))]
    # widestangle = angles[np.where(widths==widths.min())[0][0]]
    allmaxwidths.append(widestangle)
    ALLWIDTHS.append(widths)
    
    
    
adjwidths = np.array(allmaxwidths)
adjwidths[adjwidths>180] -= 360
prevadjwidths = adjwidths.copy()



#get the rotation angle with the most heavy negative y direction bias
both = np.concatenate((ALLWIDTHS[25], ALLWIDTHS[25]))
gau = gaussian_filter1d(both,20)
plt.plot(both)
plt.plot(gau)
peaks, properties = signal.find_peaks(abs(both),prominence=0.13, width=70)
angpeaks = np.concatenate((angles,angles))[peaks]