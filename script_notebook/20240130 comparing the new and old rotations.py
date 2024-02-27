# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:58:04 2024

@author: Aaron
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
from scipy import interpolate, signal
from scipy.spatial import distance
from cmocean import cm
from scipy.spatial.transform import Rotation as R
from CustomFunctions import shtools_mod
import vtk
from aicsimageio.readers.tiff_reader import TiffReader
from vtk.util import numpy_support
from itertools import groupby
from operator import itemgetter
import seaborn as sns
from skimage.filters import median, gaussian
import re
from aicssegmentation.core.utils import hole_filling, topology_preserving_thinning
from aicssegmentation.core.vessel import filament_2d_wrapper, filament_3d_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_slice_by_slice, image_smoothing_gaussian_3d
from aicsimageio.writers import OmeTiffWriter
from skimage.morphology import medial_axis
from scipy.ndimage import distance_transform_edt
from skimage.morphology import erosion, disk

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
row = stdf.loc[8]
print(row.CellID)
#get the data related to this run of this cell
data = TotalFrame[(TotalFrame.CellID==row.CellID) & (TotalFrame.frame.isin(row.actual_run))].sort_values('frame').reset_index(drop=True)



#NEW ROTATION DATA
newdata = NewTotalFrame[(NewTotalFrame.CellID==row.CellID) & (NewTotalFrame.frame.isin(row.actual_run))].sort_values('frame').reset_index(drop=True)



####### compare the rotation angles of the different rotation methods
widths = np.array(data.Width_Rotation_Angle)
widths[widths>180] -= 360
plt.plot(widths)
newwidths = np.array(newdata.Width_Rotation_Angle)
newwidths[newwidths>180] -= 360
plt.plot(newwidths)
plt.show()


####### compare the PC values AND width rotation angles
fig, axes = plt.subplots(3,1)
axes[0].plot(widths)
axes[0].set_ylabel('Normal Rotation\nAngle', fontsize=12, labelpad = -6)
axes[1].plot(data.PC1)
axes[1].set_ylabel('PC1', fontsize=16)
axes[2].plot(data.PC2)
axes[2].set_ylabel('PC2', fontsize=16)
axes[2].set_xlabel('Frame #', fontsize=16)
plt.show()
fig.legend()
plt.savefig('C:/Users/Aaron/Documents/Python Scripts/temp/20231113_488EGFP-CAAX_0.1percentDMSO_3_cell_20_oldrotation.png')


####### compare the PC values of
fig, axes = plt.subplots(2,1)
axes[0].plot(data.PC1)
axes[0].plot(newdata.PC1)
axes[1].plot(data.PC2)
axes[1].plot(newdata.PC2)
plt.show()


####### compare the PC values AND width rotation angles
fig, axes = plt.subplots(3,1)
axes[0].plot(data.PC1)
axes[0].plot(newdata.PC1)
axes[1].plot(data.PC2)
axes[1].plot(newdata.PC2)
axes[2].plot(widths)
axes[2].plot(newwidths)
plt.show()


####### compare the PC values AND width rotation angles
fig, axes = plt.subplots(3,1)
axes[0].plot(widths, label='original')
axes[0].plot(newwidths, label = 'new')
axes[0].set_ylabel('Normal Rotation\nAngle', fontsize=12, labelpad = -6)
axes[1].plot(data.PC1)
axes[1].plot(newdata.PC1)
axes[1].set_ylabel('PC1', fontsize=16)
axes[2].plot(data.PC2)
axes[2].plot(newdata.PC2)
axes[2].set_ylabel('PC2', fontsize=16)
axes[2].set_xlabel('Frame #', fontsize=16)
plt.show()
fig.legend()
plt.savefig('C:/Users/Aaron/Documents/Python Scripts/temp/20231113_488EGFP-CAAX_0.1percentDMSO_3_cell_20_rotationcomparison.png')

####### compare the PC values AND width rotation angles AND speed
fig, axes = plt.subplots(4,1)
axes[0].plot(data.PC1)
axes[0].plot(newdata.PC1)
axes[1].plot(data.PC2)
axes[1].plot(newdata.PC2)
axes[2].plot(widths)
axes[2].plot(newwidths)
axes[3].plot(data.speed)
axes[3].plot(newdata.speed)
plt.show()


####### compare the PC BIN values with widths
fig, axes = plt.subplots(3,1)
axes[0].plot(data.PC1bins)
axes[0].plot(newdata.PC1bins)
axes[1].plot(data.PC2bins)
axes[1].plot(newdata.PC2bins)
axes[2].plot(widths)
axes[2].plot(newwidths)
plt.show()


############## compare 2D trajectories

norm = matplotlib.colors.Normalize()
cmm = cm.dense
nbins = 11
fig, axes = plt.subplots(1, 2)
for i, ax in enumerate(axes):
    if i == 0:
        plotdata = data.copy()
    if i == 1:
        plotdata = newdata.copy()
    #add "grid lines" first 
    for h in np.linspace(0.5, nbins+0.5, nbins+1):
        ax.axhline(h, linestyle='-', color='grey', alpha=0.3) # horizontal lines
        ax.axvline(h, linestyle='-', color='grey', alpha=0.3) # vertical lines
    
    #interpolate along the trajectory so I can plot dots which will represent the color gradient line
    px = plotdata.PC1bins.to_numpy()
    py = plotdata.PC2bins.to_numpy()
    pz = plotdata.frame.to_numpy()
    dist = np.nansum(distance.pdist(data[['PC1bins','PC2bins']]))
    fx = interpolate.interp1d(np.arange(1,len(px)+1),px)
    newx = fx(np.arange(1,len(px), ((len(px)+1)-1)/(5*dist)))
    fy = interpolate.interp1d(np.arange(1,len(py)+1),py)
    newy = fy(np.arange(1,len(py), ((len(py)+1)-1)/(5*dist)))
    newz = np.arange(0,len(data), len(data)/len(newy))
    if len(newz)>len(newy):
        newz = newz[:-1]
    #normalize to the colors to the length of the trajectory
    norm.autoscale([0,newz.max()])
    #plot the actual transitions
    ax.scatter(newx,newy, color = cmm(norm(newz)), alpha = 0.25, edgecolors='none')
    
    
    ax.set_aspect("equal")
    ax.set_xlabel('PC1')#, fontsize = 26)
    ax.set_ylabel('PC2')#, fontsize = 26)
    # ax.set_xticks(list(range(1,nbins+1)),[round((PC1bins[i+1]+x)/2,1) for i,x in enumerate(PC1bins[:-1])], fontsize = 14)
    # ax.set_yticks(list(range(1,nbins+1)),[round((PC2bins[i+1]+x)/2,1) for i,x in enumerate(PC2bins[:-1])], fontsize = 14)
    ax.set_xlim(0,nbins+1)
    ax.set_ylim(0,nbins+1)
    
    # #plot the interpolated transitions
    # ix = np.append(interdata.from_x.to_numpy(), interdata.to_x.to_numpy()[-1])
    # iy = np.append(interdata.from_y.to_numpy(), interdata.to_y.to_numpy()[-1])
    # ax.plot(ix, iy, color ='black', alpha=0.2)
    
    
    # cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmm),shrink=0.8)
    # cb.ax.tick_params(labelsize=16)
plt.show()











curdir = 'D:/Aaron/Data/CK666/Processed_Data/Meshes/'
meshfl = [x for x in os.listdir(curdir) if '20231106_488EGFP-CAAX_0.05percentDMSO_2_cell_6' in x]
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
both = np.concatenate((ALLWIDTHS[67], ALLWIDTHS[67]))
gau = gaussian_filter1d(both,20)
plt.plot(both)
plt.plot(gau)
peaks, properties = signal.find_peaks(abs(both),prominence=0.13, width=70)
angpeaks = np.concatenate((angles,angles))[peaks]



dirrrr = 'D:/Aaron/Data/Galvanotaxis_Confocal_40x_30C_10s/Processed_Data/'
xyres = 0.3394
zstep = 0.7
str_name = ''
exceptions_list = []
normal_rotation_method = []
l_order = 10
nisos = [1,63]
sigma = 0
align_method = 'trajectory'


seglist = [x for x in os.listdir(dirrrr) if '20231019_488EGFP-CAAX_1_cell_70' in x and 'segmented' in x]
seglist.sort(key=lambda x: float(re.findall('(?<=frame_)\d*', x)[0]))
#limit it to only consecutive runs:
seglist = [x for x in seglist if float(re.findall('(?<=frame_)\d*', x)[0]) in row.actual_run]
allmaxwidths = []
ALLWIDTHS = []
for i in seglist:
    impath = dirrrr + i
    cell_name = impath.split('/')[-1].split('_segmented')[0]
    print(impath)
    #read image
    im = TiffReader(impath)
    
    #read euler angles for alignment
    infopath = '/'.join(impath.split('/')[:-1]) + '/' + cell_name + '_cell_info.csv'
    #if align_method is a numpy array, use that as the vector to align to
    if type(align_method) == np.ndarray:
        vec = align_method.copy()
        #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        xaxis = np.array([[0,0,0],[1,0,0], [5,0,0]]).astype('float64')
        current_vec = np.stack(([0,0,0],vec), axis = 0)
        current_vec = np.concatenate((current_vec,[5*vec]), axis = 0)
        rotationthing = R.align_vectors(xaxis, current_vec)
        #below is actual rotation matrix if needed
        #rot_mat = rotationthing[0].as_matrix()
        rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)
        euler_angles = np.array([rotthing_euler[0], rotthing_euler[1], rotthing_euler[2]])
    elif align_method == 'trajectory':
        info = pd.read_csv(infopath, index_col=0)
        vec = np.array([info.Trajectory_X[0], info.Trajectory_Y[0], info.Trajectory_Z[0]])
        #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        xaxis = np.array([[0,0,0],[1,0,0], [5,0,0]]).astype('float64')
        current_vec = np.stack(([0,0,0],vec), axis = 0)
        current_vec = np.concatenate((current_vec,[5*vec]), axis = 0)
        rotationthing = R.align_vectors(xaxis, current_vec)
        #below is actual rotation matrix if needed
        #rot_mat = rotationthing[0].as_matrix()
        rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)
        euler_angles = np.array([rotthing_euler[0], rotthing_euler[1], rotthing_euler[2]])
        
    if len(im.shape)>3:
        ci = im.data[0,:,:,:]
        si = im.data[1,:,:,:]
    else:
        ci = im.data
        
        
    # Converting the input image into a mesh using regular marching cubes
    mesh, image_, centroid = shtools_mod.get_mesh_from_image(image=ci, sigma=sigma)
    
    #rotate and scale mesh
    #worked from https://kitware.github.io/vtk-examples/site/Python/PolyData/AlignTwoPolyDatas/
    #rotate around z axis
    transformation = vtk.vtkTransform()
    #rotate the shape
    transformation.RotateWXYZ(euler_angles[2], 0, 0, 1)
    transformation.RotateWXYZ(euler_angles[0], 1, 0, 0)
    #set scale to actual image scale
    transformation.Scale(xyres, xyres, zstep)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transformation)
    transformFilter.SetInputData(mesh)
    transformFilter.Update()
    mesh = transformFilter.GetOutput()
    
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
    # gau = gaussian_filter1d(both,20)
    peaks, properties = signal.find_peaks(abs(both),prominence=0.11, width=50)
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
    
    
    
for x in ALLWIDTHS[:100]:
    plt.plot(x)
plt.show()


widths = ALLWIDTHS[-5]
both = np.concatenate((widths, widths))
plt.plot(both)
plt.plot(widths)
gau = gaussian_filter1d(both,20)
plt.plot(gau)
peaks, properties = signal.find_peaks(abs(both),prominence=0.13, width=70)
angpeaks = np.concatenate((angles,angles))[peaks]
tangpeaks = angpeaks.copy()
tangpeaks[tangpeaks>180] -= 360
print(peaks, tangpeaks, angpeaks[np.argmin(abs(tangpeaks))])



alltangs = []
for x in ALLWIDTHS:
    both = np.concatenate((x, x))
    peaks, properties = signal.find_peaks(abs(both),prominence=0.11, width=55)
    angpeaks = np.concatenate((angles,angles))[peaks]
    tangpeaks = angpeaks.copy()
    # tangpeaks[tangpeaks>180] -= 360
    tangpeaks = list(set(tangpeaks))
    alltangs.append(tangpeaks)





seeds = []
allallmins = []
for s in alltangs[0]:
    allmins = [s]
    for wp in alltangs[1:]:
        if bool(len(wp) == 0):
            allmins.append(allmins[-1])
        else:
            allmins.append(wp[np.argmin(abs(wp-(allmins[-1])))])
    allallmins.append(allmins)
    seeds.append(np.sum(abs(np.diff(allmins))))
allallmins[np.argmin(seeds)]
    
    
allmins = []
for y, wp in enumerate(alltangs):
    #remove duplicates and sort
    # wp = np.sort(np.array(list(set(wp))))
    if bool(len(wp) == 0):
        allmins.append(allmins[-1])
    elif y == 0:
        # allmins.append(wp[np.argmin(abs(wp))])
        allmins.append(wp[1])
    else:
        allmins.append(wp[np.argmin(abs(wp-(allmins[-1])))])
plt.plot(allmins)



####### differential look-ahead?
allmins = []

for y, wp in enumerate(alltangs):
    #remove duplicates and sort
    wp = np.sort(np.array(list(set(wp))))
    if bool(wp.size == 0):
        allmins.append(allmins[-1])
    elif y == 0:
        # allmins.append(wp[np.argmin(abs(wp))])
        allmins.append(wp[0])
    else:
        for w in wp:
            
        allmins.append(wp[np.argmin(abs(wp-(allmins[-1])))])
plt.plot(allmins)







allmins = []
for y, wp in enumerate(alltangs):
    if bool(wp.size == 0):
        allmins.append(allmins[-1])
    elif y == 0:
        allmins.append(wp[np.argmin(abs(wp))])
    else:
        #remove duplicates
        wp = np.array(list(set(wp)))
        #get difference in distances around circle
        dici = np.argsort(abs(np.cos(np.radians(wp))-np.cos(np.radians(allmins[-1]))))[::-1]
        if abs(wp[dici[0]]-allmins[-1])<120:
            allmins.append(wp[dici[0]])
        else:
            allmins.append(wp[dici[1]])
            
        allmins.append(wp[np.argmin(abs(np.cos(np.radians(wp))-np.cos(np.radians(allmins[-1]))))])
        
        np.argsort(abs(np.cos(np.radians(wp))-np.cos(np.radians(allmins[-1]))))[::-1]




########### 2D plot of ALLWIDTHS!!!!!!!!!
#duplicate the phase
phase2 = np.zeros((len(ALLWIDTHS),2*len(ALLWIDTHS[0])))
for x in range(len(ALLWIDTHS)):
    phase2[x,:] = np.concatenate((ALLWIDTHS[x],ALLWIDTHS[x]))
sns.heatmap(gaussian(phase2,2))

OmeTiffWriter.save(thresh_img,'C:/Users/Aaron/Desktop/rotation.tiff')


########## brute force the least different path
#remove duplicates from alltangs
newtangs = [list(set(x)) for x in alltangs]

        
from itertools import permutations
output = [[row[y] for row, y in zip(newtangs, permutation)]
          for permutation in permutations(range(len(newtangs)))]
        

[x for x in itertools.product(*newtangs[:4])]


## PARAMETERS for this step ##
intensity_scaling_param = [0]
gaussian_smoothing_sigma = 2
################################
# intensity normalization

struct_img = intensity_normalization(np.array(ALLWIDTHS), scaling_param=[0])
struct_img = abs(struct_img-1)
# smoothing with 2d gaussian filter 
structure_img_smooth = gaussian(struct_img, sigma=gaussian_smoothing_sigma)

# # step 1: Masked-Object (MO) Thresholding
# thresh_img = MO(structure_img_smooth, global_thresh_method='tri', object_minArea=750, local_adjust = 0.95)

thresh_img = structure_img_smooth.copy()
thresh_img[thresh_img<0.35] = 0
thresh_img[thresh_img>0] = 1

# Step 2: Perform topology-preserving thinning
thin_dist_preserve = 1
thin_dist = 50
seg = topology_preserving_thinning(thresh_img, thin_dist_preserve, thin_dist)



def twodthinning(bw, min_thickness=1, thin=1):
    bw = bw>0

    ctl = medial_axis(bw > 0)
    dist = distance_transform_edt(ctl == 0)
    safe_zone = dist > min_thickness + 1e-5

    rm_candidate = np.logical_xor(bw > 0, erosion(bw > 0, disk(thin)))

    bw[np.logical_and(safe_zone, rm_candidate)] = 0

    return bw

padded = np.pad(thresh_img,(2,2), 'constant', constant_values = 0)
twod = twodthinning(thresh_img, thin_dist_preserve, thin_dist)
twod = twod.astype(np.uint8)
twod[twod > 0] = 255
OmeTiffWriter.save(twod,'C:/Users/Aaron/Desktop/rotation.tiff')

bigger = [ele for ele in list(thresh_img) for i in range(50)]
bigger = np.array(bigger)
OmeTiffWriter.save(bigger,'C:/Users/Aaron/Desktop/rotation.tiff')

thin_dist_preserve = 0.5
thin_dist = 200
twod = twodthinning(bigger, thin_dist_preserve, thin_dist)
twod = twod.astype(np.uint8)
twod[twod > 0] = 255
OmeTiffWriter.save(twod,'C:/Users/Aaron/Desktop/rotation.tiff')



ske = medial_axis(np.pad(bigger, ((0,0),(2,2,))))
ske = ske.astype(np.uint8)
ske[ske > 0] = 255
OmeTiffWriter.save(ske,'C:/Users/Aaron/Desktop/rotation.tiff')



skimske = skimage.morphology.skeletonize(thresh_img,method='lee') #np.pad(bigger,((0,0),(2,2))))



#correctly size image
arraywidths = np.array(ALLWIDTHS)
resized = skimage.transform.resize(arraywidths*100,(np.max(struct_img.shape),np.max(struct_img.shape)))
struct_img = intensity_normalization(resized, scaling_param=[0])
struct_img = abs(struct_img-1)
structure_img_smooth = gaussian(struct_img, sigma=gaussian_smoothing_sigma)
thresh_img = structure_img_smooth.copy()
thresh_img[thresh_img<0.35] = 0
thresh_img[thresh_img>0] = 1

newske = np.zeros(thresh_img.shape)
for x in range(thresh_img.shape[0]):
    strip = thresh_img[x,:]
    #get one indicies
    ones = np.where(strip==1)[0]
    runs = []
    for k, g in groupby(enumerate(ones), lambda ix: ix[0] - ix[1]):
        runs.append(list(map(itemgetter(1), g)))

    strippoints = [round((x[-1]+x[0])/2) for x in runs]
    newske[x,strippoints] = 1

for y in range(newske.shape[0]-1):
    cur = np.where(newske[y,:]==1)[0]
    nxt = np.where(newske[y+1,:]==1)[0]
    for x in cur:
        ran = np.sort([x,nxt[np.argmin(abs(x-nxt))]])
        newske[y,ran[0]:ran[1]] = 1
