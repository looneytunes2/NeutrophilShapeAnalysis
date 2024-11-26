# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:21:03 2024

@author: Aaron
"""
import os
import numpy as np
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader
from aicscytoparam import cytoparam
from CustomFunctions.PCvisualization import animate_PCs
from CustomFunctions.shtools_mod import get_even_reconstruction_from_coeffs
from CustomFunctions.PCvisualization import save_mesh
from CustomFunctions import PILRagg
import multiprocessing
import pickle as pk
from pathlib import Path
import math
import scipy
import vtk
import re

def collect_results(result):
    """Uses apply_async's callback to setup up a separate Queue for each process.
    This will allow us to collect the results from different threads."""
    results.append(result)

def angle360(x,y):
    return abs(math.degrees(np.arctan2(y,x))) if y < 0 else abs(math.degrees(np.arctan2(y,x))-360)

############## get 1D cycle for PC dynamics WITHOUT USING CGPS BINS ##############
def linearize_cycle_continuous(
        df, #dataframe with the PCs in format 'PC_'
        centers, #dataframe with center values of the CGPS bins for the PCs
        origin = list, #list of the coordinates in the CGPS in format [x,y]
        whichpcs = list, #list of which PCs are represented by x and y in format [x,y] 
        binrange = int, #how big are the radial bins in degrees
        direction = str, #which direction is the flux 'clockwise' or 'counterclockwise'
        ):
    
    ### get centered PC bins first
    x = df[f'PC{whichpcs[0]}'].values-centers[f'PC{whichpcs[0]}'].iloc[origin[0]-1]
    y = df[f'PC{whichpcs[1]}'].values-centers[f'PC{whichpcs[1]}'].iloc[origin[1]-1]
    ### calculate angular coord and radius
    df[f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Coord'] = [angle360(x1, y1) for x1, y1 in zip(x, y)]
    df[f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Radial_Coord'] = np.sqrt((x**2) + (y**2))
    ### "bin" the Angular coord
    if direction == 'clockwise':
        df[f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins'] = np.digitize(df[f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Coord'].to_numpy(),
                               np.arange(0,360+binrange,binrange))
    elif direction == 'counterclockwise':
        df[f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins'] = np.digitize(df[f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Coord'].to_numpy(),
                               np.arange(0,360+binrange,binrange)[::-1])
    df[f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins'] = (df[f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins'] * binrange) - binrange
    
    return df
    

############### get animated representation of ALL PCs around average 1D cycle #############
def animate_linear_cycle_PCs(
        df,
        savedir,
        whichpcs = list, #list of which PCs are represented by x and y in format [x,y] 
        binrange = int, #how big are the radial bins in degrees
        lmax:int = 10,
        ):

    
    #make the directory to save this combined image
    specificdir = savedir +f'/PC{whichpcs[0]}_PC{whichpcs[1]}_Cycle_AllPC_Visualization/'
    if not os.path.exists(specificdir):
        os.makedirs(specificdir)
    
    #get the average of each PC
    avgpcs = df[[x for x in df.columns.to_list() if 'PC' in x and 'bin' not in x and '_' not in x]].mean().to_numpy()
    #open the actual pca file used to analyze this dataset
    pca = pk.load(open(str(Path(savedir).parents[0])+"/pca.pkl",'rb'))
    #define which PCs to incorporate into the view
    whichpcs = list(range(1,len(avgpcs)+1))

    
    #use 1D gaussian smoothening to get average PC curves over the 1D cycle
    allinterpvals = []
    for w in whichpcs:
        ra = df[[f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins',f'PC{w}']].groupby(f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins').mean().reset_index()
        pre = ra.values[-5:]
        pre[:,0] -=360
        post = ra.values[:5]
        post[:,0] += 360
        newra = np.vstack((pre, ra.values, post))
        f = scipy.interpolate.interp1d(newra[:,0], scipy.ndimage.gaussian_filter1d(newra[:,1],1))
        interpvals = f(np.arange(0,360,binrange))
        allinterpvals.append(interpvals)
    
    for i, a in enumerate(np.array(allinterpvals).T):
        cursavedir = specificdir + f'frame_{int(i)}_mesh.vtp'
        animate_PCs(avgpcs,
                    whichpcs,
                    a,
                    pca,
                    cursavedir,
                    lmax,)




############### get animated representation of USING AVERAGE SHCOEFFS around average 1D cycle #############

def animate_linear_cycle_shcoeffs(
        df, #pandas dataframe with the CGPS angular coordinates
        coeff_df, #pandas dataframe with the shcoeffs
        savedir,
        whichpcs = list, #list of which PCs are represented by x and y in format [x,y] 
        binrange = int, #how big are the radial bins in degrees
        lmax: int=10,
        ):

    
    coeffframe = df.merge(coeff_df[[x for x in coeff_df.columns.to_list() if 'shco' in x]+['cell']],
                                  left_on = 'cell', right_on = 'cell')
    
    #make the directory to save this combined image
    specificdir = savedir +f'/PC{whichpcs[0]}-PC{whichpcs[1]}_Cycle_AllSHCoeff_Visualization/'
    if not os.path.exists(specificdir):
        os.makedirs(specificdir)
    
    #use 1D gaussian smoothening to get average PC curves over the 1D cycle
    allinterpvals = []
    for s in [x for x in coeffframe if 'shco' in x]:
        ra = coeffframe[[f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins',s]].groupby(f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins').mean().reset_index()
        pre = ra.values[-5:]
        pre[:,0] -= 360
        post = ra.values[:5]
        post[:,0] += 360
        newra = np.vstack((pre, ra.values, post))
        f = scipy.interpolate.interp1d(newra[:,0], scipy.ndimage.gaussian_filter1d(newra[:,1],1))
        interpvals = f(np.arange(0,360,binrange))
        allinterpvals.append(interpvals)
    
    for i, a in enumerate(np.array(allinterpvals).T):
        mesh, _ = get_even_reconstruction_from_coeffs(np.reshape(a, (2,lmax+1,lmax+1)), lmax)
        save_mesh(mesh, specificdir + f'frame_{int(i)}_mesh.vtp')
    




############### get animated representation of USING AVERAGE SHCOEFFS around average 1D cycle #############

def get_linear_cycle_PILRs(
        df, #pandas dataframe with the CGPS angular coordinates
        savedir,
        pilr_fl, #directory of PILR images
        recontype: str='SH', #what metrics were used for reconstruction SH (spherical harmonic coefficients) or PC 
        lmax: int=10,
        nisos: list = [1,63],
        ):
    
    #make tiny sphere for inner mesh
    #create inner sphere
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(0.0, 0.0, 0.0)
    sphereSource.SetRadius(nisos[0]/2)
    # Make the surface smooth.
    sphereSource.SetPhiResolution(100)
    sphereSource.SetThetaResolution(100)
    sphereSource.Update()
    spherepoly = sphereSource.GetOutput()

    #generate whichpcs from the angular bin column name in the dataframe
    whichpcs = [int(x) for x in re.findall(r'PC(\d+)', [x for x in df.columns.to_list() if 'Continuous_Angular_Bins' in x][0])]

    #make the directory to save this combined image
    if recontype == 'SH':
        specificdir = savedir +f'/PC{whichpcs[0]}-PC{whichpcs[1]}_Cycle_AllSHCoeff_Visualization/'
    elif recontype == 'PC':
        specificdir = savedir +f'/PC{whichpcs[0]}_PC{whichpcs[1]}_Cycle_AllPC_Visualization/'
    pilragg_fl = specificdir + 'avgPILRs/'
    if not os.path.exists(pilragg_fl):
        os.makedirs(pilragg_fl)
    #get the list of PILRs that actually exist
    allpilrfiles = os.listdir(pilr_fl)
    
    #make sure the dataframe is sorted by angular bins
    df = df.sort_values(f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins').reset_index()
    for t, treat in df.groupby('Treatment'):
        for s, struct in treat.groupby('structure'):
            for b, temp in struct.groupby(f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins'):
                framenum = list(df[f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins'].unique()).index(b)
                #get the list of pilr file names if they exist
                pilrlist = [pilr_fl + n + '_PILR.ome.tiff' for n in temp.cell.to_list() if n + '_PILR.ome.tiff' in allpilrfiles]
                # use multiprocessing to perform segmentation and x,y,z determination
                pool = multiprocessing.Pool(processes=60)
                results = pool.map(PILRagg.read_parameterized_intensity, pilrlist)
                pool.close()
                pool.join()
                pagg = np.array(results)
                #get average representation
                pagg_avg = np.mean(pagg, axis = 0)
                #normalize representations
                pagg_norm = PILRagg.normalize_representations(pagg)
                pagg_norm_avg = np.mean(pagg_norm, axis = 0)
                dims = [['X', 'Y', 'Z', 'C', 'T'][d] for d in range(pagg_avg.ndim)]
                dims = ''.join(dims[::-1])
                OmeTiffWriter.save(pagg_avg, pilragg_fl+f'frame_{b}_{t}_{s}_repsagg.tif', dim_order=dims)
                OmeTiffWriter.save(pagg_norm_avg, pilragg_fl+f'frame_{b}_{t}_{s}_repsagg_norm.tif', dim_order=dims)
        
        
                mesh_outer = PILRagg.read_vtk_polydata(specificdir+f'frame_{framenum}_mesh.vtp')
                domain, origin = cytoparam.voxelize_meshes([mesh_outer, spherepoly])
                coords_param, _ = cytoparam.parameterize_image_coordinates(
                    seg_mem=(domain>0).astype(np.uint8),
                    seg_nuc=(domain>1).astype(np.uint8),
                    lmax=lmax,
                    nisos=nisos
                )
        
                morphed = cytoparam.morph_representation_on_shape(
                            img=domain,
                            param_img_coords=coords_param,
                            representation=pagg_avg)
                morphed = np.stack([domain, morphed])
                OmeTiffWriter.save(morphed, pilragg_fl+f'frame_{b}_{t}_{s}_aggmorph.tif', dim_order='CZYX')
                print(f'Finished frame_{b}_{t}_{s}')
    

def combine_linear_PILRs(savedir,
                         structure,
                         whichpcs,
                         projtype:str = 'sum',
                         ):
    #get linear cycle folder
    avgPILRs = savedir + f'/PC{whichpcs[0]}-PC{whichpcs[1]}_Cycle_AllSHCoeff_Visualization/avgPILRs/'
    #get all structure images
    structlist = [x for x in os.listdir(avgPILRs) if structure in x and 'aggmorph' in x]
    #sort the structlist by frame
    structlist.sort(key=lambda x: float(re.findall('(?<=frame_)\d*', x)[0]))
    #read all of the frames into a list
    imlist = []
    for s in structlist:
        imlist.append(OmeTiffReader(avgPILRs + s).data[0])
    #get max dimensions of all the images
    maxshape = np.zeros((len(imlist),4))
    for i, ii in enumerate(imlist):
        maxshape[i,:] = ii.shape
    target_shape = np.max(maxshape, axis = 0)
    
    #### insert images into a the larger image
    fullimage = np.zeros(np.concatenate(([len(imlist)], target_shape)).astype(int))
    for i, ii in enumerate(imlist):
        # Calculate padding
        pad_x = int((target_shape[-1] - ii.shape[-1]) // 2)
        pad_y = int((target_shape[-2] - ii.shape[-2]) // 2)
        pad_z = int((target_shape[-3] - ii.shape[-3]) // 2)
        
        
        fullimage[i,:,
                  pad_z:int(pad_z + ii.shape[-3]),
                  pad_y:int(pad_y + ii.shape[-2]),
                  pad_x:int(pad_x + ii.shape[-1]),
                  ] = ii
    if projtype == 'sum':
        projim = np.sum(fullimage, axis = 2)
        OmeTiffWriter.save(projim, savedir + structure + '_sum_PILR.ome.tiff', dim_order='TCYX')
    elif projtype == 'max':
        projim = np.max(fullimage, axis = 2)
        OmeTiffWriter.save(projim, savedir + structure + '_max_PILR.ome.tiff', dim_order='TCYX')
        
        