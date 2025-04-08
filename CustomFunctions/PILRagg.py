# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 21:02:14 2022

@author: Aaron
"""
from pathlib import Path
import numpy as np
from aicsimageio import AICSImage
import tifffile
from aicsimageio.readers.tiff_reader import TiffReader
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
import itertools


def read_parameterized_intensity(im):
#     code, intensity_names = None, []
#     path = f"parameterization/representations/{index}.tif"
#     path = self.control.get_staging() / path

#     if path.is_file():
    code = AICSImage(im)
    code = code.data.squeeze()
    if code.ndim == 2:
        code = code.reshape(1, *code.shape)
#         if return_intensity_names:
#             return code, intensity_names
    return code

def normalize_representations(reps):
    # Expected shape is SCMN
    if reps.ndim != 4:
        raise ValueError(f"Input shape {reps.shape} does not match expected SCMN format.")
    count = np.sum(reps, axis=(-2,-1), keepdims=True)
    reps_norm = np.divide(reps, count, out=np.zeros_like(reps), where=count>0)
    return reps_norm



def write_ome_tif(path, img, channel_names=None, image_name=None):
    # path = Path(path)
    dims = [['X', 'Y', 'Z', 'C', 'T'][d] for d in range(img.ndim)]
    dims = ''.join(dims[::-1])
    name = path.stem if image_name is None else image_name
    OmeTiffWriter.save(img, path, dim_order=dims, image_name=name, channel_names=channel_names)
    return

def pilr_correlation(
        corrpair, #iterable len 2, [0] is "cell1" and [1] is the cell to correlate it with
        ):
    cell1 = tifffile.imread(corrpair[0])
    cell2 = tifffile.imread(corrpair[1])
    cor = np.corrcoef(cell1.flatten(), cell2.flatten())
    return [Path(corrpair[0]).stem.split('_PILR')[0], Path(corrpair[1]).stem.split('_PILR')[0], cor[0,1]]


def measure_pilr_regions(
        im,
        ):
    #get lmax from image shape
    #first and second indices don't count since they are north and south poles
    width = im.shape[-1]-2
    lmax = np.sqrt(width/32)
    #the latitude has 1/2 as many coordinates as the longitude and they are both
    #increased by a factor of two for reconstruction so...
    lon = int(lmax*8)
    lat = int(lmax*4)
    
    top = np.concatenate([np.arange(0,lat/2)+(lat*i) for i in range(lon)]).flatten()+2
    bottom = np.concatenate([np.arange(lat/2,lat)+(lat*i) for i in range(lon)]).flatten()+2
    left = np.arange(lon*lat/2, lon*lat)+2 # second half
    right = np.arange(0, lon*lat/2)+2 #first half
    front = np.arange(lon*lat/4,lon*lat*3/4)+2
    back  = np.concatenate((np.arange(0,lon*lat/4), np.arange(lon*lat*3/4,lon*lat)))+2
    
    #sum all of the radial info in the pilr
    sumd = np.sum(im,axis = 0)
    
    #get all pairs of locations
    regions = [front,back,top,bottom,left,right]
    regionkeys = ['front','back','top','bottom','left','right']
    opposites = [[0,1],[1,0],[2,3],[3,2],[4,5],[5,4]]
    pairs = list(itertools.combinations(enumerate(regions),2))
    thruples = list(itertools.combinations(enumerate(regions),3))
    
    #calculate all regions
    regiondict = {}
    for i, r in enumerate(regions):
        regiondict[regionkeys[i]] = np.mean(sumd[r.astype(int)])
    
    #calculate all pairs
    for f,s in pairs:
        r1 = regionkeys[f[0]] 
        r2 = regionkeys[s[0]]
        ar1 = f[1]
        ar2 = s[1]
        #avoid comparing opposites
        if [f[0],s[0]] not in opposites:
            #first in second
            fs = ar1[np.isin(ar1, ar2)]
            regiondict[r1+r2] = np.mean(sumd[fs.astype(int)])
        
    #calculate all thruples
    for f,s,t in thruples:
        r1 = regionkeys[f[0]] 
        r2 = regionkeys[s[0]]
        r3 = regionkeys[t[0]]
        ar1 = f[1]
        ar2 = s[1]
        ar3 = t[1]
        #avoid comparing opposites
        if all(np.array([sum(np.isin(a,[f[0],s[0],t[0]])) for a in opposites])<2):
            #first in second
            fs = ar1[np.isin(ar1, ar2)]
            #first and second in third
            fst = fs[np.isin(fs,ar3)]
            regiondict[r1+r2+r3] = np.mean(sumd[fst.astype(int)])
    
    return regiondict
  

def read_pilr_regions(
        imdir
        ):
    #get the name of the cell/frame
    cellname = Path(imdir).stem.split('_PILR')[0]
    #load the data
    im = TiffReader(imdir).data
    #measure the regions
    rd = measure_pilr_regions(im)
    #add cell and structure info
    rd['cell'] = cellname
    return rd
  

