# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:52:53 2022

@author: Aaron
"""

from aicsimageio.readers.tiff_reader import TiffReader
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader
from aicsimageio.readers.bioformats_reader import BioformatsReader
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
import numpy as np
import os
import re
import skimage.io


##### read single image and save as video with whole z volume
path = 'F:/sorted caax lines/20220804'
fl = os.listdir(path)
for f in fl:
    # f = fl[-1]
    imfl = os.listdir(path + '/' + f)
    tif = [x for x in imfl if x.endswith('.tif')][0]
    im = path + '/' + f + '/' + tif
    image = TiffReader(im)
    for i, s in enumerate(image.scenes):
        image.set_scene(s)
        OmeTiffWriter.save(image, im.split('Pos')[0] + f'Pos{i}.tif')




####### read images from Visiview and save as a maximum intensity projection
path = 'D:/Data/GalvanotaxisiSIM/20230126/'
fl = os.listdir(path)
for f in fl:
    # f = fl[-1]
    imfl = os.listdir(path + f)
    tif = [x for x in imfl if x.endswith('.tif')][0]
    im = path + '/' + f + '/' + tif
    image = BioformatsReader(im)
    maxx = np.amax(image.data, axis = 2)
    OmeTiffWriter.save(maxx, im.split('_w')[0] + '_maxproj.ome.tif', dim_order=('TCYX'))




######### opening images from multiple loops on visiview
path = 'D:/Aaron/Data/Chemotaxis/Raw_Data/20230510_488EGFP-CAAX_640JF646actin-halotag_ChemDirected2/'
savedir = 'C:/Users/Aaron/Documents/Python Scripts/temp/'
fl = os.listdir(path)
looplist = []
for f in fl:
    if re.findall('t\d*_l\d*', f) and f.split('_w')[0] not in looplist:
        looplist.append(f.split('_w')[0])                        
for l in looplist:

    files = [x for x in fl if l in x and '_t' in x]
    #sort by loop and time point
    files.sort(key = lambda x: (int(x.split('_l')[1].split('.ome')[0]), int(x.split('_t')[1].split('_l')[0])))
    shape = TiffReader(path+files[0]).shape
    whole = np.zeros((2,int(len(files)/2),shape[-3],shape[-2], shape[-1]), dtype='uint16')
    tcount = 0
    rcount = 0
    for fi in files:
        if 'Trans' in fi:
            whole[0,tcount,:,:,:] = TiffReader(path + fi).data
            tcount = tcount + 1
        if 'Reflected' in fi:
            whole[1,rcount,:,:,:] = TiffReader(path + fi).data
            rcount = rcount + 1
            
    OmeTiffWriter.save(whole, savedir + l + '.ome.tif', dim_order=('CTZYX'))
    

########## Open a single channel of a giant movie ##############
path = 'F:/HL-60_Galv/20230329/'
signal = '20230329_488GFP-CAAX_640SiR-DNA_10minrandom_20mingalv4_w2'
#get all images of the channel of interest (signal)
fl = [x for x in os.listdir(path) if signal in x]
#sort properly by time point
fl.sort(key = lambda x: int(x.split('_t')[1].split('.ome')[0]))
shape = TiffReader(path+fl[0]).shape
wholeim = np.zeros((int(len(fl)),shape[-3],shape[-2], shape[-1]), dtype='uint16')
for i, f in enumerate(fl):
    wholeim[i,:,:,:] = TiffReader(path + f).data

OmeTiffWriter.save(wholeim, path + fl[0].split('_t')[0] + '.ome.tif', dim_order=('TZYX'))




############# fuse two sequential, but separate videos ##############
im1 = '//10.158.28.37/ExpansionHomesA/avlnas/caged fmlp/20230509/20230509_488EGFP-CAAX_640SPY650-DNA_Random1_w1GFP-Cy5-UVEpi-Trans_t2.ome.tf2'
im2 = 'D:/Aaron/Data/Chemotaxis/20230509/20230509_488EGFP-CAAX_640SPY650-DNA_ChemDirected0.ome.tif'

image1 = BioformatsReader(im1)
image2 = OmeTiffReader(im2)

#not sure if np.stack is the correct function
fuse = np.stack(image1.data, image2.data)





