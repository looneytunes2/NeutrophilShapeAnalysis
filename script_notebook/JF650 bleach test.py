# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:52:21 2024

@author: Aaron
"""

import os
import numpy as np
import pandas as pd
import skimage.measure
import aicsimageio
import seaborn as sns
import matplotlib.pyplot as plt

from aicssegmentation.core.utils import hole_filling
from aicsimageio.readers.czi_reader import CziReader
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter

from CustomFunctions.track_functions import MO

dirr = '//10.158.28.37/ExpansionHomesA/avlnas/LLS/2024-06-11/'
lit = [x for x in os.listdir(dirr) if 'Lattice' in x]

# bigim = CziReader(dirr + lit[0])

alldata = []
for l in lit:
    bigim = CziReader(dirr + l)
    avg = []
    maxx = []
    med = []
    minn = []
    for i in range(bigim.shape[0]):
        slic = bigim.data[i,:,:,:,:]
    
    
        thresh_img = MO(slic[1,:,:,:], local_adjust=0.999, global_thresh_method='tri', object_minArea=6000)
        # fill in the holes
        hole_max = 25000
        hole_min = 1
        thresh_img = hole_filling(thresh_img, hole_min, hole_max)
        
        filtered_elements = slic[0,:,:,:][thresh_img]

        avg.append(np.mean(filtered_elements))
        maxx.append(np.max(filtered_elements))
        med.append(np.median(filtered_elements))
        minn.append(np.min(filtered_elements))
        
        
    alldata.append([l, avg, med, maxx, minn])
    # seg = thresh_img.astype(np.uint8)
    # seg[seg > 0] = 255

    # #get objects
    # im_labeled, n_labels = skimage.measure.label(seg, background=0, return_num=True)
    # im_props = skimage.measure.regionprops(im_labeled)
    
    

    
    # OmeTiffWriter.save(seg, 'C:/Users/Aaron/Desktop/thresh.ome.tiff')
    
    # nzslic = slic.copy()
    # nzslic[nzslic==0] = np.random.randint(100,105,len(nzslic[nzslic ==0]))
    
    # OmeTiffWriter.save(nzslic[1,:,:,:], 'C:/Users/Aaron/Desktop/orig.ome.tiff')

dflist = []
for x in alldata:
    name = [x[0]]*len(x[1])
    fluor = [x[0].split('_')[3]]*len(x[1])
    time = np.array(list(range(len(x[1]))))*5
    temp = pd.DataFrame([name, fluor, time, x[1], x[2], x[3], x[4]]).T
    temp.columns = ['image','fluor','time','avg','median','max','min']
    dflist.append(temp)

df = pd.concat(dflist).reset_index(drop= True)

df.to_csv('C:/Users/Aaron/Desktop/jfbleachtest.csv')

#add normalized data
normlist = []
for i, c in df.groupby('image'):
    for f in ['avg','median','max','min']:
        c[f+'_norm'] = c[f]/c[f].iloc[0]
    normlist.append(c)
df = pd.concat(normlist).reset_index(drop= True)


fig, axes = plt.subplots(4,1)
for i, c in enumerate([x for x in df.columns.to_list() if 'norm' in x]):
    ax = axes[i]
    sns.lineplot(data = df, x = 'time', y = c, hue = 'image', estimator = None, ax = ax)
    ax.set_ylabel('Normalized '+c.split('_norm')[0]+'\nintensity (a.u.)')
    ax.get_legend().remove()
fig.legend(labels = ['646 cell 1','646 cell 2','650 cell 1','650 cell 2'], loc = 1)
ax.set_xlabel('Time (sec)')