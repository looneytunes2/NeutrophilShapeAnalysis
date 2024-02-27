# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:31:03 2023

@author: Aaron
"""

# importing required packages
from pathlib import Path
import shutil
import os







###################### copy files from one directory to another ###############################
# defining source and destination
# paths
src = 'C:/Users/Aaron/Data/Processed/Galvanotaxis/20230329/Cropped_Cells/'
trg = 'D:/Aaron/Data/Galvanotaxis/Processed/Cropped_Cells/Seg/'
 
files=[x for x in os.listdir(src) if 'segmented' in x]
 
# iterating over all the files in
# the source directory
for fname in files:
     
    # copying the files to the
    # destination directory
    shutil.copy2(os.path.join(src,fname), trg)
    
    
    
    
    
################# change the file names of files in a directory #########################
data = 'D:/Aaron/Data/Galvanotaxis/Processed/Cropped_Cells/Seg/'
for i, f in enumerate(os.listdir(data)):
    src = os.path.join(data, f)
    dst = os.path.join(data, (f.split('_segmented.tiff')[0]+'_struct_segmentation.tiff'))
    os.rename(src, dst)
    
