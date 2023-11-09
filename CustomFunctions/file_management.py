# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:07:55 2023

@author: Aaron
"""

import os
import shutil


############## put image files into their own folders ################

startdir = 'D:/Aaron/Data/Chemotaxis/Raw_Data/'
#get files only, not folders
files = [x for x in os.listdir(startdir) if '.' in x]
#get unique file names
uni = []
[uni.append(x.split('_w')[0]) for x in files if x.split('_w')[0] not in uni and 'companion' not in x]

for u in uni:
    upath = startdir + u + '/'
    #first make the directory if it doesn't already exist
    if not os.path.exists(upath):
        os.mkdir(upath)
    #next get all the relevant files
    rvfi = [x for x in files if u in x]
    #move the files
    for r in rvfi:
        shutil.move(startdir+r, upath+r)
        # print(startdir+r, upath+r)


################## change file names in a directory #####################
dirr = 'D:/Aaron/Data/Chemotaxis/Raw_Data/20230510_488EGFP-CAAX_640JF646actin-halotag_ChemDirected3/'
for o in os.listdir(dirr):
    os.rename(dirr+o,dirr+o.replace('ChemDirected3','Random7'))

