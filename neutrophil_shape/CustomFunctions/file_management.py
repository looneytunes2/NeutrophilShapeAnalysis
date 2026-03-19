# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:07:55 2023

@author: Aaron
"""
import pandas as pd
import os
import shutil


def multicsv(csv):
    return (pd.read_csv(csv, index_col = 0))


############## put image files into their own folders ################

startdir = 'D:/Aaron/Data/Chemotaxis/Raw_Data/'
def im_in_fold(startdir):#get files only, not folders
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
direct = 'D:/Aaron/Data/Chemotaxis/Raw_Data/20230510_488EGFP-CAAX_640JF646actin-halotag_ChemDirected3/'
def change_file_names(direct):
    for o in os.listdir(direct):
        os.rename(direct+o,direct+o.replace('ChemDirected3','Random7'))


################### check that all files are in two directories #################
sourcedir = '//10.158.28.37/ExpansionHomesA/avlnas/LLS/2024-05-24/'
targetdir = 'G:/Cropped_LLS/'
files = [x for x in os.listdir(sourcedir) if 'Subset' in x and 'deskew' not in x]
all([x in os.listdir(targetdir) for x in files])


##################### delete all files with certain name ##############
deldir = '//10.158.28.37/ExpansionHomesA/avlnas/LLS/2024-07-02/'
files = [x for x in os.listdir(deldir) if 'MIP' in x]
for f in files:
    os.remove(deldir + f)