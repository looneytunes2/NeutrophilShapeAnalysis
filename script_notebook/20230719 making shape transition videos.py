# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:55:43 2023

@author: Aaron
"""

######### do contour integrals for all migration modes ################

from CustomFunctions.DetailedBalance import contour_integral
from scipy import interpolate
from scipy.spatial import distance
import multiprocessing
import re
import pandas as pd
import pickle as pk
import numpy as np



fl = 'D:/Aaron/Data/Chemotaxis/'
folder_fl = fl + 'Data_and_Figs/'
savedir = folder_fl



Shape_Metrics = pd.read_csv(folder_fl + 'Shape_Metrics_with_Digitized_PCs.csv')



avgpcs = Shape_Metrics[[x for x in Shape_Metrics.columns.to_list() if 'PC' in x and 'bin' not in x]].mean().to_numpy()


#get changes in PCs between consecutive frames of a movie
znbins = 11
znbins = 11

#first bin PCs
hist1, PC1bins = np.histogram(Shape_Metrics.Cell_PC1, znbins)
hist2, PC2bins = np.histogram(Shape_Metrics.Cell_PC2, znbins)

pca = pk.load(open(folder_fl+"pca.pkl",'rb'))

whichpcs = [1,2]


fourcorners = np.array([[0,0],
[0,1],
[1,1],
[1,0]])


interpfreq = 0.1




def mesh_from_bins(binpos,
                   avgpcs,
                   pc1int,
                   pc2int,
                   pca,
                   savedir,
                   lmax,
                   ):
    temppcs = avgpcs.copy()
    #transform PCs back from bins into PCs and put them in the proper location
    #in the average PC array
    temppcs[whichpcs[0]-1] = pc1int(binpos[1])
    temppcs[whichpcs[1]-1] = pc2int(binpos[2])
    #inverse pca transform
    coeffs = pca.inverse_transform(avgpcs)
    #get mesh from coeffs
    mesh, _ = shtools.get_reconstruction_from_coeffs(coeffs.reshape(2,lmax+1,lmax+1))



    #save mesh
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(savedir+'loop_'+underscore+'_'+binpos[0]+'.vtp')
    writer.SetInputData(mesh)
    writer.Write()
    
    return



