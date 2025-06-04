
################### get the SH coeffs from the example cell and save the recon


import pandas as pd
from CustomFunctions import shtools_mod
import os

#get some directories
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/Data_and_Figs/'
shdf = pd.read_csv(basedir+'All_Data_with_CGPS_bins.csv')
#get just columns with SH coeffs
allsh = shdf[[x for x in shdf.columns.to_list() if 'shcoeff' in x]]

lmax = 10
mesh, grid = shtools_mod.get_reconstruction_from_coeffs(allsh.mean().values.reshape(2,lmax+1,lmax+1))

shtools_mod.save_polydata(mesh, os.path.dirname(__file__)+'average_cell_mesh.vtp')

