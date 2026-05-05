
################### get the SH coeffs from the example cell and save the recon


import pandas as pd
from neutrophil_shape.CustomFunctions import shtools_mod
from neutrophil_shape.config.loader import load_config
from pathlib import Path

config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'
lmax = config.common.l_order

#get some directories
basedir = config.common.savedir / 'shape_data'
shdf = pd.read_csv(basedir / 'All_Data_with_CGPS_bins.csv', index_col=0)
#get just columns with SH coeffs
allsh = shdf[[x for x in shdf.columns.to_list() if 'shcoeff' in x]]


mesh, _ = shtools_mod.get_even_reconstruction_from_coeffs(allsh.mean().values.reshape(2,lmax+1,lmax+1))

shtools_mod.save_polydata(mesh, Path(__file__).parent / 'average_cell_mesh.vtp')
