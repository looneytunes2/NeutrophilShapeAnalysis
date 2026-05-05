
################### get the SH coeffs from the example cell and save the recon



import pandas as pd
from neutrophil_shape.CustomFunctions import shtools_mod
from pathlib import Path
from neutrophil_shape.config.loader import load_config

cellname = '20231116_488EGFP-CAAX_3mA_37C_1_cell_79_frame_144'

## get config constants
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory_shape'
lmax = config.common.l_order

#get some directories
basedir = config.common.savedir / 'shape_data'
shdf = pd.read_csv(basedir / 'All_Data_with_CGPS_bins.csv', index_col=0)

#get cell in question from dataframe
cell = shdf[shdf.cell==cellname]
#get just columns with SH coeffs
cellsh = cell[[x for x in cell.columns.to_list() if 'shcoeff' in x]]

mesh, grid = shtools_mod.get_reconstruction_from_coeffs(cellsh.values.reshape(2,lmax+1,lmax+1))

shtools_mod.save_polydata(mesh, Path.cwd().joinpath('pipeline_SH_recon_width.vtp'))

