
################### get the SH coeffs from the example cell and save the recon


import pandas as pd
from CustomFunctions import shtools_mod
from pathlib import Path

#get some directories
basedir = Path('E:/Aaron/Combined_37C_Confocal_PCA_shape')
cellname = '20231116_488EGFP-CAAX_3mA_37C_1_cell_79_frame_145'
shdf = pd.read_csv(basedir.joinpath('Data_and_Figs/Shape_Metrics_Galvanotaxis_Confocal_40x_37C_10s.csv'), index_col = 0)
#get cell in question from dataframe
cell = shdf[shdf.cell==cellname]
#get just columns with SH coeffs
cellsh = cell[[x for x in cell.columns.to_list() if 'shcoeff' in x]]

lmax = 10
mesh, grid = shtools_mod.get_reconstruction_from_coeffs(cellsh.values.reshape(2,lmax+1,lmax+1))

shtools_mod.save_polydata(mesh, Path.cwd().joinpath('pipeline_SH_recon_shape.vtp').as_posix())

