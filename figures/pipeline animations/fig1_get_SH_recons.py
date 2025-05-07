
################### make a mesh animation movie from the meshes saved during
################### data processing



import os 
from CustomFunctions.PCvisualization import shcoeff_recon_mesh_timelapse_realspace


#get some directories
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
cellname = '20231116_488EGFP-CAAX_3mA_37C_1_cell_42'

# shco

######## get SH reconstructions of a cell from all time points 
######## specifically for LLS data
shcoeff_recon_mesh_timelapse_realspace(
    basedir,
    cellname,
    os.getcwd(),
    lmax = 10,
    )