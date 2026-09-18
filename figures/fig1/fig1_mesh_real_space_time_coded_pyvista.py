
################### make a mesh animation movie from the meshes saved during
################### data processing, rendered with pyvista instead of ParaView.
################### meshes are already saved in their real-space orientation,
################### so no rotation is needed -- just open each frame's mesh
################### and place it at the correct position along the trajectory


import dataclasses
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib import cm
from neutrophil_shape.config.loader import load_config


### open config and get directories
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'
savedir = config.common.savedir
datadir = savedir.joinpath('shape_data')

cellname = '20231116_488EGFP-CAAX_3mA_37C_2_cell_9'


#get all the position and trajectory info for this cell
df = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col=0)
cellinfo = df[df.CellID == cellname].copy().sort_values('frame').reset_index(drop=True)
#limit to frame window
cellinfo = cellinfo[cellinfo.frame.isin(list(range(66,92,3)))].reset_index(drop=True)

#mesh directory for this cell's experiment
exp = cellinfo.Experiment.values[0]
configdict = dataclasses.asdict(config)
meshdir = configdict['experiment'][exp]['localdir'].joinpath('meshes')

#get displacements and then cumulative position
tempc = cellinfo[['x_raw','y_raw','z_raw']].diff()
tempc.fillna(0, inplace = True)
cum_pos = np.cumsum(tempc.values, axis = 0)


#define the colors to make the meshes
cmap = cm.get_cmap('rainbow')
discrete_colors = cmap(np.linspace(0,1,len(cellinfo)))


# ---- off-screen pyvista plotter (meshes are added to it in the loop below) ---- #
try:
    pv.start_xvfb()  # needed on headless Linux without a display;
                      # harmless to skip on systems that don't need it
except Exception:
    pass

plotter = pv.Plotter(off_screen=True, window_size=[5000, 5000])
plotter.enable_lightkit()


for i, row in cellinfo.iterrows():

    meshfl = meshdir.joinpath(row.cell+'_cell_mesh.vtp')
    if not meshfl.exists():
        continue

    #meshes are already in their real-space orientation, so just move
    #them to the correct position along the trajectory
    mesh = pv.read(meshfl)
    mesh = mesh.translate(cum_pos[i], inplace=False)

    #change the visualization
    plotter.add_mesh(
        mesh,
        color=discrete_colors[i, :3],
        opacity=0.4,
        smooth_shading=True,
        )


############# SCALE BAR
slen = 10
sx = 15
sy = 7.5
# 10um line scalebar
scalebar = pv.Line((sx, sy, 0), (sx+slen, sy, 0)).tube(radius=0.25, n_sides=20)
plotter.add_mesh(scalebar, color=[0, 0, 0])


##### make a little axes at a specific position
yellow = np.array([255, 224, 102])/255
red = np.array([222, 33, 71])/255
blue = np.array([54, 111, 209])/255

arrow_pos = np.array([18, 0, 0])
xyzprops = {
    'direction': [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    'Color': [red, yellow, blue],
    }

for direction, color in zip(xyzprops['direction'], xyzprops['Color']):
    arrow = pv.Arrow(
        start=arrow_pos,
        direction=direction,
        scale=5.0,
        tip_radius=0.075,
        tip_resolution=100,
        shaft_resolution=100,
        )
    plotter.add_mesh(arrow, color=color)


#change background to white
plotter.set_background('white')


############ set up the camera
avgpos = np.mean(cum_pos, axis = 0)
plotter.camera_position = [
    tuple(avgpos + [0, 0, avgpos[0]*avgpos[1]*1.5]),  # camera position
    tuple(avgpos),  # focal point
    (0, -1, 0),  # view up
    ]


plotter.screenshot(__file__.split('.')[0]+'.png')
plotter.close()
