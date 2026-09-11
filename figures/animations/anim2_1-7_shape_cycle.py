# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:15:45 2023

@author: Aaron
"""

import vtk
from vtk.util import numpy_support
import pyvista as pv 
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from neutrophil_shape.CustomFunctions import linear_cycle_utils, utils, shtools_mod
from neutrophil_shape.config.loader import load_config


#get directories and open separated datasets
align = 'trajectory_shape'
whichpcs = (1,7)
binnum = 18
binrange = 360/binnum
direction = 'clockwise'
zerostart = 'left'


# Output
FPS = 7
# Rendering
MESH_COLOR = "corn_silk"
BACKGROUND = "lightsteelblue"
WINDOW_SIZE = [800, 800]
CAMERA_POSITION = None   # None -> auto-fit each frame; set explicitly (e.g.
                         # "iso", "xy", or a pyvista camera position tuple)
                         # to lock the camera across frames.
CAMERA_POSITION = np.array([
    (0, 0, 95),  # position
    (0, 0, 0),  # focal point
    (0.0, 1.0, 0.0),  # view up
])

AXES_POSITION = (-9,-9,0)


# ------------------------------------------------------------------------ #

def rotate_mesh(mesh, angle1, angle2, angle3):
    """Apply two sequential Euler rotations (degrees), about the mesh's own center."""
    rot = mesh.copy()
    rot = rot.rotate_x(angle1, point=rot.center_of_mass(), inplace=False)
    rot = rot.rotate_y(angle2, point=rot.center_of_mass(), inplace=False)
    rot = rot.rotate_z(angle3, point=rot.center_of_mass(), inplace=False)
    return rot


def rotate_axes_actor(axes, angle1, angle2, angle3):
    """Apply two sequential Euler rotations (degrees) to a pv.Axes actor, about its own origin."""
    actor = axes.axes_actor  # underlying vtkAxesActor
    origin = axes.origin
    actor.SetOrigin(*origin)
    actor.RotateX(angle1)
    actor.RotateY(angle2)
    actor.RotateZ(angle3)
    return axes

def make_axes_actor(angle1, angle2, angle3, position, length=10, label_size = 12):
    """Build a vtkAxesActor rotated and positioned explicitly, with no PyVista wrapper interference."""
    actor = vtk.vtkAxesActor()
    actor.SetTotalLength(length, length, length)

    # set font size for each axis label (X, Y, Z)
    for get_caption in (actor.GetXAxisCaptionActor2D,
                        actor.GetYAxisCaptionActor2D,
                        actor.GetZAxisCaptionActor2D):
        caption = get_caption()
        caption.GetTextActor().SetTextScaleModeToNone()
        caption.GetCaptionTextProperty().SetFontSize(label_size)
        # caption.GetCaptionTextProperty().SetBold(False)
        # caption.GetCaptionTextProperty().SetItalic(False)
        # caption.GetCaptionTextProperty().ShadowOff()

    transform = vtk.vtkTransform()
    transform.Translate(*position)   # applied last (outermost)
    transform.RotateZ(angle3)
    transform.RotateY(angle2)
    transform.RotateX(angle1)        # applied first (innermost) — matches rotate_mesh order

    actor.SetUserMatrix(transform.GetMatrix())
    return actor

def render_frame(plotter, mesh, CAMERA_POSITION = None, axes_actor = None):
    """Render the current mesh state off-screen and return an RGB image array."""
    plotter.clear()
    plotter.enable_lightkit()
    plotter.add_mesh(mesh, color=MESH_COLOR,
                     show_edges=False,
                     smooth_shading=True,      # interpolate normals across faces -> visible curvature
                    # lighting=True,
                    # specular=0.5,             # adds highlight/shine, helps depth perception
                    # specular_power=15,
                    # diffuse=0.8,
                    # ambient=0.2,
                    )
    
    if axes_actor is not None:
        plotter.add_actor(axes_actor)
    
    if CAMERA_POSITION is not None:
        plotter.camera_position = CAMERA_POSITION
    else:
        plotter.reset_camera()
    plotter.set_background(BACKGROUND)
    return plotter.screenshot(return_img=True)




config = load_config(microscope_type='confocal')
config._alignment = align
all_origins = config.db_params.all_origins
origin_index = config.common.pc_combos.index(whichpcs)
origin = all_origins[align][origin_index]
npcs = config.common.npcs
nbins = config.common.npcs
lmax = config.common.l_order
time_interval = config.im_params.time_interval
savedir = config.common.savedir
datadir = savedir / 'shape_data'
bin_col = f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins'



FullFrame = pd.read_csv(datadir / 'All_Data_with_CGPS_bins.csv', index_col = 0)
FullFrame['real_time'] = FullFrame.time.copy()
aers = pd.read_csv(savedir / 'detailed_balance' / f'{utils.whichpc_string(whichpcs)}_raw_transition_aer_cf.csv', index_col = 0)
#merge aer and cf info
join_keys = ['CellID','real_time']
# Keep only non-redundant columns from df2 (plus the key)
unique_aer_col = [c for c in aers.columns if c not in FullFrame.columns or c in join_keys]
TotalFrame = FullFrame.merge(aers[unique_aer_col], on=join_keys)
#open the centers of the binned PCs
centers = pd.read_csv(datadir / 'PC_bin_centers.csv', index_col=0)





angframe = linear_cycle_utils.linearize_cycle_continuous(
            TotalFrame, 
            centers,
            origin, 
            whichpcs,
            zerostart,
            direction,)

angframe =  linear_cycle_utils.bin_angular_coord(
        angframe,
        whichpcs,
        binrange,
        )


#average degrees per second in each bin
avgcf = abs(angframe.angular_velocity.mean())
#average number of seconds spent in each bin
avgtime = binrange / avgcf
#get the average speeds based on the angular bins
displacements = (angframe.groupby(bin_col).speed.mean()*avgtime).reset_index()
displacements = pd.concat((displacements, displacements.iloc[[0]]), ignore_index = True)
displacements['cumulative_position'] = displacements.speed.cumsum()

### adjust camera position to the average of the displacements
CAMERA_POSITION[0,0] = displacements.cumulative_position.mean()
CAMERA_POSITION[1,0] = displacements.cumulative_position.mean()






#get times for all the frames included in the movie
#(which may have been dropped from dataframes in analysis)

 # ---- Off-screen PyVista plotter (reused across frames for speed) ---- #
try:
    pv.start_xvfb()  # needed on headless Linux without a display;
                      # harmless to skip on systems that don't need it
except Exception:
    pass

plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
  
#use 1D gaussian smoothening to get average PC curves over the 1D cycle
allinterpvals = []
sh_cols = [x for x in angframe.columns if 'shco' in x]
print("Pre-rendering mesh frames...")
mesh_images = []
cum_pos = np.zeros(3)
for a, row in displacements.iterrows():
    
    shcoeffs = angframe.loc[angframe[bin_col]==row[bin_col],sh_cols].mean().values
    mesh, _ = shtools_mod.get_even_reconstruction_from_coeffs(np.reshape(shcoeffs, (2,lmax+1,lmax+1)))
    ## move mesh by displacement based on average speed
    coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    coords += np.array([row['cumulative_position'],0,0])
    # Translate
    mesh = shtools_mod.update_mesh_points(mesh, coords[:, 0], coords[:, 1], coords[:, 2])
    pvmesh = pv.wrap(mesh)

    # axes = pv.Axes()
    # axes = rotate_axes_actor(axes, row.Euler_angles_X, row.Euler_angles_Z,  row.Width_Rotation_Angle)
    # axes.axes_actor.SetPosition(AXES_POSITION)
    
    axes_actor = make_axes_actor(
        0,
        0,
        0,
        AXES_POSITION,
        length = 2.5,
        label_size = 16,
        )

    mesh_images.append(render_frame(plotter, pvmesh, CAMERA_POSITION, axes_actor))
    
plotter.close()



           
# ------------------------- Matplotlib figure -------------------------- #
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

im = ax.imshow(mesh_images[0])
ax.axis("off")


fig.tight_layout()

def update(i):
    im.set_data(mesh_images[i])
    return im,

anim = animation.FuncAnimation(
    fig, update, frames=binnum, interval=1000 / FPS, blit=False
)



anim.save(Path(__file__.split('.py')) / '.mp4',
         fps=FPS, dpi = 300)#, extra_args=['-vcodec', 'libx264'])


plt.close(fig)



