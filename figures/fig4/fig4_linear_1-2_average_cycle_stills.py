

import numpy as np
import pandas as pd
import pyvista as pv
import vtk
import matplotlib.pyplot as plt
from matplotlib import cm
from neutrophil_shape.CustomFunctions import linear_cycle_utils, shtools_mod
from neutrophil_shape.config.loader import load_config


#get directories and open separated datasets
treatments = ['Random']
align = 'trajectory'
whichpcs = (1,2)
binnum = 6
binrange = 360/binnum
direction = 'clockwise'
zerostart = 'left'

# Rendering
BACKGROUND = "white"
WINDOW_SIZE = [800, 800]
AXES_LENGTH = 2.5

#meshes are rendered individually (not spread into one scene), so 'xy' and
#'xz' both remain informative -- each gives a different rotation of every
#single mesh, same as the original single-mesh-at-a-time version of this script
VIEWS = ['xy', 'xz']

# ------------------------------------------------------------------------ #

def make_axes_actor(position, length=10, label_size=12):
    """Build a vtkAxesActor positioned explicitly, with no PyVista wrapper interference."""
    actor = vtk.vtkAxesActor()
    actor.SetTotalLength(length, length, length)

    for get_caption in (actor.GetXAxisCaptionActor2D,
                        actor.GetYAxisCaptionActor2D,
                        actor.GetZAxisCaptionActor2D):
        caption = get_caption()
        caption.GetTextActor().SetTextScaleModeToNone()
        caption.GetCaptionTextProperty().SetFontSize(label_size)

    transform = vtk.vtkTransform()
    transform.Translate(*position)
    actor.SetUserMatrix(transform.GetMatrix())
    return actor


def render_frame(plotter, mesh, color, camera_position, axes_actor=None):
    """Render a single mesh off-screen and return an RGB image array."""
    plotter.clear()
    plotter.enable_lightkit()
    plotter.add_mesh(mesh, color=color, show_edges=False, smooth_shading=True)
    if axes_actor is not None:
        plotter.add_actor(axes_actor)
    plotter.camera_position = camera_position
    plotter.set_background(BACKGROUND)
    #screenshot() only forces a render on the plotter's very first call ever;
    #on later calls it just grabs whatever's currently in the buffer, which can
    #still reflect a not-yet-applied camera_position from this same call when
    #clear()/add_mesh()/camera_position are cycled rapidly on one reused
    #plotter -- forcing a render here avoids capturing that stale frame.
    plotter.render()
    return plotter.screenshot(return_img=True)


config = load_config(microscope_type='confocal')
config._alignment = align
origins = config.db_params.origins
pc_combos = config.common.pc_combos
origin = origins[pc_combos.index(whichpcs)]
lmax = config.common.l_order
savedir = config.common.savedir
datadir = savedir / 'shape_data'
bin_col = f'PC{whichpcs[0]}_PC{whichpcs[1]}_Continuous_Angular_Bins'


FullFrame = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col=0)
TotalFrame = FullFrame[FullFrame.Treatment.isin(treatments)].reset_index(drop=True)
centers = pd.read_csv(datadir.joinpath('PC_bin_centers.csv'), index_col=0)


angframe = linear_cycle_utils.linearize_cycle_continuous(
            TotalFrame,
            centers,
            origin,
            whichpcs,
            zerostart,
            direction,)

angframe = linear_cycle_utils.bin_angular_coord(
        angframe,
        whichpcs,
        binrange,
        )

sh_cols = [x for x in angframe.columns if 'shco' in x]
bins_sorted = sorted(angframe[bin_col].unique())
#repeat the first bin's mesh again at the end so the cycle visibly closes
bins_sorted = bins_sorted + [bins_sorted[0]]

#one color per mesh (twilight is cyclic, so the repeated first/last bin gets
#two different-looking colors, showing the color progression across the
#whole cycle rather than an abrupt repeat)
cmap = cm.twilight
discrete_colors = cmap(np.linspace(0, 1, len(bins_sorted)))[:, :3]

#build each bin's mean-shape mesh
meshes = []
for b in bins_sorted:
    shcoeffs = angframe.loc[angframe[bin_col] == b, sh_cols].mean().values
    mesh, _ = shtools_mod.get_even_reconstruction_from_coeffs(np.reshape(shcoeffs, (2, lmax+1, lmax+1)))
    meshes.append(pv.wrap(mesh))

#a fixed axes triad, offset from the mesh so it doesn't overlap it
max_extent = max(max(m.bounds[1]-m.bounds[0], m.bounds[3]-m.bounds[2]) for m in meshes)
axes_actor = make_axes_actor(
    (-max_extent*0.75, -max_extent*0.75, 0),
    length=AXES_LENGTH,
    label_size=16,
    )


# ---- Off-screen PyVista plotter (reused across meshes/views) ---- #
try:
    pv.start_xvfb()  # needed on headless Linux without a display;
                      
except Exception:
    pass

plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)

# define camera position based on view
camera_positions = {}
for view in VIEWS:
    plotter.clear()
    plotter.add_mesh(max(meshes, key=lambda m: m.length))
    plotter.camera_position = view
    plotter.reset_camera()
    camera_positions[view] = plotter.camera_position

print("Rendering mesh frames...")
images = {view: [] for view in VIEWS}
for mesh, color in zip(meshes, discrete_colors):
    for view in VIEWS:
        img = render_frame(plotter, mesh, color, camera_positions[view], axes_actor)
        images[view].append(img)

plotter.close()


# ------------------------- Matplotlib figure -------------------------- #
fig, axes = plt.subplots(
    len(VIEWS), len(meshes),
    figsize=(len(meshes)*2, len(VIEWS)*2),
    gridspec_kw={'wspace': 0, 'hspace': 0},
    )

for v, view in enumerate(VIEWS):
    for m in range(len(meshes)):
        ax = axes[v, m]
        ax.imshow(images[view][m], aspect='auto')
        ax.axis('off')

plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi=500)
plt.close(fig)
