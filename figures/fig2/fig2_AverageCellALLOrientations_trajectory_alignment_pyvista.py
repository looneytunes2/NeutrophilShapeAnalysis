

import numpy as np
import pandas as pd
import pyvista as pv
import vtk
import matplotlib.pyplot as plt
from matplotlib import cm
from neutrophil_shape.CustomFunctions import shtools_mod
from neutrophil_shape.config.loader import load_config
from neutrophil_shape.CustomFunctions.shapePCAtools import load_shcoeffs


cellname = '20231116_488EGFP-CAAX_3mA_37C_1_cell_79_frame_144'

## get config constants
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'
lmax = config.common.l_order

#get some directories
basedir = config.common.savedir / 'shape_data'

#the SH coefficients are saved separately from the shape metrics, combine them on 'cell' (up to the config's l_order)
shdf = load_shcoeffs(basedir, lmax)
#get average of the whole dataset
mean_sh_arr = shdf.drop(columns = 'cell').mean().to_numpy()
#get average reconstruction
mesh, grid = shtools_mod.get_reconstruction_from_coeffs(mean_sh_arr.reshape(2,lmax+1,lmax+1))
mesh = pv.PolyData(mesh)


#each view is rendered on its own with a parallel (orthographic) camera at the same
#scale, so the panels can be cropped to the mesh and laid out in real (um) units
VIEWS = ['xy','xz','yz']

# ------------------------------------------------------------------------ #

PX_PER_UM = 100  # render scale, the same for every view so the panels line up
WINDOW_SIZE = [3600, 3600]  # large enough to hold the whole mesh at PX_PER_UM
AXES_WINDOW_SIZE = [1200, 1200]
IN_PER_UM = 0.2  # size of the final figure; its dpi is set to keep PX_PER_UM
PAD = 0.7  # um of space around the mesh in each panel
GAP = 1.0  # um between panels

cameradist = 57
CAMERA_POSITIONS = {
'xy':[
     (0, 0, cameradist),  # position
     (0, 0, 0),  # focal point
     (0.0, 1.0, 0.0),  # view up
     ],
'xz':[
     (0, -cameradist, 0),  # position
     (0, 0, 0),  # focal point
     (0.0, 0.0, 1.0),  # view up
     ],
'yz':[
     (-cameradist, 0, 0),  # position
     (0, 0, 0),  # focal point
     (0.0, 0.0, 1.0),  # view up
     ]
}
#which way the screen's right (u) and up (v) directions point in each view, as (axis, sign)
SCREEN_AXES = {
    'xy': (('x', 1), ('y', 1)),
    'xz': (('x', 1), ('z', 1)),
    'yz': (('y', -1), ('z', 1)),
}

def render_frame(
        plotter,
        actors,
        camera_position,
        ):
    """Render actors off-screen with a parallel camera at PX_PER_UM and return an RGBA image array
    whose center is the world origin."""
    plotter.clear()
    plotter.enable_lightkit()
    for actor in actors:
        if isinstance(actor, pv.DataSet):
            plotter.add_mesh(actor, color="corn_silk", show_edges=False, smooth_shading=True)
        else:
            plotter.add_actor(actor)

    plotter.camera_position = camera_position
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = plotter.window_size[1] / (2 * PX_PER_UM)
    plotter.reset_camera_clipping_range()
    plotter.render()

    img = plotter.screenshot(return_img=True, transparent_background=True)
    assert img.shape[:2] == (plotter.window_size[1], plotter.window_size[0])
    return img

yellow = np.array([255, 224, 102])/255
red = np.array([222, 33, 71])/255
blue = np.array([54, 111, 209])/255
total_length = 4
def make_custom_axes_actor(
    total_length=total_length,
    shaft_radius=0.05,
    tip_radius=0.22,
    tip_length_ratio=0.4,     # fraction of total_length used by the tip
    shaft_type="cylinder",      # "cylinder" or "line"
    tip_type="cone",            # "cone" or "sphere"
    colors = [red, yellow, blue],
    labels=("X", "Y", "Z"),
    label_size=0.4,
    show_labels=False,
):
    """
    Build a vtkAxesActor with independent color/shaft/tip styling
    per axis. Returns the actor (add via plotter.add_actor(actor)).
    """
    axes = vtk.vtkAxesActor()

    # Uniform arm length for all three axes
    axes.SetTotalLength(total_length, total_length, total_length)

    # Shaft (line vs cylinder) and tip (cone vs sphere) geometry
    axes.SetShaftType(
        vtk.vtkAxesActor.CYLINDER_SHAFT
        if shaft_type == "cylinder"
        else vtk.vtkAxesActor.LINE_SHAFT
    )
    axes.SetTipType(
        vtk.vtkAxesActor.CONE_TIP
        if tip_type == "cone"
        else vtk.vtkAxesActor.SPHERE_TIP
    )

    # Thickness / proportions
    axes.SetCylinderRadius(shaft_radius)
    axes.SetConeRadius(tip_radius)
    axes.SetSphereRadius(tip_radius)
    axes.SetNormalizedTipLength(tip_length_ratio, tip_length_ratio, tip_length_ratio)
    axes.SetNormalizedShaftLength(
        1 - tip_length_ratio, 1 - tip_length_ratio, 1 - tip_length_ratio
    )

    # Per-axis colors (shaft + tip separately, so both match)
    axes.GetXAxisShaftProperty().SetColor(*colors[0])
    axes.GetXAxisTipProperty().SetColor(*colors[0])
    axes.GetYAxisShaftProperty().SetColor(*colors[1])
    axes.GetYAxisTipProperty().SetColor(*colors[1])
    axes.GetZAxisShaftProperty().SetColor(*colors[2])
    axes.GetZAxisTipProperty().SetColor(*colors[2])

    # Labels
    axes.SetXAxisLabelText(labels[0])
    axes.SetYAxisLabelText(labels[1])
    axes.SetZAxisLabelText(labels[2])
    axes.SetAxisLabels(1 if show_labels else 0)

    for c, cap in enumerate((
        axes.GetXAxisCaptionActor2D(),
        axes.GetYAxisCaptionActor2D(),
        axes.GetZAxisCaptionActor2D(),
    )):
        cap.GetCaptionTextProperty().SetColor(*colors[c])
        cap.GetCaptionTextProperty().SetBold(True)
        cap.GetCaptionTextProperty().SetItalic(False)
        cap.GetCaptionTextProperty().ShadowOff()
        cap.SetWidth(label_size * 0.1)
        cap.SetHeight(label_size * 0.1)

    return axes


def place_axes_actor(axes_actor, origin=(0, 0, 0), x_dir=(1, 0, 0), y_dir=(0, 1, 0), z_dir=(0,0,1)):
    """
    Position/rotate a vtkAxesActor so its local X/Y/Z point along the
    given world-space direction vectors (right-handed basis assumed;
    z_dir is computed automatically if not given).
    """


    # Build 4x4 transform: columns are the rotated basis vectors
    m = np.eye(4)
    m[:3, 0] = x_dir
    m[:3, 1] = y_dir
    m[:3, 2] = z_dir
    m[:3, 3] = origin

    vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, m[i, j])

    transform = vtk.vtkTransform()
    transform.SetMatrix(vtk_matrix)
    axes_actor.SetUserTransform(transform)



# ---- Off-screen PyVista plotters ---- #
try:
    pv.start_xvfb()  # needed on headless Linux without a display;
except Exception:
    pass

plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
axes_plotter = pv.Plotter(off_screen=True, window_size=AXES_WINDOW_SIZE)

bounds = dict(zip(['xmin','xmax','ymin','ymax','zmin','zmax'], mesh.bounds))
def screen_range(view):
    """(u0, u1), (v0, v1): the um extent of the mesh along the screen's right and up directions."""
    ranges = []
    for axis, sign in SCREEN_AXES[view]:
        lo, hi = bounds[axis + 'min'] * sign, bounds[axis + 'max'] * sign
        ranges.append((min(lo, hi), max(lo, hi)))
    return ranges

def crop(img, urange, vrange, pad = 0):
    """Crop an image centered on the world origin to a um range, returns the crop and its size in um."""
    h, w = img.shape[:2]
    c0, c1 = [int(round(w / 2 + u * PX_PER_UM)) for u in (urange[0] - pad, urange[1] + pad)]
    r0, r1 = [int(round(h / 2 - v * PX_PER_UM)) for v in (vrange[1] + pad, vrange[0] - pad)]
    assert c0 >= 0 and r0 >= 0 and c1 <= w and r1 <= h, 'window too small for the crop'
    return img[r0:r1, c0:c1], ((c1 - c0) / PX_PER_UM, (r1 - r0) / PX_PER_UM)

### the mesh panels, cropped to the mesh bounds
panels, sizes = {}, {}
for view in VIEWS:
    img = render_frame(plotter, [mesh], CAMERA_POSITIONS[view])
    urange, vrange = screen_range(view)
    panels[view], sizes[view] = crop(img, urange, vrange, PAD)

### the axes actors, cropped to the arrows (the y arrow points left in the yz view)
ax_len = total_length + 0.5
axes_crops = {}
for view in VIEWS:
    axes_actor = make_custom_axes_actor()
    img = render_frame(axes_plotter, [axes_actor], CAMERA_POSITIONS[view])
    uaxis = (-ax_len, 0.5) if SCREEN_AXES[view][0][1] < 0 else (-0.5, ax_len)
    axes_crops[view], _ = crop(img, uaxis, (-0.5, ax_len))

#close plotters
plotter.close()
axes_plotter.close()


# ------------------------- Matplotlib figure -------------------------- #
# layout in um, filling the upper left triangle of a 2x2 grid: xy in the upper left, yz to its right (the
# same height), xz below xy (the same width), with the axes in the margins to the left of xz and above xy and yz
AX = total_length + 1.2  # size of an axes margin
Wx, Hz = sizes['xz']
_, Hxy = sizes['xy']
Wyz, _ = sizes['yz']
Hrow = max(Hxy, sizes['yz'][1])
fig_w = AX + Wx + GAP + Wyz
fig_h = AX + Hrow + GAP + Hz

fig = plt.figure(figsize=(fig_w * IN_PER_UM, fig_h * IN_PER_UM), dpi=PX_PER_UM / IN_PER_UM)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis('off')

def place(img, left, bottom, size):
    ax.imshow(img, extent=(left, left + size[0], bottom, bottom + size[1]), aspect='auto', interpolation='none')

xz_bottom = 0
xy_bottom = Hz + GAP + (Hrow - Hxy) / 2
yz_bottom = Hz + GAP + (Hrow - sizes['yz'][1]) / 2
yz_left = AX + Wx + GAP
place(panels['xz'], AX, xz_bottom, sizes['xz'])
place(panels['xy'], AX, xy_bottom, sizes['xy'])
place(panels['yz'], yz_left, yz_bottom, sizes['yz'])

#axes: left of the xz panel, centered vertically on its cell; above the xy and yz panels, centered
#horizontally on their cells (the axes crops are centered on the arrows)
box = (ax_len + 0.5, ax_len + 0.5)
place(axes_crops['xz'], AX - box[0] - 0.2, xz_bottom + (sizes['xz'][1] - box[1]) / 2, box)
axes_bottom = fig_h - AX + 0.2
place(axes_crops['xy'], AX + (Wx - box[0]) / 2, axes_bottom, box)
place(axes_crops['yz'], yz_left + (Wyz - box[0]) / 2, axes_bottom, box)

fig.savefig(__file__.split('.')[0]+'.png', dpi=PX_PER_UM / IN_PER_UM)
plt.close(fig)
