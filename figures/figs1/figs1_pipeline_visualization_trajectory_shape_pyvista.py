import dataclasses
import pandas as pd
import numpy as np
import pyvista as pv
import vtk
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from neutrophil_shape.config.loader import load_config
from neutrophil_shape.CustomFunctions import PCvisualization
from neutrophil_shape.CustomFunctions.shtools_mod import get_reconstruction_from_coeffs
# ----------------------------- CONFIG ---------------------------------- #


WINDOW_SIZE = [5000, 5000]

cameradist = 57
CAMERA_POSITION = [
     (0, 0, cameradist),  # position
     (0, 0, 0),  # focal point
     (0.0, 1.0, 0.0),  # view up
     ]
    
def render_frame(
        plotter,
        mesh_and_arrow,
        axes_actor = None,
        camera_position = None,
        mesh_color = "corn_silk",
        background_color = "lightsteelblue",
        ):
    """Render the current mesh state off-screen and return an RGB image array."""
    plotter.clear()
    plotter.enable_lightkit()
    plotter.add_mesh(
        mesh_and_arrow[0],
        color=mesh_color,
        show_edges=False,
        smooth_shading = True
        )
    
    plotter.add_mesh(
        mesh_and_arrow[1],
        color=[77/255, 130/255, 56/255],
        specular=1.0,       # shininess
        specular_power=100,  # sharpness of highlight
        smooth_shading=True,
        ambient=0.1,
        diffuse=0.6,
    )
    
    if axes_actor is not None:
        plotter.add_actor(axes_actor)
    
    if camera_position is not None:
        plotter.camera_position = camera_position
    else:
        plotter.reset_camera()

    plotter.set_background(background_color)

    plotter.render()

    return plotter.screenshot(return_img=True)

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

def generate_traj_arrow(start, direction):
    return pv.Arrow(
        start=start,
        direction=direction,
        tip_length=0.3,
        tip_radius=0.2,
        shaft_radius=0.1,
        scale=5.0
    )



### open config and get directories
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory_shape'


lmax = config.common.l_order
npcs = config.common.npcs
savedir = config.common.savedir
datadir = savedir.joinpath('shape_data')
### open the whole dataset
df = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col = 0)
## define cell and get its data
cellname = '20231116_488EGFP-CAAX_3mA_37C_1_cell_79_frame_144'
cellinfo = df[df.cell == cellname].copy()
## get experiment info to find mesh
exp = cellinfo.Experiment.values[0]
configdict = dataclasses.asdict(config)
localdir = configdict['experiment'][exp]['localdir']




### read mesh
meshpath = localdir.joinpath('meshes', cellname + '_cell_mesh.vtp')
mesh = pv.read(meshpath)
### get trajectory
traj_cols = ['Trajectory_Vec_X','Trajectory_Vec_Y','Trajectory_Vec_Z']
traj_vec = cellinfo[traj_cols].values[0]
### get the partial rotation angles
rotation, _ = R.align_vectors([1,0,0], traj_vec)
partial_euler_angles = rotation.as_euler('xyz', degrees = True)
### get the full rotation angles
euler_cols = ['Euler_Angles_X', 'Euler_Angles_Y', 'Euler_Angles_Z']
full_euler_angles = cellinfo[euler_cols].values[0]


all_euler_angles = [
    np.zeros(3),
    partial_euler_angles,
    full_euler_angles,
    ]

all_traj_arrow_pos = [
    np.array([-6.5,4,0]),
    np.array([-3.5,-6,0]),
    np.array([-3.5,-8.5,0]),
    ]

ax_origin = (-9,-10,0)


# ---- Off-screen PyVista plotter (reused across frames for speed) ---- #
try:
    pv.start_xvfb()  # needed on headless Linux without a display;
                      # harmless to skip on systems that don't need it
except Exception:
    pass

plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
rendered_images = []

for i in range(3):
    #first get the angles
    current_euler_angles = all_euler_angles[i]
    #rotate the mesh
    rotated_mesh = PCvisualization.rotate_mesh(
        mesh,
        current_euler_angles[0],
        current_euler_angles[1],
        current_euler_angles[2],
        )
    
    # also rotate the trajectory arrow
    rotation = R.from_euler('xyz', current_euler_angles, degrees = True)
    rotated_traj_vec = rotation.apply(traj_vec)
    # Create the trajectory arrow
    current_traj_arrow_pos = all_traj_arrow_pos[i]
    rotated_traj_arrow = generate_traj_arrow(
        start = current_traj_arrow_pos,
        direction = rotated_traj_vec,
    )

    ### add real-world axes 
    axes_actor = make_custom_axes_actor()
    #rotate real-world axes
    rotated_frame = rotation.apply(np.eye(3))
    place_axes_actor(
        axes_actor,
        origin=ax_origin,
        x_dir = rotated_frame[0],
        y_dir = rotated_frame[1],
        z_dir = rotated_frame[2],
        )

    #render mesh
    im = render_frame(
        plotter,
        [rotated_mesh, rotated_traj_arrow],
        camera_position = CAMERA_POSITION,
        background_color = 'white',
        axes_actor = axes_actor,
        )
    rendered_images.append(im)
    

## get the reconstruction of this mesh
shcols = [x for x in cellinfo.columns if 'shco' in x]
shcoeffs = cellinfo[shcols].values
recon_mesh, grid = get_reconstruction_from_coeffs(shcoeffs.reshape(2,lmax+1,lmax+1))
recon_mesh = pv.PolyData(recon_mesh)
### add real-world axes 
place_axes_actor(
    axes_actor,
    origin=ax_origin,
    x_dir = rotated_frame[0],
    y_dir = rotated_frame[1],
    z_dir = rotated_frame[2],
    )

#render mesh
im = render_frame(
    plotter,
    [recon_mesh, rotated_traj_arrow],
    camera_position = CAMERA_POSITION,
    background_color = 'white',
    axes_actor = axes_actor,
    )
rendered_images.append(im)

#close plotter
plotter.close()







###### plot the meshes
hspace = 0 #-0.05
wspace = 0

fig, axes = plt.subplots(
    1, 4,
    figsize = (16, 4),
    gridspec_kw={'wspace': wspace, 'hspace': hspace},
    )

for a, ax in enumerate(axes):
    ax.imshow(rendered_images[a], aspect = 'auto')
    ax.axis('off')
 


plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)
# plt.savefig(f'C:/Users/Aaron/Desktop/{align}_shape_space.png',dpi = 300, bbox_inches = 'tight')

