
import vtk
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from neutrophil_shape.config.loader import load_config
from neutrophil_shape.CustomFunctions import utils
from tqdm import tqdm

# ----------------------------- CONFIG ---------------------------------- #

# Output
FPS = 7

# Rendering
MESH_COLOR = "corn_silk"
BACKGROUND = "lightsteelblue"
WINDOW_SIZE = [800, 800]
CAMERA_POSITION = None   # None -> auto-fit each frame; set explicitly (e.g.
                         # "iso", "xy", or a pyvista camera position tuple)
                         # to lock the camera across frames.
CAMERA_POSITION = [
    (0, 0, 50),  # position
    (0, 0, 0),  # focal point
    (0.0, 1.0, 0.0),  # view up
]

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

def render_frame(plotter, mesh, axes_actor = None):
    """Render the current mesh state off-screen and return an RGB image array."""
    plotter.clear()
    plotter.enable_lightkit()
    plotter.add_mesh(mesh, color=MESH_COLOR,
                     show_edges=False,
                    #  smooth_shading=True,      # interpolate normals across faces -> visible curvature
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





whichpcs = (2,8)

### open config and get directories
config = load_config(microscope_type='lls')
config._alignment = 'trajectory'
savedir = config.common.savedir
localdir = config.experiment.lls.localdir
moviedir = localdir / 'singlecells'
meshdir = localdir / 'meshes'
time_interval = config.im_params.time_interval
datadir = savedir.joinpath('shape_data')
dbdir = savedir.joinpath('detailed_balance')


FullFrame = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col = 0)


for cellname in FullFrame.CellID.unique():

    #open all of the data
    celldf = FullFrame[FullFrame.CellID == cellname].copy()
    aers = pd.read_csv(dbdir / f'{utils.whichpc_string(whichpcs)}_raw_transition_aer_cf.csv', index_col = 0)
    aers = aers.rename(columns={'real_time':'time'})
    ### merge aers
    TotalFrame = pd.merge(celldf, aers, on=[x for x in aers.columns if x in celldf.columns], how = 'left')
    TotalFrame = TotalFrame.sort_values('time').reset_index(drop=True)
    n_frames = TotalFrame.shape[0]
    times = TotalFrame.time.values
    
    #add a movie column
    TotalFrame['Movie'] = [x.split('_frame')[0] for x in TotalFrame.cell.to_list()]
    TotalFrame['time_min'] = TotalFrame.time.values/60
    TotalFrame['area_enclosed'] = TotalFrame.aer.values * TotalFrame.time_elapsed.values
    # TotalFrame['ae_cumsum'] = TotalFrame.area_enclosed.cumsum()
    #get times for all the frames included in the movie
    #(which may have been dropped from dataframes in analysis)

     # ---- Off-screen PyVista plotter (reused across frames for speed) ---- #
    try:
        pv.start_xvfb()  # needed on headless Linux without a display;
                          # harmless to skip on systems that don't need it
    except Exception:
        pass
    
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    
    print("Pre-rendering mesh frames...")
    mesh_images = []
    for i, row in tqdm(TotalFrame.iterrows(), total=n_frames):
        celldir = next(meshdir.glob(f"*{row.cell}*"))
        mesh = pv.read(celldir)
        mesh = rotate_mesh(mesh, row.Euler_Angles_X, row.Euler_Angles_Y, row.Euler_Angles_Z)
        
        # axes = pv.Axes()
        # axes = rotate_axes_actor(axes, row.Euler_angles_X, row.Euler_angles_Z,  row.Width_Rotation_Angle)
        # axes.axes_actor.SetPosition(AXES_POSITION)
        
        axes_actor = make_axes_actor(
            row.Euler_Angles_X,
            row.Euler_Angles_Y,
            row.Euler_Angles_Z,
            AXES_POSITION,
            length = 2.5,
            label_size = 16,
            )
               
        
        mesh_images.append(render_frame(plotter, mesh, axes_actor))
    plotter.close()
    
    

               
    # ------------------------- Matplotlib figure -------------------------- #
    fig, (ax_mesh, ax_plot) = plt.subplots(1, 2, figsize=(12, 6))
    
    im = ax_mesh.imshow(mesh_images[0])
    ax_mesh.axis("off")
    # ax_mesh.set_title("Mesh")
    ### time label
    timer = ax_mesh.text(3,18,utils.format_seconds(0), color = 'black', fontdict = {'fontsize': 16})
    

    #aer graph
    ax_plot.plot(TotalFrame.time_min, TotalFrame.area_enclosed.cumsum(), color = 'black', zorder = 2)
    ax_plot.set_xlabel('Time (min)', color = 'black')
    ax_plot.set_ylabel('Area Enclosed', color = 'black')
    ax_plot.set_xticks(range(0,65,5))
    ax_plot.set_xticklabels(np.arange(0,65,5).astype(str), fontsize = 8)

    #vline for the aer graph
    time_cursor = ax_plot.axvline(color='0.6', zorder=1)
    
    fig.tight_layout()
    
    def update(i):
        im.set_data(mesh_images[i])
        vtime = times[i]/60
        time_cursor.set_xdata([vtime])
        #timer animation
        timer.set_text(utils.format_seconds(times[i]))
        return im, time_cursor
    
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / FPS, blit=False
    )
    
    
    
    anim.save(moviedir.joinpath(cellname, cellname + f'_animated_{utils.whichpc_string(whichpcs)}_mesh_ae_plot.mp4'),
             fps=FPS, dpi = 300)#, extra_args=['-vcodec', 'libx264'])


    plt.close(fig)

    

    
      
    
    # scalebar_x_displacement = xyproj.shape[-1]-10
    # scalebar_y_displacement = xyproj.shape[-2]-14
    # scalebar_length = 10
    # resolution = 0.145*4 #um / pixel
    
    # #scalebar
    # sb = ax2.plot([scalebar_x_displacement-(scalebar_length/resolution), scalebar_x_displacement],
    #         [scalebar_y_displacement, scalebar_y_displacement],
    #         lw = 3,
    #         color = 'white',
    #         zorder=2)
    
    # #scalbar text
    # sb_label = ax2.text(scalebar_x_displacement-(scalebar_length/resolution)-6,
    #                     scalebar_y_displacement + 10,
    #                     f'{scalebar_length} μm',
    #                     color = 'white',
    #                     fontdict = {'fontsize': 10})
    

    
