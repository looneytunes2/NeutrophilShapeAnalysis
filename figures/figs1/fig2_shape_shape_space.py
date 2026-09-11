
import pickle as pk
import pyvista as pv
import matplotlib.pyplot as plt
from neutrophil_shape.config.loader import load_config
from neutrophil_shape.CustomFunctions.PCvisualization import render_frame

# ----------------------------- CONFIG ---------------------------------- #

## perspective list for projection plane in pc order
perspective_dict = {
    'shape':['xy','xy','xy','xz','xz','xy','xy','xy'],
    'trajectory_shape':['xy','xy','xz','xz','xy','xz','xy','xy'],
    'trajectory':['xy','xy','xz','xz','yz','xy','xz','xy'],
}
    

WINDOW_SIZE = [5000, 5000]

cameradist = 40
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
     (cameradist, 0, 0),  # position
     (0, 0, 0),  # focal point
     (0.0, 0.0, 1.0),  # view up
     ],
    }
    

AXES_POSITIONS = {
    'xy': (-9,-9,0),
    'xz': (-9,0,-9),
    'yz': (0,-9,-9),
    }
    

align = 'shape'
    
### open config and get directories
config = load_config(microscope_type='confocal')
config._alignment = align
npcs = config.common.npcs
savedir = config.common.savedir
datadir = savedir.joinpath('shape_data')
pcmeshdir = datadir.joinpath('PC_meshes')
pcmeshlist = list(pcmeshdir.glob('*.vtp'))
pcbinnum = int(len(pcmeshlist)/npcs)
perspective_list = perspective_dict[align]

#also open PCA for % variance explained
pcapath = datadir.joinpath('pca.pkl')
pca = pk.load(open(pcapath,'rb')) 
# How much variance is explained?
variance_percent = pca.explained_variance_ratio_ * 100



hspace = -0.05
wspace = 0

#set up figure
fig, axes = plt.subplots(
    npcs, pcbinnum,
    figsize = (2*pcbinnum, 2*npcs),
    gridspec_kw={'wspace': wspace, 'hspace': hspace},
    )
# ---- Off-screen PyVista plotter (reused across frames for speed) ---- #
try:
    pv.start_xvfb()  # needed on headless Linux without a display;
                      # harmless to skip on systems that don't need it
except Exception:
    pass

plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)

for pcmeshpath in pcmeshlist[::-1]:
    # get pc and pc bin from filename
    pc_label, pcbin = pcmeshpath.stem.split('_')
    pcbin = int(pcbin)
    pc_num = int(pc_label.split('PC')[-1])
    
    # get perspective for this PC
    current_perspective = perspective_list[pc_num-1]
    
    # select axis
    ax = axes[pc_num-1, pcbin-1]
    
    # open the mesh and render
    mesh = pv.read(pcmeshpath)          
    im = render_frame(
        plotter,
        mesh,
        camera_position = CAMERA_POSITIONS[current_perspective],
        background_color = 'white',
        )
    
    ax.imshow(im, aspect = 'auto')
    ax.axis('off')
    
plotter.close()    




# --- Column labels (top row, centered above each column) ---
sigmalist = [str(r)+'σ' for r in range(-2,3)]
for col in range(pcbinnum):
    ax = axes[0, col]
    ax.annotate(
        sigmalist[col],
        xy=(0.5, 1), xycoords='axes fraction',
        xytext=(0, 2), textcoords='offset points',
        ha='center', va='bottom', fontfamily='sans-serif',
        fontsize=16, fontweight='bold',
        annotation_clip=False,
    )

# --- Row labels (left column, centered beside each row) ---
for row in range(npcs):
    ax = axes[row, 0]
    ax.annotate(
        f'PC{row + 1}\n{perspective_list[row].upper()}\n{round(variance_percent[row],1)}%',
        xy=(0, 0.5), xycoords='axes fraction',
        xytext=(-20, 0), textcoords='offset points',
        ha='center', va='center', fontfamily='sans-serif',
        fontsize=16, fontweight='bold',
        annotation_clip=False,
    )

fig.subplots_adjust(left=0.05, top=0.97, wspace=wspace, hspace=hspace)


# plt.savefig(f'C:/Users/Aaron/Desktop/{align}_shape_space.png',dpi = 300, bbox_inches = 'tight')
plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)
    
