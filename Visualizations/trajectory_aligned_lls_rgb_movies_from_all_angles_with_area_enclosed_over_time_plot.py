

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import seaborn as sns
import tifffile
import matplotlib.gridspec as gridspec
from neutrophil_shape.config.loader import load_config
from neutrophil_shape.CustomFunctions import utils
import multiprocessing

whichpcs = (1,2)

down_factor = 2
config = load_config(microscope_type='lls')
config._alignment = 'trajectory'
time_interval = config.im_params.time_interval
whichpcs = (1,2)
origins = config.db_params.origins
basedir = config.common.savedir
datadir = basedir.joinpath('shape_data')
dbdir = basedir.joinpath('detailed_balance')
localdir = config.experiment.lls.localdir
moviedir = localdir / 'singlecells'
xyres = config.im_params.xyres


state_order = ['zero','low','high']

FullFrame = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col = 0)
FullFrame['real_time'] = FullFrame.time.copy()

#open aers previously calculated
allaers = pd.read_csv(dbdir.joinpath(utils.whichpc_string(whichpcs) + '_raw_transition_aer_cf.csv'), index_col = 0)

#merge aer and cf info
TotalFrame = pd.merge(FullFrame, allaers,on=['CellID','real_time','frame'],how='left')
TotalFrame = TotalFrame.sort_values(['CellID','time'])


############# GET aer_state for REAL DATA

real_states = TotalFrame.groupby('CellID').apply(utils.get_aer_state).reset_index(level=1, drop=True).reset_index()
real_states = utils.get_aer_state_chunk_ids(real_states, group_factor = 'CellID')
observedstarts, observedstops = utils.get_observed_aer_state_chunk_starts_stops(real_states, group_factor = 'CellID')

### get chunks where both start and stop are observed
whole_chunks = [o for o in observedstarts if o in observedstops]
whole_chunk_real_states = real_states[real_states.chunk_id.isin(whole_chunks)].copy()
highdf = whole_chunk_real_states[whole_chunk_real_states.aer_state == 'high'].copy()



image_color_dict = {
    'membrane': np.array([0,1,0]),
    'nucleus': np.array([1,0,1]), #[0,1,1]
    'actin': np.array([1,0,1]),
    'myosin': np.array([1,0,1]),
    }



dirs = [d for d in moviedir.glob("*/") if d.is_dir()]
# dirs = [d for d in moviedir.glob('*') if '20240611_488_EGFP-CAAX_640_actin-halotag_cell2_01' in d.name]
for d in dirs:
    cellname = d.name
    image = tifffile.imread(d / f'{cellname}_traj_aligned.ome.tiff')
    framelist = pd.read_csv(d / f'{cellname}_framelist.csv', index_col = 0)
    framelist = framelist[framelist.columns[0]]
    #open all of the data
    celldf = real_states[real_states.CellID==cellname].sort_values('time').reset_index(drop=True)
    #add a movie column
    celldf['Movie'] = [x.split('_frame')[0] for x in celldf.cell.to_list()]
    celldf['time_min'] = celldf.time.values/60
    celldf['area_enclosed'] = celldf.aer.values * celldf.time_elapsed.values
    #get times for all the frames included in the movie
    times = celldf.time


    #make all of the max projections
    xyproj = np.max(image, axis = -3)
    xzproj = np.max(image, axis = -2)
    yzproj = np.max(image, axis = -1)
    #flip the yx projection to be portrait orientation
    yzproj = yzproj[..., ::-1].copy()
    
    
    # #bleaching correction
    # xyproj_bc = np.zeros(xyproj.shape)
    # xzproj_bc = np.zeros(xzproj.shape)
    # yzproj_bc = np.zeros(yzproj.shape)
    # linearmaxes = np.linspace(np.percentile(xyproj[0,0],99),np.percentile(xyproj[-1,0],99), len(xyproj))
    # for n in range(len(xyproj)):
    #     #all images have a min of zero so just divide by max
    #     xyproj_bc[n] = xyproj[n]/linearmaxes[n]
    #     xzproj_bc[n] = xzproj[n]/linearmaxes[n]
    #     yzproj_bc[n] = yzproj[n]/linearmaxes[n]

    ### convert to rgb
    structure = celldf.structure.iloc[0]
    allrgb = []
    for stack in [xyproj, xzproj, yzproj]:
        # Assume (N, C, H, W) - adjust if your layout differs
        N, C, H, W = stack.shape
        stack = stack.astype(np.float32)
    
        normed = np.zeros_like(stack, dtype=np.float32)
    
        # Percentile normalization, per-image, computed separately for each channel
        for c in range(C):
            channel = stack[:, c]                      # (N, H, W)
            flat = channel.reshape(N, -1)               # (N, H*W)
            
            lo_percent = 95 if c == 1 else 5
            hi_percent = 99.9999 if c == 1 else 99.8
            lo = np.percentile(flat, lo_percent, axis=1)         # (N,)
            hi = np.percentile(flat, hi_percent, axis=1)     # (N,)
    
            lo = lo.reshape(N, 1, 1)
            hi = hi.reshape(N, 1, 1)
    
            denom = np.where(hi > lo, hi - lo, 1.0)     # avoid divide-by-zero for flat channels
            normed[:, c] = np.clip((channel - lo) / denom, 0, 1)
    
        # Apply colors: (N, C, H, W) x (C, 3) -> (N, H, W, 3)
        rgb = (normed[:, 0, ..., np.newaxis] * image_color_dict['membrane'] +
               normed[:, 1, ..., np.newaxis] * image_color_dict[structure])
        rgb = np.clip(rgb, 0, 1)
        rgb_8bit = (rgb * 255).astype(np.uint8)
        allrgb.append(rgb_8bit)
    
    
    
    
    # Create a figure
    fig = plt.figure(figsize=(7, 7))
    # Create a GridSpec with 2 rows and 2 columns
    gs = gridspec.GridSpec(2, 2, width_ratios=[image.shape[-3], image.shape[-1]],
                           height_ratios=[image.shape[-2], image.shape[-3]],
                           wspace = 0.01,
                           hspace = 0.05,
                           figure=fig)
    fig.patch.set_facecolor('black')
    fig.subplots_adjust(left=0.08, right=1, top=1, bottom=0.08, wspace=0, hspace=0)
    # Use first column for both subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 0])
    
    
    scalebar_x_displacement = xyproj.shape[-1]-10
    scalebar_y_displacement = xyproj.shape[-2]-14
    scalebar_length = 10
    resolution = 0.145*4 #um / pixel
    
    #scalebar
    sb = ax2.plot([scalebar_x_displacement-(scalebar_length/resolution), scalebar_x_displacement],
            [scalebar_y_displacement, scalebar_y_displacement],
            lw = 3,
            color = 'white',
            zorder=2)
    
    #scalbar text
    sb_label = ax2.text(scalebar_x_displacement-(scalebar_length/resolution)-6,
                        scalebar_y_displacement + 10,
                        f'{scalebar_length} μm',
                        color = 'white',
                        fontdict = {'fontsize': 10})
    
    
    ### time label
    timer = ax2.text(3,18,'00:00', color = 'white', fontdict = {'fontsize': 24})
    
    #all the images
    xyimsh = ax2.imshow(allrgb[0][0], zorder = 1)
    xzimsh = ax3.imshow(allrgb[1][0], zorder = 1)
    yzimsh = ax1.imshow(allrgb[2][0], zorder = 1)

    ### add colored text for the channels
    ax1.text(5,10,"Membrane", fontdict = {'fontsize': 14, 'color':image_color_dict['membrane']})
    ax1.text(5,20,structure.title(), fontdict = {'fontsize': 14, 'color':image_color_dict[structure]})

    x = celldf.time_min.to_numpy()
    y = celldf.area_enclosed.cumsum().to_numpy()
    factors = celldf.aer_state.to_numpy()

    # if color_map is None:
    categories = pd.unique(factors)
    # palette = plt.cm.tab10.colors
    zero_color = '#a8a8a8'   
    low_color = '#d1a53d'  
    high_color = '#d14c45'
    palette = ['#a5a5a5',high_color, low_color, zero_color]
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

    
    # color_map = {'high': high_color, 'low': low_color, 'zero': zero_color, 'nan': np.nan}


    # Build line segments: each segment connects point i to point i+1
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color each segment by the factor value at its starting point
    seg_colors = [color_map[f] for f in factors[:-1]]

    lc = LineCollection(segments, colors=seg_colors, linewidth=2)
    ax4.add_collection(lc)
    # ax4.set_xlim(x.min(), x.max())
    ax4.set_ylim(y[~np.isnan(y)].min(), y[~np.isnan(y)].max())

    # Build a legend manually since LineCollection doesn't auto-generate one
    handles = [plt.Line2D([0], [0], color=c, lw=2, label=cat)
               for cat, c in list(color_map.items())[1:]]
    ax4.legend(handles=handles, title='AER state', loc = 'lower right')
    
    
    ax4.set_xlabel('Time (min)', color = 'white')
    ax4.set_ylabel('Area Enclosed', color = 'white')
    ax4.set_facecolor('black')
    ax4.tick_params(axis='x', colors='white')
    ax4.tick_params(axis='y', colors='white')
    ax4.set_xticks(range(0,65,5))
    ax4.set_xticklabels(np.arange(0,65,5).astype(str), fontsize = 8)
    for spine in ax4.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(1)
    #vline for the aer graph
    vline = ax4.axvline(color='0.6', zorder=1)
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(1)
    
    
    
    
    # make function for updating point position
    def animate(i,):
        #set the current set of data
        xyimsh.set_data(allrgb[0][i])
        xyimsh.set_zorder(0)
        xzimsh.set_data(allrgb[1][i])
        xzimsh.set_zorder(0)
        yzimsh.set_data(allrgb[2][i])
        yzimsh.set_zorder(0)
        
        #animate the vline on the aer plot
        vtime = times[i]/60
        vline.set_xdata([vtime])
        
        ### scale bar info
        scalebar_x_displacement = xyproj.shape[-1]-10
        scalebar_y_displacement = xyproj.shape[-2]-14
        scalebar_length = 10
        resolution = 0.145*4 #um / pixel
        #scalebar "animation"
        sb[0].set_data([scalebar_x_displacement-(scalebar_length/resolution), scalebar_x_displacement],
                [scalebar_y_displacement, scalebar_y_displacement])
        #scalebar label "animation
        sb_label.set_text(f'{scalebar_length} μm')
        
        #timer animation
        timer.set_text(utils.format_seconds(times[i]))
        
        return xyimsh, xzimsh, yzimsh, sb[0], sb_label, timer, vline#, aerplot
    
    
    
    #add two to the frame count to adjust the range function and to add a blank frame at the beginning
    ani = FuncAnimation(fig, animate, interval=10, blit=True, repeat=True,
                        frames=len(xyproj),)
    
    
    plt.show()
    

    ani.save(d.joinpath(cellname + '_traj_aligned_aeplot_rgb.mp4'), fps=5, dpi = 300)#, extra_args=['-vcodec', 'libx264'])

    
    plt.close(fig)