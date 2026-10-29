

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from neutrophil_shape.config.loader import load_config
from neutrophil_shape.CustomFunctions.DetailedBalance import get_dwell_times
from neutrophil_shape.CustomFunctions.utils import whichpc_string




#define some variables
treatments = ['Random']
whichpcs = (1,2)
wpc_str = whichpc_string(whichpcs)
wpc_binlist = [f'PC{w}bins' if w > 0 else f'PC{abs(w)}bins_abs' for w in whichpcs]
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'
pc_combos = config.common.pc_combos
time_interval = config.im_params.time_interval
origin = config.db_params.origins[pc_combos.index(whichpcs)]

#define some directories
savedir = config.common.savedir
datadir = savedir.joinpath('shape_data')
dbdir = savedir.joinpath('detailed_balance')

#open the centers of the binned PCs
centers = pd.read_csv(datadir.joinpath('PC_bin_centers.csv'), index_col=0)
nbins = len(centers.iloc[:,0])

######## open all of the data
### shape and behavior ###
shapedf = pd.read_csv(datadir / 'All_Data_with_CGPS_bins.csv', index_col = 0)
shapedf = shapedf[shapedf.Treatment.isin(treatments)]
########### interpolated transitions ###########da
transdf = pd.read_csv(dbdir.joinpath(f'{wpc_str}_interpolated_transitions.csv'), index_col=0)
transdf = transdf[transdf.Treatment.isin(treatments)]
transdf['dwell_time'] = get_dwell_times(transdf)
transdf = transdf.rename(columns = {
    xy:wpc for xy, wpc in zip(['from_x','from_y'],wpc_binlist)
    })


## which metrics to average in the CGPS
metrics = ['dwell_time','speed','directional_autocorrelation']
metric_labels = ['Dwell Time','Speed','Persistence']
units = ['Seconds','μm/second','']
map_dict = {}


#### get data for all of the metrics
binrange = np.arange(1,nbins+1)
for m, metric in enumerate(metrics):
    if metric == 'dwell_time':
        tempdf = transdf
    else:
        tempdf = shapedf
    key = metric_labels[m]
    map_dict[key] = {
        'units': units[m],
        'data': (
            tempdf.groupby(wpc_binlist)[metric].mean()
              .unstack(wpc_binlist[0])
              .reindex(index=binrange, columns=binrange)
              .fillna(0)
              .to_numpy()
              ),
        }
    map_dict[key]['vmax'] = np.percentile(map_dict[key]['data'],99)
    # map_dict[key]['vmin'] = min(map_dict[key]['data'][map_dict[key]['data']>0.1]) if metric == 'directional_autocorrelation' else 0



for metric_label in map_dict.keys():
    
    tempdict = map_dict[metric_label]
    vmin = 0.5 if metric_label == 'Persistence' else 0
    #### start building the figure
    fig, ax = plt.subplots(figsize=(5,5))
    cbar_ax = fig.add_axes([0.94, .188, .025, .661])

    #single colorbar axis
    # cbar_ax = fig.add_axes([.95, .133, .02, .685])

    
    #plot heatmap with seaborn
    sns.heatmap(
        tempdict['data'],
        vmin= vmin, #tempdict['vmin'], 
        vmax = tempdict['vmax'], 
        # center=0,
        cmap='rocket',
        square=True,
        xticklabels = True,
        yticklabels = True,
        ax = ax,
        cbar_ax = cbar_ax,
    )
    
    #correct axis orientations
    ax.invert_yaxis()
    #get rid of ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(0.5,nbins+0.5)[[0,nbins//2,-1]])
    ax.set_xticklabels([round(centers[f'PC{whichpcs[1]}'].iloc[x],2) for x in [0,nbins//2, int(nbins-1)]],
                       fontsize = 12)
    ax.set_yticks(np.arange(0.5,nbins+0.5)[[0,nbins//2,-1]])
    ax.set_yticklabels([round(centers[f'PC{whichpcs[1]}'].iloc[x],2) for x in [0,nbins//2, int(nbins-1)]],
                       fontsize = 12)
    #set axis titles
    ax.set_xlabel(f'PC{whichpcs[0]}', fontsize = 20)
    ax.set_ylabel(f'PC{whichpcs[1]}', fontsize = 20)

    #set title
    ax.set_title(metric_label, fontsize = 28)


    # adjust colorbar tick label size
    cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(),fontsize=12)
    cbar_ax.get_yaxis().labelpad = 22
    cbar_ax.set_ylabel(tempdict['units'], fontsize = 20, rotation=270)
    
    
    plt.tight_layout()
    
    
    
    plt.savefig(__file__.split('.')[0] + f'_{metric_label}.png', dpi = 500, bbox_inches='tight')
    
    


