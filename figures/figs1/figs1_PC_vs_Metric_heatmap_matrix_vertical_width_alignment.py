

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from neutrophil_shape.config.loader import load_config

def closest(lst, K):  
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]
color_scale = pd.DataFrame({'color':list(sns.diverging_palette(20, 220, n=200).as_hex()),
              'value':list(np.arange(-1,1,2/200))})
#Scatter plots for cell metrics and the PCs



#get directories and open separated datasets
#get directories and open separated datasets
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory_shape'
datadir = config.common.savedir / 'shape_data'
TotalFrame = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col=0)



#all the metrics we want to plot by their name in the dataframe
metrics =  [['Cell_Volume',
             'Cell_SurfaceArea',
             'Volume_Front_Ratio',
             'Volume_Left_Ratio',
             'Volume_Top_Ratio',
             'Cell_Sphericity',
             ],
            ['Cell_MajorAxis_Length',
             'Cell_MedianAxis_Length',
             'Cell_MinorAxis_Length',
             'Cell_Aspect_Ratio',
             'Cell_MajorAxis_Vec_X',
            'Cell_MajorAxis_Vec_Y',
            'Cell_MajorAxis_Vec_Z',
            'Cell_MedianAxis_Vec_X',
            'Cell_MedianAxis_Vec_Y',
            'Cell_MedianAxis_Vec_Z',
            'Cell_MinorAxis_Vec_X',
            'Cell_MinorAxis_Vec_Y',
            'Cell_MinorAxis_Vec_Z',
             ],
            ['speed',
             'directional_autocorrelation']
            ]

labelz = [['Cell Volume (µm$^3$)',
           'Cell Surface\nArea (µm$^2$)',
           'Front-Back Volume\nRatio',
           'Left-Right Volume\nRatio',
           'Top-Bottom Volume\nRatio',
           'Cell Sphericity',
           ],
          ['Cell Major Axis\nLength (µm)',
           'Cell Median Axis\nLength (µm)',
           'Cell Minor Axis\nLength (µm)',
           'Aspect Ratio',
           'Major Axis X\nComponent',
           'Major Axis Y\nComponent',
           'Major Axis Z\nComponent',
           'Median Axis X\nComponent',
           'Median Axis Y\nComponent',
           'Median Axis Z\nComponent',
           'Minor Axis X\nComponent',
           'Minor Axis Y\nComponent',
           'Minor Axis Z\nComponent',
           ],
          ['Instantaneous\nSpeed (µm/sec)',
           'Persistence',
           ]#,'Directional Autocorrelation',
          ]
#get PCs in order
npcs = config.common.npcs
PCs = ['PC'+str(i) for i in range(1,npcs+1)]
#add them together and select them in the dataframe
totalcorr = TotalFrame[[x for y in metrics for x in y]+PCs].corr()
PCsAndMetrics = totalcorr.loc[:,PCs]
PCsAndMetrics = PCsAndMetrics.drop(index=PCs)

fig, axes = plt.subplots(len(metrics), 1, figsize=(15,25),
                         gridspec_kw={'height_ratios':[len(x) for x in metrics],
                                      'hspace':0.05})
for i, m in enumerate(metrics):
    ax = axes[i]
    temp = PCsAndMetrics.loc[m,:].copy()
    cbarbool = False if i != len(metrics)-1 else True
    sns.heatmap(
        temp, 
        vmin=-1, 
        vmax=1,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
        # xticklabels = True,
        # yticklabels = True,
        # annot = True,
        # fmt = '.2f',
        cbar = False,
        cbar_kws={'fraction':0.05, 'pad':0.01},#, 'shrink': 0.5}
        ax = ax)
    if  i == 0:
        ax.set_xticklabels(
            PCs,
            fontsize = 26
        )
        ax.tick_params('x',top=True, labeltop=True, bottom=False, labelbottom=False ,length=6, width=3)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
        
    ax.set_yticklabels(
        labelz[i],
        # rotation=45,
        # horizontalalignment='right',
        fontsize = 26
    )
    
    


    #tick params
    ax.tick_params('y',length=6, width=3)

    # #scooch the x axis labels by a certain amount
    # dx = 10/72.; dy = 0/72. 
    # offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    # for label in ax.xaxis.get_majorticklabels():
    #     label.set_transform(label.get_transform() + offset)
        
    # if i == 1:
    #     pos = ax.get_position()
    #     ax.set_position([pos.x0-1, pos.y0, pos.width, pos.height])
    # if i == 2:
    #     pos = ax.get_position()
    #     ax.set_position([pos.x0 - 0.3, pos.y0, pos.width, pos.height])

# Get the bounding boxes of the first and last heatmap axes
top_ax_pos    = axes[0].get_position()
bottom_ax_pos = axes[-1].get_position()

# Extract left edge and width from the heatmap axes
hm_left  = top_ax_pos.x0
hm_width = top_ax_pos.width

# Place colorbar just below the bottom heatmap
cbar_bottom = bottom_ax_pos.y0 - 0.02  # gap below last heatmap
cbar_height = 0.013

cbar_ax = fig.add_axes([hm_left, cbar_bottom, hm_width, cbar_height])

# cbar_ax = fig.add_axes([0.84, 0.3, 0.027, 0.30])  # [left, bottom, width, height]
# cbar_ax = fig.add_axes([0.2798, 0.09, 0.4455, 0.013]) 


# Add the colorbar to the new axis
cbar = fig.colorbar(axes[-1].collections[0], cax=cbar_ax, orientation='horizontal')
cbar.set_label('Pearson Coefficient', fontsize=26)
cbar.ax.xaxis.set_label_position('bottom')
cbar.ax.set_xticklabels(np.linspace(-1,1,len(cbar.ax.get_xticklabels())).astype(str),fontsize=20)


plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi=500)