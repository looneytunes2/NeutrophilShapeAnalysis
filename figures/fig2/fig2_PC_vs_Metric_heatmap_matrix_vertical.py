# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 16:05:31 2025

@author: Aaron
"""


################ smaller PC vs metric plots for CICON ################
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import seaborn as sns

def closest(lst, K):  
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]
color_scale = pd.DataFrame({'color':list(sns.diverging_palette(20, 220, n=200).as_hex()),
              'value':list(np.arange(-1,1,2/200))})
#Scatter plots for cell metrics and the PCs



#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
TotalFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
# make sure all categories are ordered
TotalFrame['Treatment'] = pd.Categorical(TotalFrame.Treatment.to_list(), categories=['Random','Pre-Galvanotaxis','Galvanotaxis','DMSO','CK666','Para-Nitro-Blebbistatin'], ordered=True)
TotalFrame['Experiment'] = pd.Categorical(TotalFrame.Experiment.to_list(), categories=['Galvanotaxis','Drug'], ordered=True)




#all the metrics we want to plot by their name in the dataframe
metrics =  [['Cell_Volume','Cell_SurfaceArea','Volume_Front_Ratio','Volume_Right_Ratio','Volume_Top_Ratio','Cell_Sphericity'],
            ['Cell_MajorAxis','Cell_MinorAxis','Cell_Aspect_Ratio','Cell_UpDownAngle','Cell_LeftRightAngle','Cell_TotalAngle','LengthAlongTrajectory'],
            ['speed','directional_autocorrelation']
            ]

labelz = [['Cell Volume (µm$^3$)','Cell Surface\nArea (µm$^2$)','Front-Back Volume\nRatio','Right-Left Volume\nRatio','Top-Bottom Volume\nRatio','Cell Sphericity'],
          ['Cell Major Axis\nLength (µm)','Cell Minor Axis\nLength (µm)','Aspect Ratio','Long-Axis X-Z\nAngle (°)','Long-Axis X-Y\nAngle (°)','Long-Axis Total\nAngle (°)','Length Along\nTrajectory (µm)'],
          ['Instantaneous\nSpeed (µm/sec)','Persistence']#,'Directional Autocorrelation',
          ]
#get PCs in order
PCs = list(np.unique([re.search('PC\d*',x)[0] for x in TotalFrame.columns.to_list() if re.search('PC\d*',x) is not None]))
PCs.sort(key=lambda x: float(x.split('PC')[1]))
#add them together and select them in the dataframe
totalcorr = TotalFrame[[x for y in metrics for x in y]+PCs].corr()
PCsAndMetrics = totalcorr.loc[:,PCs]
PCsAndMetrics = PCsAndMetrics.drop(index=PCs)

fig, axes = plt.subplots(len(metrics), 1, figsize=(15,25), gridspec_kw={'height_ratios':[len(x) for x in metrics]})
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
            fontsize = 28
        )
        ax.tick_params('x',top=True, labeltop=True, bottom=False, labelbottom=False ,length=6, width=3)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
        
    ax.set_yticklabels(
        labelz[i],
        # rotation=45,
        # horizontalalignment='right',
        fontsize = 28
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

# cbar_ax = fig.add_axes([0.84, 0.3, 0.027, 0.30])  # [left, bottom, width, height]
cbar_ax = fig.add_axes([0.211, 0.09, 0.603, 0.013]) 


# Add the colorbar to the new axis
cbar = fig.colorbar(axes[-1].collections[0], cax=cbar_ax, orientation='horizontal')
cbar.set_label('Pearson Coefficient', fontsize=28)
cbar.ax.xaxis.set_label_position('bottom')
cbar.ax.set_xticklabels(np.linspace(-1,1,len(cbar.ax.get_xticklabels())).astype(str),fontsize=22)


plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi=500)