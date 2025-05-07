# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:38:41 2024

@author: Aaron
"""

################ smaller PC vs metric plots for CICON ################
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import seaborn as sns

def closest(lst, K):  
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]
color_scale = pd.DataFrame({'color':list(sns.diverging_palette(20, 220, n=200).as_hex()),
              'value':list(np.arange(-1,1,2/200))})
#Scatter plots for cell metrics and the PCs



#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_smooth/'
datadir = basedir + 'Data_and_Figs/'
TotalFrame = pd.read_csv(datadir + 'Shape_Metrics_transitionPCbins.csv', index_col=0)
# make sure all categories are ordered
TotalFrame['Treatment'] = pd.Categorical(TotalFrame.Treatment.to_list(), categories=['Random','Pre-Galvanotaxis','Galvanotaxis','DMSO','CK666','Para-Nitro-Blebbistatin'], ordered=True)
TotalFrame['Experiment'] = pd.Categorical(TotalFrame.Experiment.to_list(), categories=['Galvanotaxis','Drug'], ordered=True)




#all the metrics we want to plot by their name in the dataframe
metrics =  ['Cell_Volume','Cell_SurfaceArea','Cell_MajorAxis','Cell_MinorAxis',
          'LengthAlongTrajectory','WidthAlongTrajectory','speed','Turn_Angle',
          'Cell_Aspect_Ratio','Volume_Front_Ratio','Volume_Right_Ratio','Volume_Top_Ratio',
          'Cell_Sphericity','Cell_UpDownAngle','Cell_LeftRightAngle','Cell_TotalAngle']
#get PCs in order
PCs = list(np.unique([re.search('PC\d*',x)[0] for x in TotalFrame.columns.to_list() if re.search('PC\d*',x) is not None]))
PCs.sort(key=lambda x: float(x.split('PC')[1]))
#add them together and select them in the dataframe
metric_frame = TotalFrame[metrics+PCs]



fig, axes = plt.subplots(len(PCs), len(metrics), 
                         figsize=(2.15*len(metrics),2*len(PCs)))#, sharex=True)
# #one colorbar for full axis
# cbar_ax = fig.add_axes([1, .2, .02, .7])
# palette = sns.diverging_palette(20, 220, n=200)
# # Convert seaborn palette to a LinearSegmentedColormap
# cmap = LinearSegmentedColormap.from_list("my_cmap", palette)
# # Create a ScalarMappable object using the colormap
# sm = ScalarMappable(cmap=cmap)
# # adjust colorbar tick label size
# cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(),fontsize=22)
# fig.colorbar(sm, cbar_ax)


for q in range(axes.shape[0]):
    for i, ax in enumerate(axes[q,:]):
        tempframe = metric_frame[[metrics[i],f'PC{q+1}']].dropna().reset_index(drop = True)
        x = tempframe.iloc[:,0]
        y = metric_frame[f'PC{q+1}']
        coef = np.polyfit(x,y,1)
        poly1d_fn = np.poly1d(coef) 
        p_corr = tempframe.corr().loc[f'PC{q+1}', metrics[i]]
        color = color_scale.color.loc[color_scale.value == closest(list(color_scale.value), p_corr)].values[0]
        ax.scatter(x,y, color = color)
        ax.plot(x, poly1d_fn(x), 'k')
#         ax.text(0.1,0.1,str(np.around(p_corr, decimals=2)))


# xlabels = [x.replace('_','\n') for x in metric_frame.columns]
[ax.set_ylabel(PCs[i], fontsize = 35) for i, ax in enumerate(axes[:,0])];
labelz = ['Cell Volume\n(µm$^3$)','Cell Surface Area\n(µm$^2$)','Cell Major Axis\nLength (µm)','Cell Minor Axis\nLength (µm)',
          'Cell Mini Axis\nLength (µm)','Persistence\n(a.u.)','Speed\n(µm/sec)','Turn Angle (°)',
          'Cell Elongation\n(a.u.)','Front-Back Volume\nRatio (a.u.)','Right-Left Volume\nRatio (a.u.)','Top-Bottom Volume\nRatio (a.u.)',
          'Cell Sphericity\n(a.u.)','Long-Axis X-Z\nAngle (°)','Long-Axis X-Y\nAngle (°)','Long-Axis Total\nAngle (°)']
[ax.set_xlabel(labelz[i], rotation=35, horizontalalignment='center', fontsize = 20) for i, ax in enumerate(axes[-1,:])];

plt.tight_layout()
plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi=500)