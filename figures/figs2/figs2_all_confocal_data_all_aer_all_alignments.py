# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:33:38 2025

@author: Aaron
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from CustomFunctions.shapePCAtools import filter_extremes_based_on_percentile
from matplotlib import cm
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

####### load common directories and data
dirlist = ['Combined_37C_Confocal_PCA_shape','Combined_37C_Confocal_PCA_s5','Combined_37C_Confocal_PCA_planar']
alignlist = ['Shape Only','Trajectory + Shape', 'Trajectory Only']
colorlist = cm.Set2.colors[:3][::-1]
ntrans = 1
time_interval = 10 #sec/frame


    
    
for d, di in enumerate(dirlist):
    
    ### get directories and constants
    basedir = Path('E:/Aaron',di)
    aerdir = basedir.joinpath('Detailed_Balance','alldatabs')
    centers = pd.read_csv(basedir.joinpath('Data_and_Figs/PC_bin_centers.csv'), index_col=0)
    binlist = centers.columns.to_list()
        
    ########### ONE BIG DIAGONAL GRAPH OF ALL PC CGPS's ##############    
    fig, axes = plt.subplots(len(binlist),len(binlist), figsize = (40,40), sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.05, .303, .012, .376])
    ## set variables to adjust axis limits
    xmin, xmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf
    axlist = []
    for ycol, bin1 in enumerate(binlist):
        for xrow, bin2 in enumerate(binlist):
            #define axis
            ax = axes[xrow, ycol]
            #get hypothetical directory for this PC pair
            dfdir = aerdir.joinpath(f'{bin1}-{bin2}_bootstrapped_{ntrans}_Area_Enclosed_Linear_Reg.csv')
        
            #Plot the data if this PC pair exists
            if dfdir.exists():
                #open data
                tempdf = pd.read_csv(dfdir, index_col = 0)
                 
                ## add cycle period not just angular velocity
                tempdf['cycle_period'] = 360/(tempdf.angular_velocity_coeff.abs()*60) ## (degrees/cycle)/((degrees/sec)*(sec/min)) = min/cycle
                
                ### filter the dataframe to avoid really extreme values
                tempdf_filtered = filter_extremes_based_on_percentile(
                    tempdf,
                    ['aer_coeff','cycle_period'],
                    0.1)
                lvl = np.arange(0.2,1.2,0.2)
                light = sns.light_palette(colorlist[d], n_colors=len(lvl))[2:int(len(lvl)-1)]
                dark = sns.dark_palette(colorlist[d], n_colors=len(lvl), reverse=True)
                cmap = LinearSegmentedColormap.from_list(alignlist[d], light+dark)
                ### plot 2d density
                ax.set_yscale("log")
                dens = sns.kdeplot(data = tempdf, x='aer_coeff', y = 'cycle_period',
                            levels = lvl,
                            cmap = cmap,
                            fill = True,
                            cbar = True,
                            cbar_ax = cbar_ax,
                            ax = ax, zorder = 1)
                
                

        
                ## plot the zero line
                ax.axvline(0, ls = '--', lw = 1, color = 'black', alpha = 0.6, zorder = 2)   
                
                ### remove legend
                ax.legend_ = None
                
                # remove upper and right box lines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
                #change tick font sizes
                ax.tick_params(labelsize = 16)
                #rotate and horizontally align x axis labels because they're long
                ax.tick_params('x', rotation = 30)
                
                # Extract extents from all contour collections
                outer = ax.get_children()[0]
                # outer = contour_sets[0].collections[0]  # lowest-density level
                for path in outer.get_paths():
                    v = path.vertices
                    xmin = min(xmin, v[:,0].min())
                    xmax = max(xmax, v[:,0].max())
                    ymax = max(ymax, v[:,1].max())
                #get the data ymin
                ymin = min(ymin, tempdf_filtered.cycle_period.min())
                #add axis array position to adjust axis limits later
                axlist.append([xrow,ycol])
                
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment('right')
                
                if ycol == 0:
                    ax.set_ylabel(bin2, fontsize = 40)
                    if xrow == range(len(binlist))[-1]:
                        ax.set_xlabel('')
        
                elif xrow == range(len(binlist))[-1]:
                    
                    ax.set_xlabel('')
                
    
            
        
            #keep the upper right plot but remove plot box
            elif (ycol==0) & (xrow==0):
                ax.set_ylabel(bin2, fontsize = 40, labelpad = 24)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
            else:
                ax.remove()
    
            
    for al in axlist:
        axes[al[0],al[1]].set_xlim(xmin,xmax)
        axes[al[0],al[1]].set_ylim(ymin,ymax)
            
                
                
    ##### add common x axis label
    fig.text(0.5, 0.075, "Area Enclosing Rate (PC units²/sec)", fontsize = 40, ha='center')
    ##### add common y axis label
    fig.text(0.083, 0.5, "Cycle Period (min/cycle)", fontsize = 40, rotation = 90, va='center')
                            
    
    # remove tick stuff from the upper right plot, but maintain the sharex sharey
    axes[0,0].tick_params(which ='both', left=False, bottom=False, labelleft=False, labelbottom=False)
    
    
    cbar = cbar_ax._colorbar
    cbar.ax.yaxis.set_ticklabels(np.round(lvl,2))
    cbar.ax.yaxis.set_tick_params(labelsize=20)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    cbar.set_label("Density Proportion", fontsize = 40, labelpad = 5, rotation=90)
    
    
    
    plt.savefig(__file__.split('.')[0]+f'_{alignlist[d]}.png', bbox_inches='tight', dpi = 500)
    
    plt.close(fig)
    


