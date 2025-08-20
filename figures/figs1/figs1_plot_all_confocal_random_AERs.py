# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:33:38 2025

@author: Aaron
"""


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from CustomFunctions.shapePCAtools import filter_extremes_based_on_percentile
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy import interpolate


####### load common directories and data
ntrans = 1
time_interval = 10 #sec/frame
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())

#set the max average aer for the colorbar stuff
aermax = 0.00435
### restrict data to RANDOM
treatments = ['Random']


#restrict dataframe to only random experiments
TotalFrame = FullFrame[FullFrame.Treatment=='Random'].copy()





########### ONE BIG DIAGONAL GRAPH OF ALL PC CGPS's ##############

binlist = [i for i in TotalFrame.columns.to_list() if 'bin' in i]



# make an interpolation of black values
f = interpolate.interp1d([0, aermax],[1,0])

   
fig, axes = plt.subplots(len(binlist),len(binlist), figsize = (40,40), sharex=True, sharey=True)
for ycol, a in enumerate(binlist):
    for xrow, b in enumerate(binlist):
        bin1 = a.split('bin')[0]
        bin2 = b.split('bin')[0]
        ind1 = int(bin1.split('PC')[-1])-1
        ind2 = int(bin2.split('PC')[-1])-1
        ax = axes[ind2, ind1]

        if os.path.exists(savedir+ 'allCGPS/' +f'{bin1}-{bin2}_{ntrans}_transition_Area_Enclosing_Rates.csv'):
            aerdf = pd.read_csv(savedir+ 'allCGPS/' +f'{bin1}-{bin2}_{ntrans}_transition_Area_Enclosing_Rates.csv', index_col=0)
            print(f'Opened {bin1}-{bin2} aer files average aer mean is ',aerdf.groupby('iter').aer.mean().mean())

            ### add average cycle period
            avgcf = aerdf.groupby('iter').angular_velocity.mean().mean() #degrees/sec
            cycle_period = abs(360/avgcf/60) #minutes/cycle
            ax.text(0.05,0.8, str(round(cycle_period,1))+r' ($\frac{min}{cycle}$)',
                    transform=ax.transAxes, fontsize = 20)

            
            #get average aer values
            avgaerdf = aerdf.groupby('iter').mean()
            avgaerdf = filter_extremes_based_on_percentile(
                avgaerdf,
                ['aer'],
                1)
            
            #get color based on the mean of the means
            color = str(f(abs(avgaerdf.aer.mean())))

            #plot the filled plot
            sns.kdeplot(data = avgaerdf, x='aer',
                        fill = True, color = color, alpha = 1, # cut = 0
                        ax = ax)
            #### make separate plot to change the line color
            sns.kdeplot(data = avgaerdf.aer.squeeze(),
                        fill = False, color = '0.5', lw = 3,#cut = 0
                        ax = ax)
            
            
            ax.axvline(0, ls = '--', color = 'black', alpha = 0.4)

            # remove upper and right box lines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            #change tick font sizes
            ax.tick_params(labelsize = 16)
            #rotate and horizontally align x axis labels because they're long
            ax.tick_params('x', rotation = 30)
            
            #center x axis
            ax.set_xlim(-0.01,0.01)
            
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
            
            if ind1 == 0:
                ax.set_ylabel(bin2, fontsize = 40)
                if ind2 == range(len(binlist))[-1]:
                    ax.set_xlabel('')

            elif ind2 == range(len(binlist))[-1]:
                
                ax.set_xlabel('')

            

        #keep the upper right plot but remove plot box
        elif (ind1==0) & (ind2==0):
            ax.set_ylabel(bin2, fontsize = 40, labelpad = 24)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

        else:
            print('remove this plot')
            ax.remove()
            
# remove tick stuff from the upper right plot, but maintain the sharex sharey
axes[0,0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# add colorbar data
cbar_ax = fig.add_axes([0.075, 0.3065, 0.015, 0.377]) #vertical
cbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(0, aermax), cmap='Greys'), cax=cbar_ax)
cbar.ax.yaxis.set_tick_params(labelsize=20)
cbar.ax.yaxis.set_ticks_position("left")
cbar.ax.yaxis.set_label_position("left")
cbar.set_label("Average Area Enclosing Rate (PC units²/sec)", fontsize = 40, labelpad = 7)

#     plt.tight_layout() 
# plt.subplots_adjust(wspace=0.01, hspace=0.01)



plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)