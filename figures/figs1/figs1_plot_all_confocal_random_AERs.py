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


####### load common directories and data
time_interval = 10 #sec/frame
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())


### restrict data to RANDOM
treatments = ['Random']

savedir = basedir + 'random/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

#restrict dataframe to only random experiments
TotalFrame = FullFrame[FullFrame.Treatment=='Random'].copy()





########### ONE BIG DIAGONAL GRAPH OF ALL PC CGPS's ##############

binlist = [i for i in TotalFrame.columns.to_list() if 'bin' in i]

# Define normalization between 0 and 0.234
norm = Normalize(vmin=0, vmax=0.02985)
# Choose a colormap (e.g., 'viridis')
cmap = cm.get_cmap('cool')


   
fig, axes = plt.subplots(len(binlist),len(binlist), figsize = (40,40), sharex=True, sharey=True)
for ycol, a in enumerate(binlist):
    for xrow, b in enumerate(binlist):
        bin1 = a.split('bin')[0]
        bin2 = b.split('bin')[0]
        ind1 = int(bin1.split('PC')[-1])-1
        ind2 = int(bin2.split('PC')[-1])-1
        ax = axes[ind2, ind1]

        if os.path.exists(savedir+ 'allCGPS/' +f'{bin1}-{bin2}_Area_Enclosing_Rates.csv'):
            aerdf = pd.read_csv(savedir+ 'allCGPS/' +f'{bin1}-{bin2}_Area_Enclosing_Rates.csv', index_col=0)
            print(f'Opened {bin1}-{bin2} aer files')
            

            #get average aer values
            avgaerdf = aerdf.groupby('iter').mean()
            avgaerdf = filter_extremes_based_on_percentile(
                avgaerdf,
                ['aer'],
                1)
            
            #get color based on the mean of the means
            color = cmap(norm(abs(avgaerdf.aer.mean())))
            #plot the filled plot
            sns.kdeplot(data = avgaerdf, x='aer',
                        fill = True, color = color#, cut = 0
                        , ax = ax)
            #### make separate plot to change the line color
            sns.kdeplot(data = avgaerdf.aer.squeeze(),
                        fill = False, color = '0.5'#, cut = 0
                        , ax = ax)
            
            
            ax.axvline(0, ls = '--', color = 'black', alpha = 0.4)

            # axis label stuff

            # #set limits
            # ax.set_xlim(0,nbins+1)
            # ax.set_ylim(0,nbins+1)




                
            if ind1 == 0:
                ax.set_ylabel(bin2, fontsize = 30)
                if ind2 == range(len(binlist))[-1]:
                    ax.set_xlabel('', fontsize = 30)

            elif ind2 == range(len(binlist))[-1]:
                
                ax.set_xlabel('', fontsize = 30)

                
            # else:
            #     # ax.set_xticks([])
            #     ax.set_xticklabels([])
            #     ax.set_xlabel('')
            #     # ax.set_yticks([])
            #     ax.set_yticklabels([])
            #     ax.set_ylabel('')
            #     print('something else')
             
            # if ind2 == range(len(binlist))[-1]:
                
            # else:
            
            # ax.set_aspect('equal','box')
             
        else:
            print('remove this plot')
            ax.remove()


#     plt.tight_layout() 
# plt.subplots_adjust(wspace=0.01, hspace=0.01)



plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)