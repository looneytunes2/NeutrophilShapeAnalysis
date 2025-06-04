# -*- coding: utf-8 -*-
"""
Created on Thu May 29 16:34:16 2025

@author: Aaron
"""


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from CustomFunctions.shapePCAtools import filter_extremes_based_on_percentile
from CustomFunctions.file_management import multicsv
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy import interpolate


####### load common directories and data
time_interval = 10 #sec/frame
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col=0)
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
nbins = np.max(FullFrame[[x for x in FullFrame.columns.to_list() if 'bin' in x]].to_numpy())
scale = 0.0005 #for CGPS flux

### restrict data to RANDOM
treatments = ['Random']

savedir = basedir + 'random/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

#restrict dataframe to only random experiments
TotalFrame = FullFrame[FullFrame.Treatment=='Random'].copy()





########### ONE BIG DIAGONAL GRAPH OF ALL PC CGPS's ##############

binlist = [i for i in TotalFrame.columns.to_list() if 'bin' in i]

# Define normalization between 0 and the max aer average
norm = Normalize(vmin=0, vmax=0.02985)
# Choose a colormap (e.g., 'viridis')
cmap = cm.get_cmap('cool')

# make an interpolation of black values

# f = interpolate.interp1d([0, ],[1,0])

   
fig, axes = plt.subplots(len(binlist),len(binlist), figsize = (40,40), sharex=True, sharey=True)
for ycol, a in enumerate(binlist):
    for xrow, b in enumerate(binlist):
        bin1 = a.split('bin')[0]
        bin2 = b.split('bin')[0]
        ind1 = int(bin1.split('PC')[-1])-1
        ind2 = int(bin2.split('PC')[-1])-1

        if ind1==ind2:
            print('remove this plot')
            axes[ind2,ind1].remove()        
        else:
            ########## AER PLOT 
            aerdf = pd.read_csv(savedir+ 'allCGPS/' +f'{bin1}-{bin2}_Area_Enclosing_Rates.csv', index_col=0)
            print(f'Opened {bin1}-{bin2} aer files',aerdf.groupby('iter').mean().aer.mean())
            
            #define the axis for AER
            aerax = axes[ind2, ind1]
            
            #get average aer values
            avgaerdf = aerdf.groupby('iter').mean()
            avgaerdf = filter_extremes_based_on_percentile(
                avgaerdf,
                ['aer'],
                1)
            
            #get color based on the mean of the means
            color = cmap(norm(abs(avgaerdf.aer.mean())))
            # alph = 
            #plot the filled plot
            sns.kdeplot(data = avgaerdf, x='aer',
                        fill = True, color = color#, cut = 0
                        , ax = aerax)
            #### make separate plot to change the line color
            sns.kdeplot(data = avgaerdf.aer.squeeze(),
                        fill = False, color = '0.5'#, cut = 0
                        , ax = aerax)
            
            
            aerax.axvline(0, ls = '--', color = 'black', alpha = 0.4)

            # remove upper and right box lines
            aerax.spines['top'].set_visible(False)
            aerax.spines['right'].set_visible(False)

                
            if ind1 == 0:
                aerax.set_ylabel(bin2, fontsize = 30)
                if ind2 == range(len(binlist))[-1]:
                    aerax.set_xlabel('', fontsize = 30)

            elif ind2 == range(len(binlist))[-1]:
                
                aerax.set_xlabel('', fontsize = 30)

                
            ############## CGPS PLOT
            transdf_sep = pd.read_csv(savedir+ 'allCGPS/' +f'interpolated_{bin1}-{bin2}_transitions_separated.csv', index_col=0)
            trans_rate_df_sep = pd.read_csv(savedir+ 'allCGPS/' +f'{bin1}-{bin2}_binned_transition_rates_separated.csv', index_col=0)
            print(f'Opened {bin1}-{bin2} transition rate files')


            cgpsax = axes[ind1,ind2]
            ########### PDFs AND PROBABILITY FLUX OF THE SEPARATED MIGRATION MODES #############

            ################ heatmap of probability density #############
            ttot = transdf_sep.time_elapsed.sum()
            #make numpy array with heatmap data
            bighm = np.zeros((nbins,nbins))
            #get total time observed in the system

            for x in range(nbins):
                for y in range(nbins):
                    current =  transdf_sep[(transdf_sep['from_x'] == x+1) & (transdf_sep['from_y'] == y+1)]
                    if current.empty:
                        bighm[y,x] = 0
                    else:
                        bighm[y,x] = current.time_elapsed.sum()/ttot
            #plot heatmap with seaborn
            sns.heatmap(
                bighm,
                vmin=0, vmax=0.045, #center=0,
                cmap='rocket',
                square=True,
                xticklabels = True,
                yticklabels = True,
                ax = cgpsax,
                cbar=False,
#                     cbar_ax = None if i else cbar_ax,
        #         cbar_kws=cbar_kws
            )

            ######################### vector map of probability flux ################
            for x in range(1,nbins+1):
                for y in range(1,nbins+1):
                    current = trans_rate_df_sep[(trans_rate_df_sep['x'] == x) & (trans_rate_df_sep['y'] == y)]
                    xcurrent = (current.x_plus_rate - current.x_minus_rate)/2
                    ycurrent = (current.y_plus_rate - current.y_minus_rate)/2
                    cgpsax.quiver(x-0.5,
                               y-0.5, 
                               xcurrent,
                               ycurrent,
                              angles = 'xy',
                              scale_units = 'xy',
                              scale = scale,
        #                       width = 0.012,
        #                       minlength = 0.8,
                              color = 'white')


            # axis label stuff

            #set limits
            cgpsax.set_xlim(0,nbins+1)
            cgpsax.set_ylim(0,nbins+1)



            if a == binlist[0]:
                cgpsax.set_title(bin2, fontsize = 30)
            if ycol == xrow-1:
                cgpsax.set_ylabel('', fontsize = 30)
            
                cgpsax.set_xticks(np.arange(0.5,nbins+0.5),[round(x,1) for x in centers[bin1].to_list()])
                cgpsax.set_xticklabels(cgpsax.get_xticklabels(), fontsize = 11)
                cgpsax.set_yticks(np.arange(0.5,nbins+0.5),[round(x,1) for x in centers[bin2].to_list()])
                cgpsax.set_yticklabels(cgpsax.get_yticklabels(), fontsize = 11)
            else:
                cgpsax.set_xticks([])
                cgpsax.set_xticklabels([])
                cgpsax.set_yticks([])
                cgpsax.set_yticklabels([])
             



#     plt.tight_layout() 
# plt.subplots_adjust(wspace=0.01, hspace=0.01)



plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)