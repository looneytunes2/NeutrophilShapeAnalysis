# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:21:49 2025

@author: Aaron
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, FancyArrow
import seaborn as sns
from pathlib import Path
from CustomFunctions import DetailedBalance


####### load common directories and data
time_interval = 10 #sec/frame
ntrans = 1
# inverse scale for arrows
scale = 0.0008
alignlist = ['Shape Only','Trajectory + Shape', 'Trajectory Only']


#manually define origins for all of the CGPSs
allallorigins = [
#### SHAPE alignment origins
[[[8,8],[8,8],[8,8],[8,8],[8,7],[8,8],[8,8]],
    [[8,8],[8,7],[8,8],[8,8],[8,8],[8,8]],
        [[7,8],[6,8],[8,8],[8,8],[7,9]],
            [[8,8],[8,8],[8,8],[8,8]],
                [[8,8],[8,8],[8,8]],
                    [[8,8],[8,8]],
                        [[8,8]]],


#### WIDTH alignment origins
[[[8,8],[8,8],[9,8],[8,8],[9,7],[9,8],[9,8]],
    [[8,8],[8,8],[8,8],[8,8],[8,8],[8,8]],
        [[8,8],[8,8],[8,8],[8,8],[8,8]],
            [[8,8],[8,8],[8,8],[8,8]],
                [[8,8],[8,8],[8,8]],
                    [[6,8],[8,8]],
                        [[8,8]]],

#### PLANAR alignment origins
[[[7,8],[8,6],[8,7],[8,8],[8,8],[8,8],[9,8]],
    [[8,6],[8,7],[8,8],[8,7],[8,8],[8,8]],
        [[8,8],[7,8],[8,8],[8,8],[9,8]],
            [[7,8],[9,8],[8,8],[8,8]],
                [[8,8],[8,8],[9,8]],
                    [[8,7],[8,8]],
                        [[8,8]]]
]



    
for d, di in enumerate(['Combined_37C_Confocal_PCA_shape',
                     'Combined_37C_Confocal_PCA_s5',
                     'Combined_37C_Confocal_PCA_planar']):

    basedir = Path('E:/Aaron',di)
    datadir = basedir.joinpath('Data_and_Figs')
    savedir = basedir.joinpath('Detailed_Balance')
    centers = pd.read_csv(datadir.joinpath('PC_bin_centers.csv'), index_col=0)
    nbins = centers.shape[0]
    binlist = centers.columns.to_list()

    allorigins = allallorigins[d]

    ########### ONE BIG DIAGONAL GRAPH OF ALL PC CGPS's ##############
    fig, axes = plt.subplots(len(binlist),len(binlist), figsize = (40,40))
    
    #single colorbar axis
    cbar_ax = fig.add_axes([.91, .303, .012, .376])
    for xrow, bin1 in enumerate(binlist):
        for ycol, bin2 in enumerate(binlist):

                
            ax = axes[int(bin1.split('PC')[-1])-1,int(bin2.split('PC')[-1])-1]
    
            if savedir.joinpath(f'{bin1}-{bin2}_binned_transition_rates_separated.csv').exists():
                transdf_sep = pd.read_csv(savedir.joinpath(f'{bin1}-{bin2}_interpolated_transitions_separated.csv'), index_col=0)
                bsfield_sep = pd.read_csv(savedir.joinpath('alldatabs', f'{bin1}-{bin2}_bootstrapped_{ntrans}_transitions_average_currents.csv'), index_col=0)
                print(f'Opened {bin1}-{bin2} transition rate files')
                
    
                #### calculate the overall average transition rates
                #unify all treatments
                transdf_sep.loc[:,'Treatment'] = 'alldata'
                #get total time observed in the system
                ttot = transdf_sep.time_elapsed.sum()
                ratesargs = (transdf_sep, nbins, ttot)
                trans_rate_df_sep = DetailedBalance.transition_count_wrapper(ratesargs) 
    
        
                
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
                    ax = ax,
                    cbar=False,
                    zorder = 1
                )
    
    
                
                ######################### vector map of probability flux ################
                #combine relevant data
                elldf = trans_rate_df_sep.merge(bsfield_sep, on = ['x','y'])
                for x in range(1,nbins+1):
                    for y in range(1,nbins+1):
                        current = elldf[(elldf['x'] == x) & (elldf['y'] == y)]
                        xcurrent = (current.x_plus_rate - current.x_minus_rate)/2
                        ycurrent = (current.y_plus_rate - current.y_minus_rate)/2
                        ax.quiver(x-0.5,
                                    y-0.5, 
                                    xcurrent,
                                    ycurrent,
                                  angles = 'xy',
                                  scale_units = 'xy',
                                  scale = scale,
                                  color = 'white',
                                  zorder = 3)
    
    
                        #determine ellipse width, height and angle
                        #always set eval1 to width and adjust angle accordingly
                        eh = np.sqrt(abs(current.eval2))*(2/scale)
                        ew = np.sqrt(abs(current.eval1))*(2/scale)
                        evec = current[['evec1x','evec1y']].values[0]
                        evec = evec if evec[1]>0 else -evec
                        eang = np.degrees(np.arctan2(evec[1],evec[0]))
                
                        ell = Ellipse(xy=(x-0.5+(xcurrent.values*(1/scale)),y-0.5+(ycurrent.values*(1/scale))),
                                      width=ew,
                                      height=eh,
                                      angle=eang,
                                      color = 'lightblue',
                                      alpha = 0.12,
                                      zorder = 2)
                        ax.add_artist(ell)
                
    
    
                # axis label stuff
    
                #set limits
                ax.set_xlim(0,nbins+1)
                ax.set_ylim(0,nbins+1)
    
    
                ##### draw a little blue dot for the origin
                # if ycol!= 0:
                    
                orig = allorigins[int(xrow)][int(ycol-(1+xrow))]
                ax.scatter(orig[0]-0.5, orig[1]-0.5, s = 90, color = '#11bd20', zorder=2)
    
    
                if bin1 == binlist[0]:
                    ax.set_title(bin2, fontsize = 40)
                    if bin2!=binlist[-1]:
                        ax.set_yticks([])
                        ax.set_yticklabels([])
                    else:
                        ax.tick_params(left=False, labelleft = False, right = True, labelright=True, labelsize = 16)
                        ax.set_yticks(np.arange(0.5,nbins+0.5)[[0,(round(nbins/2)-1),-1]],
                                     [round(centers[bin1].iloc[x],1) for x in [0,int(round(nbins/2)-1), int(nbins-1)]])
                        ax.spines['right'].set_position(('outward', -24))
                    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, labelsize = 16)
                    ax.set_xticks(np.arange(0.5,nbins+0.5)[[0,(round(nbins/2)-1),-1]],
                                 [round(centers[bin2].iloc[x],1) for x in [0,int(round(nbins/2)-1), int(nbins-1)]])
    
                    ax.spines['top'].set_position(('outward', -24))
    
                elif bin2 == binlist[-1]:
                    ax.set_xticks([])
                    ax.set_xticklabels([])
                    ax.tick_params(left=False, labelleft = False, right = True, labelright=True, labelsize = 16)
                    ax.set_yticks(np.arange(0.5,nbins+0.5)[[0,(round(nbins/2)-1),-1]],
                                 [round(centers[bin1].iloc[x],1) for x in [0,int(round(nbins/2)-1), int(nbins-1)]])
                
                    ax.spines['right'].set_position(('outward', -24))
                    
                else:
                    ax.set_xticks([])
                    ax.set_xticklabels([])
                    ax.set_yticks([])
                    ax.set_yticklabels([])
    
            elif (bin1=='PC1') & (bin2=='PC1'):
                ax.set_title(bin2, fontsize = 40)#, pad=24)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
            else:
                print('remove this plot')
                ax.remove()
    
    
    ### add flux vector scale bar
    #legend background
    lxp = 0.35
    lyp = 0.35
    legh = 1.9
    legw = 8.2
    rect = Rectangle((lxp, lyp), legw, legh, linewidth=1, edgecolor='black', facecolor='#80858a')
    axes[0,1].add_patch(rect)
    rect.set_zorder(4 * 5)
    scalevalue = 0.0017
    #x-axis legend arrow
    arrxp = lxp + legw/2 - (scalevalue/scale)/2 
    axes[0,1].quiver(arrxp,lyp+1.55,scalevalue,0,angles = 'xy',scale_units = 'xy',scale = scale,color = "white",zorder = 4 * 5)
    #x-axis legend text
    xsc = f'{scalevalue:.1e}'
    xsc = xsc.split('e')[0] + 'x10$^{' +  str(int(xsc.split('e')[1])) + '}$'
    axes[0,1].text(lxp+0.1,lyp+0.25,xsc+' $s^{-1}$', color = 'white', fontsize = 20, fontweight = 'bold',zorder = 4 * 5)
    
    
    #remove box around upper right plot
    axes[0,0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    #colorbar stuff
    cbar = fig.colorbar(axes[0, 1].collections[0], cax=cbar_ax)
    cbar.ax.yaxis.set_tick_params(labelsize=20)
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")
    cbar.set_label("Probability", fontsize = 40, labelpad = 36, rotation=-90)
    #     plt.tight_layout() 
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    
    
    plt.savefig(__file__.split('.')[0]+f'_{alignlist[d]}.png', bbox_inches='tight', dpi = 500)



###### now make a separate figure to make the legend
fig, ax = plt.subplots()

ax.patch.set_facecolor(sns.color_palette('rocket')[0])

#arrow marker
arrow = FancyArrow(0.25, -1, -0.5, 0, width=0.15, head_length = 0.1,
                   facecolor='white', edgecolor = None, label='Arrow')
ax.add_artist(arrow)

#error oval marker
ell = Ellipse(xy=(0,-2),
              width=0.6,
              height=0.4,
              color = 'lightblue',
              alpha = 0.12,
              )
ax.add_artist(ell)

# origin marker
ax.scatter(0, -3, s = 300, color = '#11bd20')

#### all the labels
## arrow label
ax.text(0.4,-1.1,'Average flux', c = 'white', fontsize = 24)
## error label
ax.text(0.4,-2.1,'Error estimation', c = 'white', fontsize = 24)
## origin label
ax.text(0.4,-3.1,'Flux origin', c = 'white', fontsize = 24)

#adjust plot limits
ax.set_xlim(-0.42,1.6)
ax.set_ylim(-3.5,-0.5)

#legend title
fig.suptitle('CGPS legend', fontsize = 32)

#get rid of axis framing
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


plt.savefig(__file__.split('.')[0]+'_legend.png', bbox_inches='tight', dpi = 500)


