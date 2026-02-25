# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:33:38 2025

@author: Aaron
"""


import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from CustomFunctions import utils
from CustomFunctions.shapePCAtools import filter_extremes_based_on_percentile
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from pathlib import Path
from matplotlib.patches import Patch

####### load common directories and data
dirlist = ['Combined_37C_Confocal_PCA_shape','Combined_37C_Confocal_PCA_s5','Combined_37C_Confocal_PCA_planar']
alignlist = ['Shape Only','Trajectory + Shape', 'Trajectory Only']
colorlist = cm.Set2.colors[:3][::-1]
ntrans = 1
time_interval = 10 #sec/frame
checkdir = Path('E:/Aaron',dirlist[-1])
centers = pd.read_csv(checkdir.joinpath('Data_and_Figs/PC_bin_centers.csv'), index_col=0)


########### ONE BIG DIAGONAL GRAPH OF ALL PC CGPS's ##############
binlist = [i+'bin' for i in centers.columns]


   
fig, axes = plt.subplots(len(binlist),len(binlist), figsize = (40,40), sharex=True, sharey=True)



for ycol, a in enumerate(binlist):
    for xrow, b in enumerate(binlist):
        
        bin1 = a.split('bin')[0]
        bin2 = b.split('bin')[0]
        ind1 = int(bin1.split('PC')[-1])-1
        ind2 = int(bin2.split('PC')[-1])-1
        ax = axes[ind2, ind1]
        bigdflist = []


        currentcheckdir = checkdir.joinpath('Detailed_Balance/alldatabs')
        if os.path.exists(currentcheckdir.joinpath(f'{bin1}-{bin2}_bootstrapped_{ntrans}_Area_Enclosing_Rates.csv')): 
            for ad_num, ad in enumerate(dirlist):
                #define specific directories
                basedir = Path('E:/Aaron').joinpath(ad)
                savedir = basedir.joinpath('Detailed_Balance/alldatabs')
                color = colorlist[ad_num]

                aerdf = pd.read_csv(savedir.joinpath(f'{bin1}-{bin2}_bootstrapped_{ntrans}_Area_Enclosing_Rates.csv'), index_col=0)

                #fit lines to get aer and cf
                dflist = []
                for i, t in aerdf.groupby(['Treatment','iter']):
                    t = t.rename(columns = {'cumulative_time':'time'})
                    t = t.sort_values('time').reset_index(drop=True)
                    #fit aer
                    aerresid, aercoef = utils.fit_AER(t,time_interval,'aer')
                    
                    ###### fit angular velocity
                    ##add angle
                    t['angle'] = t.angular_velocity*time_interval
                    ## fit a line to av over time
                    cfreg = LinearRegression().fit(t.time.values.reshape(-1, 1),
                                                   t.angle.cumsum().values.reshape(-1, 1))
                    cfresid = cfreg.score(t.time.values.reshape(-1, 1),
                                            t.angle.cumsum().values.reshape(-1, 1))
                    dflist.append({'Alignment Method':alignlist[ad_num],'iter':i[1],
                                   'aerresid':aerresid,'aercoef':aercoef,
                                   'cfresid':cfresid,'cfcoef':cfreg.coef_[0][0]})
                avgdf = pd.DataFrame(dflist)
    
    
                avgdf_filtered = filter_extremes_based_on_percentile(
                    avgdf,
                    ['aercoef','aerresid'],
                    1)
    
                print(f'Opened {bin1}-{bin2} aer files average aer mean is ',avgdf.aercoef.mean())
    
    
                ### add average cycle period
                avgcf = aerdf.groupby('iter').angular_velocity.mean().mean() #degrees/sec
                cycle_period = abs(360/avgcf/60) #minutes/cycle
                if (bin1 =='PC1') and (bin2 == 'PC2') and (ad_num == len(dirlist)-1):
                    cyclestring = str(round(cycle_period,1))+r' ($\frac{min}{cycle}$)'
                else:
                    cyclestring = str(round(cycle_period,1))
                ax.text(0.45,[0.6,0.7,0.8][ad_num], cyclestring, transform=ax.transAxes,
                        fontsize = 20, color = colorlist[ad_num])
    

                ### append current alignment dataframe
                bigdflist.append(avgdf_filtered)
             
            ### combine data
            allavgdf = pd.concat(bigdflist, ignore_index = True)
            
            #plot the KDEs
            sns.kdeplot(data = allavgdf, x='aercoef', hue = 'Alignment Method',
                        common_norm = True, fill = True, palette = colorlist,
                        alpha = 0.6, ax = ax, zorder = 2)
    
            ## plot the zero line
            ax.axvline(0, ls = '--', lw = 0.5, color = 'black', alpha = 0.5, zorder = 1)   
            
            ### remove legend
            ax.legend_ = None
            
            # remove upper and right box lines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
            #change tick font sizes
            ax.tick_params(labelsize = 16)
            #rotate and horizontally align x axis labels because they're long
            ax.tick_params('x', rotation = 30)
            
            #center x axis
            # ax.set_xlim(-0.01,0.01)
            
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

            
        
        
            
            
##### add common x axis label
fig.text(0.5, 0.06, "Area Enclosing Rate (PC units²/sec)", fontsize = 40, ha='center')
##### add common x axis label
fig.text(0.5, 0.08, "Probability Density", fontsize = 40, ha='center')
                        

# remove tick stuff from the upper right plot, but maintain the sharex sharey
axes[0,0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)




plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)







########## MAKE A LEGEND FOR THE BIG MATRIX

fig, ax = plt.subplots()
leghands = [Patch(color=colorlist[i], label=a) for i, a in enumerate(alignlist)]
ax.legend(handles=leghands, title = 'Alignment Method', title_fontsize = 14,
          loc = 'center')

#get rid of axis framing
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


plt.tight_layout()



plt.savefig(__file__.split('.')[0]+'_legend.png', bbox_inches='tight', dpi = 500)







