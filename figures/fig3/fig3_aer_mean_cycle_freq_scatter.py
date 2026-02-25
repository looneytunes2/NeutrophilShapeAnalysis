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
from matplotlib import cm
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LinearRegression
from pathlib import Path

####### load common directories and data
dirlist = ['Combined_37C_Confocal_PCA_shape','Combined_37C_Confocal_PCA_s5','Combined_37C_Confocal_PCA_planar']
alignlist = ['Shape Only','Trajectory + Shape', 'Trajectory Only']
colorlist = cm.Set2.colors[:3][::-1]
ntrans = 1
time_interval = 10 #sec/frame
basedir = Path('E:/Aaron',dirlist[-1])
centers = pd.read_csv(basedir.joinpath('Data_and_Figs/PC_bin_centers.csv'), index_col=0)


########### ONE BIG DIAGONAL GRAPH OF ALL PC CGPS's ##############
binlist = [i+'bin' for i in centers.columns]



collected_data = []
for ad_num, ad in enumerate(dirlist):
    #define specific directories
    basedir = Path('E:/Aaron').joinpath(ad)
    savedir = basedir.joinpath('Detailed_Balance/alldatabs')
    # make an interpolation of black values
    # f = interpolate.interp1d([0, aermax],[0,255])
    color = colorlist[ad_num]
    for ycol, a in enumerate(binlist):
        for xrow, b in enumerate(binlist):
            bin1 = a.split('bin')[0]
            bin2 = b.split('bin')[0]
            ind1 = int(bin1.split('PC')[-1])-1
            ind2 = int(bin2.split('PC')[-1])-1

            if os.path.exists(savedir.joinpath(f'{bin1}-{bin2}_bootstrapped_{ntrans}_Area_Enclosing_Rates.csv')): 
                
                aerdf = pd.read_csv(savedir.joinpath(f'{bin1}-{bin2}_bootstrapped_{ntrans}_Area_Enclosing_Rates.csv'), index_col=0)
                # print(f'Opened {bin1}-{bin2} aer files average aer mean is ',aerdf.groupby('iter').aer.mean().mean())
    
    
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
                    dflist.append({'Treatment':i[0],'iter':i[1],
                                   'aerresid':aerresid,'aercoef':aercoef,
                                   'cfresid':cfresid,'cfcoef':cfreg.coef_[0][0]})
                avgdf = pd.DataFrame(dflist)
    
    
                ### add average cycle period
                avgcf = avgdf.cfcoef.mean() #degrees/sec
                cycle_period = abs(360/avgcf/60) #minutes/cycle

                
                ### add data to overall dataframe
                meandict = {
                    'alignment': alignlist[ad_num],
                    'pc_combo': bin1+'_'+bin2,
                    'aer_mean': avgdf.aercoef.mean(),
                    'cycle_period_mean': cycle_period,
                    }
                collected_data.append(meandict)

                print(f'finised {ad} {bin1}-{bin2}')

#make a dataframe out of collected data
df = pd.DataFrame(collected_data)




#plot the scattered data
fig, ax = plt.subplots()
sns.scatterplot(data = df, x='aer_mean', y = 'cycle_period_mean', hue = 'alignment',
                palette = colorlist, alpha = 0.6, ax = ax)
#Change legend title and size
leg = ax.legend_
leg.set_title(title = 'Align Method', prop = FontProperties(size=12))

ax.set_xlabel("Mean Area Enclosing Rate (PC units²/sec)", fontsize = 16)         
ax.set_ylabel("Mean Cycle Period (min/cycle)", fontsize = 16)

# remove upper and right box lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)





plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)