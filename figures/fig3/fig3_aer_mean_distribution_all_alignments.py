# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:33:38 2025

@author: Aaron
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from neutrophil_shape.CustomFunctions import utils
from neutrophil_shape.CustomFunctions.shapePCAtools import filter_extremes_based_on_percentile
from neutrophil_shape.config.loader import load_config
from matplotlib.font_manager import FontProperties
from matplotlib import cm
from pathlib import Path

####### load common directories and data
config = load_config(microscope_type = 'confocal')
config._alignment = 'shape'

alignlist = ['Shape Only','Trajectory + Shape', 'Trajectory Only']
maxaerlist = [(1,2),(4,5),(2,8)]
colorlist = cm.Set2.colors[:3][::-1]
ntrans = config.db_params.ntrans
time_interval = config.im_params.time_interval



fig, ax = plt.subplots()
bigdflist = []
for i in range(len(dirlist)):
    
    #define specific directories
    basedir = Path('E:/Aaron').joinpath(dirlist[i])
    savedir = basedir.joinpath('Detailed_Balance/alldatabs')
    
    aerdf = pd.read_csv(savedir.joinpath(f'PC{maxaerlist[i][0]}-PC{maxaerlist[i][1]}_bootstrapped_{ntrans}_Area_Enclosing_Rates.csv'), index_col=0)

    #fit lines to get aer and cf
    dflist = []
    for a, t in aerdf.groupby(['Treatment','iter']):
        t = t.rename(columns = {'cumulative_time':'time'})
        t = t.sort_values('time').reset_index(drop=True)
        #fit aer
        aerresid, aercoef = utils.fit_AER(t,time_interval,'aer')

        dflist.append({'alignment':alignlist[i] + f' (PC{maxaerlist[i][0]}-PC{maxaerlist[i][1]})','iter':a[1],
                       'aerresid':aerresid,'aercoef':aercoef})
    avgdf = pd.DataFrame(dflist)


    avgdf_filtered = filter_extremes_based_on_percentile(
        avgdf,
        ['aercoef','aerresid'],
        1)

    bigdflist.append(avgdf_filtered)


bigdf = pd.concat(bigdflist, ignore_index = True)





#plot the distributions
fig, ax = plt.subplots()
sns.kdeplot(data = bigdf, x='aercoef', hue = 'alignment', common_norm = True,
            fill = True, palette = colorlist, alpha = 0.6, # cut = 0
            ax = ax)

#Change legend title and size
leg = ax.legend_
leg.set_title(title = 'Alignment Method', prop = FontProperties(size=12))

# set axis labels
ax.set_xlabel("Area Enclosing Rate (PC units²/sec)", fontsize = 16)   
ax.set_ylabel("Probability Density", fontsize = 16)  



# remove upper and right box lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)





plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)




