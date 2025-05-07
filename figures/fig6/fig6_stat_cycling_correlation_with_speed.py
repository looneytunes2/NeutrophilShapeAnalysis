# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 14:39:51 2025

@author: Aaron
"""


import os
import numpy as np
import pandas as pd
from CustomFunctions import utils, DetailedBalance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate, signal, stats
import matplotlib

scale = 4 # scale of the plot
time_interval = 5
whichpcs = [1,7]
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_Apply/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'random/'


TotalFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col = 0)
nbins = np.max(TotalFrame[[x for x in TotalFrame.columns.to_list() if 'bin' in x]].to_numpy())
#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)




##################### average cycle frequency versus average speed ###########
fpdict = {
          'Cell_Aspect_Ratio':{'distance':8,'prominence':0.4,'width':3,'s':1},
          'LengthAlongTrajectory':{'distance':8,'prominence':5,'width':3,'s':1},
          'LengthAlongTrajectoryFront':{'distance':8,'prominence':2.5,'width':3,'s':1},
          'LengthAlongTrajectoryRear':{'distance':8,'prominence':2.5,'width':3,'s':1},
          'Cell_Volume':{'distance':8,'prominence':50,'width':5,'s':1},
          'Volume_Front_Ratio':{'distance':8,'prominence':0.08,'width':3,'s':1},
          # 'Cell_TotalAngle':{'distance':6,'prominence':5,'width':3},
          }

labelz = ['Aspect Ratio', 'Length Along Trajectory', 'Front Length Along Trajectory','Rear Length Along Trajectory',
          'Cell Volume', 'Front-Back Volume Ratio','pr','rp'  #, 'Deviation From Trajectory']
          ]



#define the colors to make the meshes
set1 = plt.cm.Set1
set2 = plt.cm.Set2
set3 = plt.cm.Set3
cmap = matplotlib.colors.ListedColormap(list(set3.colors)+[set2.colors[-2]] + [set1.colors[-1]])


scale = 4
fig, axes = plt.subplots(1, len(fpdict), figsize=(scale*len(fpdict), scale))#, sharex=True, sharey=True)
for a, f in enumerate(fpdict.keys()):
    props = fpdict[f]
    ax = axes.flatten()[a]
    alldfs = []
    leg = False #if a!=0 else True
    for c, cell in TotalFrame.groupby('CellID'):
        cell = cell.sort_values('time').reset_index(drop=True)
        #interpolate through all time point measured and unmeasured
        tck = interpolate.splrep(cell.time.values,cell[str(f)].values, k=1, s=props['s'])
        x = interpolate.splev(np.arange(5,cell.time.max(),5),tck,der=0)
        peaks, properties = signal.find_peaks(x, distance = props['distance'],
                                              prominence= props['prominence'],
                                              width= props['width'])
        #get time differences between peaks
        times = np.arange(5,cell.time.max(),5)[peaks]
        #convert to minutes
        times = times/60
        timediff = np.diff(times)
        df = pd.DataFrame({'cell':[c],
                           'freq':1/np.mean(timediff),
                           'freq_std': stats.sem(1/timediff)/np.sqrt(len(timediff)),
                           'speed':cell.speed.mean(),
                          'speed_std':cell.speed.sem()})
        alldfs.append(df)

    df = pd.concat(alldfs).dropna()
    #plot best fit line
    coef = np.polyfit(df.freq.values,df.speed.values,1)
    poly1d_fn = np.poly1d(coef) 
    p_corr, p_val = stats.pearsonr(df.freq,df.speed)
    ax.plot(df.freq, poly1d_fn(df.freq), 'k', zorder=3)
    #plot scatter
    sns.scatterplot(data = df, x='freq',y='speed',hue = 'cell', palette = cmap.colors,
                    edgecolor = '0.5', ax = ax, legend = None, zorder = 2)
    # #plot error bars
    ax.errorbar(df.freq.values, df.speed.values, xerr = df.freq_std.values, yerr= df.speed_std.values,
                color=[0.3,0.3,0.3], alpha=0.5, capsize=3, ls = 'none', zorder=1)
    # if a ==0:
    #     sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    # ax.set_title(str(f))
    x_min, x_max = ax.get_xlim()
    ax.text(x_max-(x_max-x_min)/2*1.9,0.258, 'pcorr='+'{0:.3f}'.format(p_corr)+
            '\npval='+'{0:.3f}'.format(p_val))
    
    #set axis labels
    ax.set_xlabel(labelz[a]+'\nFrequency (min⁻¹)')
    if a == 0:    
        ax.set_ylabel('Average Speed (µm/sec)')
    else:
        ax.set_ylabel('')
        
        
plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')