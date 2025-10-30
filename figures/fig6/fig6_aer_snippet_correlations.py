# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 16:51:29 2025

@author: Aaron
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CustomFunctions import utils
from scipy import interpolate, stats
import seaborn as sns
import matplotlib
from sklearn.linear_model import LinearRegression

whichpcs = [1,7]
time_interval = 5
ntrans = 1
mind = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_Apply/'
datadir = mind + 'Data_and_Figs/'
randir = mind + 'random/'
moviedir = 'E:/Aaron/random_lls/singlecells/'


FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col = 0)
# open aers
allaers = pd.read_csv(randir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_raw_transition_aer_cf.csv', index_col = 0)

#merge aer and cf info
TotalFrame = FullFrame.merge(allaers[['aer','angular_velocity','cell']],left_on='cell',right_on='cell')
TotalFrame = TotalFrame.sort_values(['CellID','time'])



###### label snippets of the specified length and get snippet measurements
runlengthlist = [13,25,37] #number of frames in each snippet
aerrunlist = [] #list to append different snippet IDs to
snippetmetrics = [] #list to append snippet metrics to
for rl in runlengthlist:
    srcount = 0
    label = f'aerrun_{(rl-1)*time_interval}'
    # rl = 25 #run length in number of frames
    aertime = np.arange(1,rl)*time_interval
    cellstateruns = []

    for i, cell in TotalFrame.groupby('CellID'):
        ##### identify consecutive runs of different aer states
        cs, runs = utils.get_consecutive_timepoints(cell, 'time', time_interval)
        for r in runs:
            c = cs.iloc[r].copy()
            allshifts = np.arange(0,len(c),rl)
            stateruns = []
            for n in range(len(allshifts)):
                #treat all equally sized snippets normally
                #the last snippet will almost never be of the right length
                if n!= len(allshifts)-1:
                    #get snippet
                    tempc = c.iloc[allshifts[n]:allshifts[n+1]].copy()
                    tempc.loc[:,label] = srcount
                    cellstateruns.append(tempc[['cell',label]])
                    #measure snippet metrics
                    persistence = tempc.persistence.mean()
                    speed = tempc.speed.mean()
                    euclid = np.sqrt((tempc.iloc[-1].x - tempc.iloc[0].x)**2+
                                    (tempc.iloc[-1].y - tempc.iloc[0].y)**2+
                                    (tempc.iloc[-1].z - tempc.iloc[0].z)**2)
                    #fit aer
                    
                    aerreg = LinearRegression().fit(aertime.reshape(-1, 1),
                                                    tempc.aer[1:].cumsum().values.reshape(-1, 1))
                    aerresid = aerreg.score(aertime.reshape(-1, 1),
                                           tempc.aer[1:].cumsum().values.reshape(-1, 1))
                    
                    snippetmetrics.append({
                        'CellID':tempc.iloc[0].CellID,
                        'speed':speed,
                        'persistence':persistence,
                        'euclid':euclid,
                        'aercoef':aerreg.coef_[0][0],
                        'aerresid':aerresid,
                        'aerrun':label+'_'+str(srcount)})
                    
                    srcount = srcount + 1
                else:
                    tempc = c.iloc[allshifts[n]:len(c)].copy()
                    tempc.loc[:,label] = np.nan
                    cellstateruns.append(tempc[['cell',label]])
    aerrunlist.append(pd.concat(cellstateruns))
    
    
#combine and merge with other data
aerrunframe = TotalFrame.copy()
for a in aerrunlist:
    aerrunframe = aerrunframe.merge(a, on='cell')

#combine snippet metrics
snippetframe = pd.DataFrame(snippetmetrics)





#metrics to plot
minute_labels = [str(int((x-1)*time_interval/60))+[' minute',' minutes',' minutes'][i] for i,x in enumerate(runlengthlist)]

#define the colors for the lls cell IDs
set1 = plt.cm.Set1
set2 = plt.cm.Set2
set3 = plt.cm.Set3
cmap = matplotlib.colors.ListedColormap(list(set3.colors)+[set2.colors[-2]] + [set1.colors[-1]])



metrics = ['speed', 'persistence']
fig, axes = plt.subplots(len(metrics), len(runlengthlist), figsize = (3*len(runlengthlist),6), sharex = True) #

for mm, met in enumerate(metrics):
    for r, rl in enumerate(runlengthlist):
        ax = axes[mm,r]
        tempsnip = snippetframe[[f'aerrun_{(rl-1)*time_interval}' in x for x in snippetframe.aerrun]]
        y = tempsnip[met].values
        x = tempsnip['aercoef'].values
        #linear regression
        aerreg = LinearRegression().fit(x.reshape(-1, 1),y.reshape(-1, 1))
        m = aerreg.coef_[0][0]
        b = aerreg.intercept_[0]
        #scatterplot
        sns.scatterplot(data = tempsnip, x = 'aercoef', y = met, hue = 'CellID', palette = cmap.colors, linewidth = 0,
                        edgecolor=None, ax = ax)
       
        #set y limits the same for each metric
        if met == 'speed':
            ax.set_ylim(0.024,0.560)
        elif met == 'persistence':
            ax.set_ylim(0.168,1.074)
        #regression line
        xmin,xmax = ax.get_xlim()
        fp = m*xmin + b
        lp = m*xmax + b
        ax.plot([xmin,xmax],[fp,lp], c = 'black')
        #pearson coef label
        p_corr, pval = stats.pearsonr(x,y)
        txx = 0.0118
        txy = 0.18 if met == 'speed' else 0.5
        ax.text(txx,txy,str(np.around(p_corr, decimals=2)))
        
        ####plot visuals
        #yaxis labels
        if met == 'speed' and r == 0:
            ax.set_ylabel('Mean Speed (µm/s)', fontsize = 18)
        elif met == 'persistence' and r == 0:
            ax.set_ylabel('Mean Persistence', fontsize = 18)
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        #xaxis labels
        if mm == len(metrics)-1 and r == len(runlengthlist)//2:
            ax.set_xlabel('Area Enclosing Rate (PC units²/sec)', fontsize = 22)
        else:
            ax.set_xlabel('')
            
        #titles
        if met == metrics[0]:
            tmin = int((rl-1)*time_interval/60)
            justmin = ' minute' if tmin == 1 else ' minutes'
            ax.set_title(str(tmin)+justmin, fontsize = 18)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
        ax.legend_ = None

plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')

        

    