# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:20:32 2025

@author: Aaron
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D


def get_stars(pv):
    if pv < 0.001:
        stars = '***'
    elif pv < 0.01:
        stars = '**'
    elif pv < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'
    return stars


#define some variables
treatments = ['DMSO','Para-Nitro-Blebbistatin','CK666']
time_interval = 10 #sec/frame
whichpcs = [1,7]
vmin = -3 #lower bound for heatmap 
vmax = 3 #upper bound for heatmap 

#define some directories
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'drug/'

#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
nbins = len(centers.iloc[:,0])
#trim bins outside 2 std
bintrim = 2
nbins_trim = nbins - 2*bintrim

######## open all of the data
########### interpolate all transitions so that only individual transitions are made ###########
transdf_sep = pd.read_csv(savedir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_interpolated_transitions_separated.csv', index_col=0)
#ensure that DMSO is the first in order
transdf_sep['Treatment'] = pd.Categorical(transdf_sep.Treatment, categories=treatments, ordered=True)
transdf_sep = transdf_sep.sort_values(by='Treatment')




########### calculate the DWELL TIME DIFFERENCE of the treatments in the CGPS #############
hms = np.zeros((len(transdf_sep.Treatment.unique()), nbins_trim, nbins_trim))
countmap = np.zeros((len(transdf_sep.Treatment.unique()), nbins_trim, nbins_trim))
for i, (treat, tdf) in enumerate(transdf_sep.groupby('Treatment')):
    ################ heatmap of counts ###############
    tdf[[x for x in tdf.columns.to_list() if 'to_' in x or 'from_' in x]] = \
        tdf[[x for x in tdf.columns.to_list() if 'to_' in x or 'from_' in x]] - bintrim
    for x in range(nbins_trim):
        for y in range(nbins_trim):
            current =  tdf[(tdf['from_x'] == x+1) & (tdf['from_y'] == y+1)]
            if current.empty:
                hms[i,y,x] = 0
            else:
                hms[i,y,x] = current.time_elapsed.mean()
                #add the number of counts in this bin
                countmap[i,y,x] = len(current)



#create the difference heatmap
differences = [h-hms[0] for h in hms[1:]]
#print the results of the one-sample ttest
[print(treatments[i+1], stats.ttest_1samp(dif.flatten(), popmean=0)) for i, dif in enumerate(differences)]


#### start building the figure
fig, axes = plt.subplots(1,len(differences),figsize=(5*len(differences),5))
#single colorbar axis
cbar_ax = fig.add_axes([.95, .133, .02, .685])
for i, ax in enumerate(axes):
    
    #plot heatmap with seaborn
    sns.heatmap(
        differences[i],
        vmin=vmin,
        vmax=vmax, 
        # center=0,
        cmap=sns.diverging_palette(220, 20, n=200),
        square=True,
        xticklabels = True,
        yticklabels = True,
        ax = ax,
        cbar_ax = cbar_ax,
    )
    #correct axis orientations
    ax.invert_yaxis()
    #get rid of ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(0.5,nbins_trim+0.5)[[0,(round(nbins_trim/2)-1),-1]])
    ax.set_xticklabels([round(centers.iloc[bintrim:nbins-bintrim].PC1.iloc[x],1) for x in [0,int(round(nbins_trim/2)-1), int(nbins_trim-1)]],
                       fontsize = 14)
    ax.set_yticks(np.arange(0.5,nbins_trim+0.5)[[0,(round(nbins_trim/2)-1),-1]])
    ax.set_yticklabels([round(centers.iloc[bintrim:nbins-bintrim].PC7.iloc[x],1) for x in [0,int(round(nbins_trim/2)-1), int(nbins_trim-1)]],
                       fontsize = 14)
    #set axis titles
    ax.set_xlabel(f'PC{whichpcs[0]}', fontsize = 24)

    #set title
    ax.set_title([x[:11]+'\n'+x[11:] if x == 'Para-Nitro-Blebbistatin' else x for x in treatments][int(i+1)], fontsize = 32)

axes[0].set_ylabel(f'PC{whichpcs[1]}', fontsize = 24)

# adjust colorbar tick label size
cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(),fontsize=12)
cbar_ax.get_yaxis().labelpad = 18
cbar_ax.set_ylabel('Relative Dwell Time (sec)', fontsize = 16, rotation=270)


plt.tight_layout()

plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')








############# DOT PLOT VERSION ###################
#change treatment list so that drugs are in the correct order visually
treatments = ['DMSO','CK666','Para-Nitro-Blebbistatin']
#change the order of the differences to match
differences = differences[::-1]

diflist = []
for i, d in enumerate(differences):
    diflist.append(pd.DataFrame({'diffs':list(d.flatten()),
                             'Treatment':[treatments[int(i+1)]]*len(d.flatten()),
                             'counts':(countmap[int(i+1)].flatten())}))
diffdf = pd.concat(diflist)
    
#print the results of the one-sample ttest
sig = [stats.ttest_1samp(dif.flatten(), popmean=0)[1] for dif in differences]



#set the color palette
colorlist = ['#9c836b','#faa7a7','#faf191']
sns.set_palette(palette=colorlist)
group_colors = {treatments[1]: colorlist[2], treatments[2]: colorlist[1]}
colors = diffdf['Treatment'].map(group_colors)


#### start building the figure
fig, ax = plt.subplots(figsize = (5,3))

# Map treatment categories to x positions
treatment_to_x = {t: i for i, t in enumerate(diffdf['Treatment'].unique())}
x_positions = diffdf['Treatment'].map(treatment_to_x).astype(float)

# Apply jitter to x positions (uniform noise)
jitter_strength = 0.33  # smaller = less spread
jittered_x = x_positions + np.random.uniform(-jitter_strength, jitter_strength, size=len(diffdf))

ax.scatter(diffdf['diffs'], jittered_x, s=diffdf['counts']/3, c = colors, edgecolor = '0.4',
           alpha=0.3, zorder = 2)

#lines for the weighted means of the populations
#calculate weighted means
weightedmeans = []
for t in diffdf.Treatment.unique():
    wm = np.repeat(diffdf.loc[diffdf.Treatment==t,'diffs'].values,
              diffdf.loc[diffdf.Treatment==t,'counts'].values.astype(int))
    weightedmeans.append(np.mean(wm))
#plot weighted mean lines
ax.plot([weightedmeans[0]]*2,[-0.33,0.33], c='black')
ax.plot([weightedmeans[1]]*2,[1-0.33,1+0.33], c='black')

#line at zero
ax.axvline(0,-1,2, color = '0.7', ls = '--', zorder = 1)


#significance stars
ax.text(3, 0, get_stars(sig[0]), fontsize = 12, va = 'center', ha='center')
ax.text(3, 1, get_stars(sig[1]), fontsize = 12, va = 'center', ha='center')

#adjust plot limits to center 0
ax.set_ylim(-0.5,1.5)
ax.set_yticks([0,1])
ax.set_xlim(-3,3.05)
# ax.set_xticks([-3,0,3])

ax.set_yticklabels(treatments[1:])
ax.set_yticklabels([x[:11]+'\n'+x[11:] if x == 'Para-Nitro-Blebbistatin' else x for x in treatments[1:]], fontsize = 14,
                   rotation=90, va = 'center')

ax.set_xlabel('Relative Dwell Time (s)', fontsize = 14)


#legend sizes for the biggest and smallest dots on plot
legend_sizes = [diffdf[(diffdf.diffs>-3) & (diffdf.diffs<3)].counts.max()/3,
                diffdf[(diffdf.diffs>-3) & (diffdf.diffs<3)].counts.min()/3]
legend_labels = [str(int(s*3))+' samples' for s in legend_sizes]
handles = [
    Line2D([], [], marker='o', linestyle='None',
           markersize=np.sqrt(s), label=label, color='gray', alpha=0.6)
    for s, label in zip(legend_sizes, legend_labels)
]

ax.legend(handles=handles,
          title='Dot Size',
          bbox_to_anchor=[0.175,0.5],
          loc = 'center',
          borderpad=0.75,
          title_fontsize = 11,
          fontsize = 8,
          labelspacing=1.5,)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

plt.savefig(__file__.split('.')[0] + '_dots.png', dpi = 500, bbox_inches='tight')



