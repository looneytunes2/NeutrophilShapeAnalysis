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
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from pathlib import Path


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


treatments = ['Random','Galvanotaxis']
time_interval = 10 #sec/frame
whichpcs = [1,2]
vmin = -1.5 #lower bound for heatmap 
vmax = 1.5 #upper bound for heatmap 


#get directories and open separated datasets
basedir = Path('E:/Aaron/Combined_37C_Confocal_PCA_planar')
datadir = basedir.joinpath('Data_and_Figs')
savedir = basedir.joinpath('Detailed_Balance')

#open the centers of the binned PCs
centers = pd.read_csv(datadir.joinpath('PC_bin_centers.csv'), index_col=0)
nbins = len(centers.iloc[:,0])
#trim bins outside 2 std
bintrim = 3
nbins_trim = nbins - 2*bintrim

######## open all of the data
########### interpolate all transitions so that only individual transitions are made ###########
transdf_sep = pd.read_csv(savedir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_interpolated_transitions_separated.csv'), index_col=0)
#limit to treatments
transdf_sep = transdf_sep[transdf_sep.Treatment.isin(treatments)]
transdf_sep['Treatment'] = pd.Categorical(transdf_sep.Treatment, categories=treatments, ordered=True)
transdf_sep = transdf_sep.sort_values(by='Treatment')



############## get the counts of cells leaving 
trans_rate_df_sep = pd.read_csv(savedir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_binned_transition_rates_separated.csv'), index_col=0)
#limit to treatments
trans_rate_df_sep = trans_rate_df_sep[trans_rate_df_sep.Treatment.isin(treatments)]
trans_rate_df_sep['Treatment'] = pd.Categorical(trans_rate_df_sep.Treatment, categories=treatments, ordered=True)
trans_rate_df_sep = trans_rate_df_sep.sort_values(by='Treatment')




########### calculate the DWELL TIME DIFFERENCE of the treatments in the WHOLE CGPS #############
hms = np.zeros((len(transdf_sep.Treatment.unique()), nbins, nbins))
countmap = np.zeros((len(transdf_sep.Treatment.unique()), nbins, nbins))
for i, (treat, tdf) in enumerate(transdf_sep.groupby('Treatment')):
    for x in range(nbins):
        for y in range(nbins):
            current =  tdf[(tdf['from_x'] == x+1) & (tdf['from_y'] == y+1)]
            if current.empty:
                hms[i,y,x] = 0
            else:
                hms[i,y,x] = current.time_elapsed.mean()
                #add the number of counts in this bin
                countmap[i,y,x] = len(current)





#create the difference heatmap
differences = [h-hms[0] for h in hms[1:]][0]
#print the results of the one-sample ttest
print(stats.ttest_1samp(differences.flatten(), popmean=0))



fig, ax = plt.subplots(1,1,figsize=(5,5))
#single colorbar axis
cbar_ax = fig.add_axes([.945, .165, .05, .719])

#plot heatmap with seaborn
sns.heatmap(
    differences,
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
ax.set_xticks(np.arange(0.5,nbins+0.5)[[0,nbins//2,-1]])
ax.set_xticklabels([round(centers[f'PC{whichpcs[1]}'].iloc[x],2) for x in [0,nbins//2, int(nbins-1)]],
                   fontsize = 14)
ax.set_yticks(np.arange(0.5,nbins+0.5)[[0,nbins//2,-1]])
ax.set_yticklabels([round(centers[f'PC{whichpcs[1]}'].iloc[x],2) for x in [0,nbins//2, int(nbins-1)]],
                   fontsize = 14)


######################### vector map of probability flux ################
mdf = trans_rate_df_sep[trans_rate_df_sep.Treatment==treatments[1]]
scale = 0.0008
for x in range(1,nbins+1):
    for y in range(1,nbins+1):
        current = mdf[(mdf['x'] == x) & (mdf['y'] == y)]
        xcurrent = (current.x_plus_rate - current.x_minus_rate)/2
        ycurrent = (current.y_plus_rate - current.y_minus_rate)/2

        #add flux current arrow        
        ax.quiver(x-0.5,
                   y-0.5, 
                   xcurrent,
                   ycurrent,
                  angles = 'xy',
                  scale_units = 'xy',
                  scale = scale,
                  color = 'black',
                  alpha = 0.1,
                    zorder = 3 * 5)


#set axis titles
ax.set_xlabel(f'PC{whichpcs[0]}', fontsize = 24)

#set title
ax.set_title('Electrotaxis', fontsize = 32)

ax.set_ylabel(f'PC{whichpcs[1]}', fontsize = 24)

# adjust colorbar tick label size
cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(),fontsize=14)
cbar_ax.get_yaxis().labelpad = 22
cbar_ax.set_ylabel('Relative Dwell Time (sec)', fontsize = 20, rotation=270)


plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')







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
differences = [h-hms[0] for h in hms[1:]][0]
#print the results of the one-sample ttest
print(stats.ttest_1samp(differences.flatten(), popmean=0))



fig, ax = plt.subplots(1,1,figsize=(5,5))
#single colorbar axis
cbar_ax = fig.add_axes([.945, .165, .05, .719])

#plot heatmap with seaborn
sns.heatmap(
    differences,
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
#get rid of tick labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticks(np.arange(0.5,nbins_trim+0.5)[[0,nbins_trim//2,-1]])
ax.set_xticklabels([round(centers.PC1.iloc[x+bintrim],1) for x in [0,nbins_trim//2, int(nbins_trim-1)]],
                   fontsize = 14)
ax.set_yticks(np.arange(0.5,nbins_trim+0.5)[[0,nbins_trim//2,-1]])
ax.set_yticklabels([round(centers.PC7.iloc[x+bintrim],1) for x in [0,nbins_trim//2, int(nbins_trim-1)]],
                   fontsize = 14)
#set axis titles
ax.set_xlabel(f'PC{whichpcs[0]}', fontsize = 24)
ax.set_ylabel(f'PC{whichpcs[1]}', fontsize = 24)



#set title
ax.set_title('Electrotaxis', fontsize = 32)


# adjust colorbar tick label size
cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(),fontsize=14)
cbar_ax.get_yaxis().labelpad = 18
cbar_ax.set_ylabel('Relative Dwell Time (sec)', fontsize = 18, rotation=270)


plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '_center.png', dpi = 500, bbox_inches='tight')






############## DOT PLOT VERSION ###################


diffdf = pd.DataFrame({'diffs':list(differences.flatten()),
                         'Treatment':[treatments[1]]*len(differences.flatten()),
                         'counts':(countmap[1].flatten())})

    
#print the results of the one-sample ttest
sig = stats.ttest_1samp(differences.flatten(), popmean=0)[1]



#set the color palette
colors = ['#6cb875']*len(diffdf)


#### start building the figure
fig, ax = plt.subplots(figsize = (5,2))

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


#add ns to plot
star = get_stars(sig)
nsfs = 10 if star == 'n.s.' else 12
ax.text(weightedmeans[0],0.55, star, fontsize = nsfs, ha = 'center')


#line at zero
ax.axvline(0,-1,2, color = '0.7', ls = '--', zorder = 1)

#adjust plot limits to center 0
ax.set_ylim(-0.5,0.5)
ax.set_xlim(vmin,vmax)
# ax.set_xticks([-3,-2,-1,0,1,2,3])


ax.set_yticks([0])
ax.set_yticklabels(['Electrotaxis'], fontsize = 14, rotation=90, va = 'center')

ax.set_xlabel('Relative Dwell Time (s)', fontsize = 14)


########## legend stuff
#legend sizes for the biggest and smallest dots on plot
legend_sizes = [diffdf[(diffdf.diffs>-3) & (diffdf.diffs<3)].counts.max()/3,
                diffdf[(diffdf.diffs>-3) & (diffdf.diffs<3)].counts.min()/3]
# legend_labels = [str(int(s*3)) if int(s*3)>1 else str(int(s*3))+' sample' for s in legend_sizes]
legend_labels = [str(int(s*3)) for s in legend_sizes]
handles = [
    Line2D([], [], marker='o', linestyle='None',
           markersize=np.sqrt(s), label=label, color='gray', alpha=0.6)
    for s, label in zip(legend_sizes, legend_labels)
]

ax.legend(handles=handles,
          title='Sample Size',
          loc=[1.05,0.25],
          title_fontsize = 12,
          labelspacing=1,)



ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '_dots.png', dpi = 500, bbox_inches='tight')


