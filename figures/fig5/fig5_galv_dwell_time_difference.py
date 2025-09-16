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



treatments = ['Random','Galvanotaxis']
time_interval = 10 #sec/frame
whichpcs = [1,7]
origin = [9,9] #origin of flux in the full-sized CGPS
vmin = -1.5 #lower bound for heatmap 
vmax = 1.5 #upper bound for heatmap 


#get directories and open separated datasets
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
savedir = [basedir + 'galv/', basedir + 'random/']

#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
nbins = len(centers.iloc[:,0])
#trim bins outside 2 std
bintrim = 3
nbins_trim = nbins - 2*bintrim

######## open all of the data
########### interpolate all transitions so that only individual transitions are made ###########
translist = []
for s in savedir:
    translist.append(pd.read_csv(s+f'PC{whichpcs[0]}-PC{whichpcs[1]}_interpolated_transitions_separated.csv', index_col=0))
transdf_sep = pd.concat(translist, ignore_index=True)
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



plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')





####### Get the quadrant info
ori_adj = np.array(origin) - bintrim

d = differences.copy()
c = countmap[1]
orix = ori_adj[0]
oriy = ori_adj[1]

#empty dataframe
quad = []
quad.append({'quadrant':['upperleft']*(len(d[oriy:,:orix].flatten())),
              'diffs': d[oriy:,:orix].flatten(),
              'counts': c[oriy:,:orix].flatten()})
quad.append({'quadrant':['upperright']*(len(d[oriy:,orix:].flatten())),
              'diffs': d[oriy:,orix:].flatten(),
              'counts': c[oriy:,orix:].flatten()})
quad.append({'quadrant':['lowerright']*(len(d[:oriy,orix:].flatten())),
              'diffs': d[:oriy,orix:].flatten(),
              'counts': c[:oriy,orix:].flatten()})
quad.append({'quadrant':['lowerleft']*(len(d[:oriy,:orix].flatten())),
              'diffs': d[:oriy,:orix].flatten(),
              'counts': c[:oriy,:orix].flatten()})
quaddf = pd.concat([pd.DataFrame(x) for x in quad])

print(stats.f_oneway(*[q.diffs.to_list() for _, q in quaddf.groupby('quadrant')]))
tukey = pairwise_tukeyhsd(endog=quaddf.diffs.values, groups=quaddf.quadrant.values, alpha=0.05)
tukey_df = pd.DataFrame(
    data=tukey._results_table.data[1:], 
    columns=tukey._results_table.data[0] 
)


# #### start building the figure
# fig, ax = plt.subplots(1,1,figsize=(3,5))
# sns.stripplot(data = quaddf, x = 'quadrant', y = 'diffs', ax = ax)

# #set title
# ax.set_title([x[:11]+'\n'+x[11:] if x == 'Para-Nitro-Blebbistatin' else x for x in treatments][int(i+1)])#, fontsize = 32)


# #plot significance stars
# ticklabels = [x.get_text() for x in ax.xaxis.get_ticklabels()]
# slv = 0 #star level
# ymin, ymax = ax.get_ylim()
# for r, row in tukey_df[tukey_df.reject].iterrows():
#     print('star')
#     pstar = get_stars(row['p-adj'])
#     xp = np.sort(np.array([ticklabels.index(row.group1),ticklabels.index(row.group2)]))
#     starinc = (ymax-ymin)*0.02 if pstar == 'n.s.' else (ymax-ymin)*0.001
#     barinc = (ymax-ymin)*0.08
#     #star
#     nsfs = 10 if pstar=='n.s.' else 12
#     # ax.text(xp.mean(), ymax+starinc, pstar, fontsize = nsfs, ha='center')
#     ax.text(xp.mean(), ymax+(barinc*slv), pstar, fontsize = nsfs, ha='center')
#     #bar
#     ax.plot([xp[0]+0.1,xp[1]-0.1], [ymax+(barinc*slv),ymax+(barinc*slv)], color = 'black')

#     slv = slv+1

# plt.tight_layout()

# plt.savefig(__file__.split('.')[0] + '_quadrants.png', dpi = 500, bbox_inches='tight')





############## DOT PLOT VERSION ###################


diffdf = pd.DataFrame({'diffs':list(differences.flatten()),
                         'Treatment':[treatments[1]]*len(differences.flatten()),
                         'counts':(countmap[1].flatten())})

    
#print the results of the one-sample ttest
sig = stats.ttest_1samp(differences.flatten(), popmean=0)[1]



#set the color palette
colors = ['#6cc46d']*len(diffdf)


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
ax.text(0,0.55, 'n.s.', fontsize = 10, ha = 'center')


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
legend_labels = [str(int(s*3))+' samples' if int(s*3)>1 else str(int(s*3))+' sample' for s in legend_sizes]
handles = [
    Line2D([], [], marker='o', linestyle='None',
           markersize=np.sqrt(s), label=label, color='gray', alpha=0.6)
    for s, label in zip(legend_sizes, legend_labels)
]

ax.legend(handles=handles,
          title='Dot Size',
          loc=[1.05,0.25],
          title_fontsize = 12,
          labelspacing=1,)



ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '_dots.png', dpi = 500, bbox_inches='tight')


