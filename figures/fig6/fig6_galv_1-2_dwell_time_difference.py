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
from neutrophil_shape.config.loader import load_config
import scikit_posthocs


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
whichpcs = (1,2)
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'
pc_combos = config.common.pc_combos
nbins = config.db_params.nbins
origin = config.db_params.origins[pc_combos.index(whichpcs)]
vmin = -1.5 #lower bound for heatmap 
vmax = 1.5 #upper bound for heatmap 


#define some directories
savedir = config.common.savedir
datadir = savedir.joinpath('shape_data')
dbdir = savedir.joinpath('detailed_balance')

#open the centers of the binned PCs
centers = pd.read_csv(datadir.joinpath('PC_bin_centers.csv'), index_col=0)
#trim bins outside 2 std
bintrim = 3
nbins_trim = nbins - 2*bintrim

######## open all of the data
########### interpolate all transitions so that only individual transitions are made ###########
transdf_sep = pd.read_csv(dbdir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_interpolated_transitions_separated.csv'), index_col=0)
#limit to treatments
transdf_sep = transdf_sep[transdf_sep.Treatment.isin(treatments)]
transdf_sep['Treatment'] = pd.Categorical(transdf_sep.Treatment, categories=treatments, ordered=True)
transdf_sep = transdf_sep.sort_values(by='Treatment')



############## get the counts of cells leaving 
trans_rate_df_sep = pd.read_csv(dbdir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_binned_transition_rates_separated.csv'), index_col=0)
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


####### add lines to separate quadrants based on flux origin
ax.axvline(origin[0]-bintrim-0.5, ls = '--', lw = 1.5, color = '0.3')
ax.axhline(origin[1]-bintrim-0.5, ls = '--', lw = 1.5, color = '0.3')

    
#add the quadrant label
qlpos = [
    [2,7],
    [7,7],
    [7,3],
    [2,3]
    ]
for q, ql in enumerate(qlpos):
    ax.text(ql[0]-0.5, ql[1]-0.5, ['i','ii','iii','iv'][q],
            fontsize = 14, color = 'black', ha = 'center', va = 'center')
    


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
wilcoxon_pval = stats.wilcoxon(differences.flatten())[1]



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
star = get_stars(wilcoxon_pval)
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






########  Quandrant analysis


d = differences.copy()
ori_adj = np.array(origin) - bintrim
c = countmap[1]
orix = ori_adj[0]
oriy = ori_adj[1]

#make the quadrant dicts
#define quadrants not to include bins with quadrant line through them
quad = []
quad.append({'quadrant':['upperleft']*(len(d[oriy:,:orix-1].flatten())),
              'diffs': d[oriy:,:orix-1].flatten(),
              'counts': c[oriy:,:orix-1].flatten()})
quad.append({'quadrant':['upperright']*(len(d[oriy:,orix:].flatten())),
              'diffs': d[oriy:,orix:].flatten(),
              'counts': c[oriy:,orix:].flatten()})
quad.append({'quadrant':['lowerright']*(len(d[:oriy-1,orix:].flatten())),
              'diffs': d[:oriy-1,orix:].flatten(),
              'counts': c[:oriy-1,orix:].flatten()})
quad.append({'quadrant':['lowerleft']*(len(d[:oriy-1,:orix-1].flatten())),
              'diffs': d[:oriy-1,:orix-1].flatten(),
              'counts': c[:oriy-1,:orix-1].flatten()})
diffdf = pd.concat([pd.DataFrame(x) for x in quad])
diffdf['Treatment'] = treatments[1]


kw_stat, kw_pval = stats.kruskal(*[q.diffs.to_list() for _, q in diffdf.groupby('quadrant')])
print(f'Kruskal-Wallis p value for electrotaxis quadrants is {kw_pval}')
dunn = scikit_posthocs.posthoc_dunn(diffdf, val_col = 'diffs', group_col = 'quadrant', p_adjust = 'fdr_bh')
dunn_melt = pd.melt(dunn.reset_index(), id_vars = ['index'], value_vars = dunn.columns.to_list())
## rename melt variables
dunndf = dunn_melt.rename(columns = {'index': 'group1',
                                        'variable': 'group2',
                                        'value': 'p-adj'})
dunndf['Treatment'] = treatments[1]




#### set colors for the quadrant jitter plot
colors = ['#6cb875']*len(diffdf)


# Map treatment categories to x positions
treatment_to_x = {t: [0,4][i] for i, t in enumerate(diffdf['Treatment'].unique())}
treat_positions = diffdf['Treatment'].map(treatment_to_x).astype(float)
qo = 0.9
quad_offsets = np.linspace(-qo,qo,4)
quad_spacing = np.diff(quad_offsets)[0]
quad_to_x = {t: quad_offsets[i] for i, t in enumerate(diffdf['quadrant'].unique())}
quad_positions = diffdf['quadrant'].map(quad_to_x).astype(float)
#combine the treatment positions and the quadrant modifications
x_positions = treat_positions + quad_positions
unique_x_pos = np.unique(x_positions)
# Apply jitter to x positions (uniform noise)
# needs to be smaller than the quadrant offsets
jitter_strength = quad_spacing * 0.2 # smaller = less spread
jittered_x = x_positions + np.random.uniform(-jitter_strength, jitter_strength, size=len(diffdf))



#### start building the figure
fig, ax = plt.subplots(figsize = (5.5,3))
ax.scatter(jittered_x, diffdf['diffs'], s=diffdf['counts']/3, c = colors, edgecolor = '0.4',
            alpha=0.3, zorder = 2)
##get y limits for plotting significance stars later
ymin, ymax = ax.get_ylim()
# ax.set_ylim(-1.5,1.5)
### x axis labels
ax.set_xticks(unique_x_pos)
ax.set_xticklabels(['i','ii','iii','iv'])
### y axis fontsizes
ax.set_yticks(np.arange(-2.0,1.5,0.5))
ax.set_yticklabels(np.arange(-2.0,1.5,0.5))
ax.tick_params('both', labelsize =14)

### yaxis label
ax.set_ylabel('Relative Dwell Time (s)', fontsize = 18)


#legend sizes for the biggest and smallest dots on plot
legend_sizes = [diffdf[(diffdf.diffs>-3) & (diffdf.diffs<3)].counts.max()/3,
                diffdf[(diffdf.diffs>-3) & (diffdf.diffs<3)].counts.min()/3]
legend_labels = [str(int(s*3)) for s in legend_sizes]
handles = [
    Line2D([], [], marker='o', linestyle='None',
            markersize=np.sqrt(s), label=label, color='gray', alpha=0.6)
    for s, label in zip(legend_sizes, legend_labels)
]
ax.legend(handles=handles,
          title='Sample Size',
          bbox_to_anchor=[1.05,0.5],
          loc = 'center left',
           borderpad=0.25,
          title_fontsize = 16,
          fontsize = 14,
          labelspacing=0.75,)

#remove box lines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


#plot significance stars
#build x position keys
pos_keys = {treat: {quad: unique_x_pos[j+4*i] for j, quad in enumerate(diffdf.quadrant.unique())} for i, treat in enumerate(treatments[1:])}
for r, row in dunndf[dunndf['p-adj']<0.05].iterrows():
    
    pstar = get_stars(row['p-adj'])
    xp = np.sort(np.array([pos_keys[row.Treatment][row.group1],pos_keys[row.Treatment][row.group2]]))
    #increase the height if comparison isn't adjacent
    if (xp[0] == unique_x_pos[1]) | (xp[1] == unique_x_pos[1]):
        slv = 1.2
    elif (xp[0] == unique_x_pos[2]) | (xp[1] == unique_x_pos[2]):
        slv = 0
    elif np.diff(xp)[0] > quad_spacing+0.0001:
        slv = 1.2
    else:
        slv = 0
    barinc = (ymax-ymin)*0.08
    ax.text(xp.mean(), (ymax*0.985)+(barinc*slv), pstar, fontsize = 12, ha='center')
    #bar
    ax.plot([xp[0]+0.1,xp[1]-0.1], [ymax+(barinc*slv),ymax+(barinc*slv)], lw = 0.5, color = 'black')



#line at zero
xmin, xmax = ax.get_xlim()
ax.axhline(0,xmin,xmax, color = '0.7', ls = '--', zorder = 0)


#lines for the weighted means of the populations
#calculate weighted means
weightedmeans = []
for t in diffdf.Treatment.unique():
    wm = np.repeat(diffdf.loc[diffdf.Treatment==t,'diffs'].values,
              diffdf.loc[diffdf.Treatment==t,'counts'].values.astype(int))
    weightedmeans.append(np.mean(wm))
#plot weighted mean lines
widthaug = 1.2 #amount to multiply by to extend the avg lines
firstcent = treatment_to_x[treatments[1]]
ax.plot([firstcent - quad_offsets[-1] * widthaug,firstcent + quad_offsets[-1] * widthaug], [weightedmeans[0]]*2, c='black')

##### significance of whole pop
pstar = get_stars(wilcoxon_pval)
ax.text(1.15, weightedmeans[0]/2, pstar, fontsize = 12, va='center', rotation = -90)

plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '_quad_dots.png', dpi = 500, bbox_inches='tight')



