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
from matplotlib.patches import Rectangle
import math
from statsmodels.stats.multicomp import pairwise_tukeyhsd


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
origin = [9,9]
vmin = -1.5 #lower bound for heatmap 
vmax = 1.5 #upper bound for heatmap 

#define some directories
basedir = 'E:/Aaron/Combined_37C_Confocal_PCA_s5/'
datadir = basedir + 'Data_and_Figs/'
savedir = basedir + 'drug/'

#open the centers of the binned PCs
centers = pd.read_csv(datadir+'PC_bin_centers.csv', index_col=0)
nbins = len(centers.iloc[:,0])
#trim bins outside 2 std
bintrim = 3
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
    ax.set_xticks(np.arange(0.5,nbins_trim+0.5)[[0,nbins_trim//2,-1]])
    ax.set_xticklabels([round(centers.PC1.iloc[x+bintrim],1) for x in [0,nbins_trim//2, int(nbins_trim-1)]],
                       fontsize = 14)
    ax.set_yticks(np.arange(0.5,nbins_trim+0.5)[[0,nbins_trim//2,-1]])
    ax.set_yticklabels([round(centers.PC7.iloc[x+bintrim],1) for x in [0,nbins_trim//2, int(nbins_trim-1)]],
                       fontsize = 14)
    #set axis titles
    ax.set_xlabel(f'PC{whichpcs[0]}', fontsize = 24)

    #set title
    ax.set_title([x[:11]+'\n'+x[11:] if x == 'Para-Nitro-Blebbistatin' else x for x in treatments][int(i+1)], fontsize = 32)


    ### make the quadrant boxes
    #upper left
    bpadj = 0.07
    boxpos = [[bpadj, math.ceil(nbins_trim/2) + bpadj],
              [math.ceil(nbins_trim/2) + bpadj, math.ceil(nbins_trim/2) + bpadj],
              [math.ceil(nbins_trim/2) + bpadj, bpadj],
              [bpadj,bpadj]]
    qlpos = [
        [3,8],
        [8,8],
        [8,3],
        [3,3]
        ]
    for b, bp in enumerate(boxpos):
        if b==0: 
            w = nbins_trim//2 - 2*bpadj + 1
            h = nbins_trim//2 - bpadj*2
        elif b==1:
            w = nbins_trim//2 - 2*bpadj
            h = nbins_trim//2 - 2*bpadj
        elif b==2:
            w = nbins_trim//2 - 2*bpadj
            h = nbins_trim//2 - 2*bpadj + 1
        elif b==3:
            w = nbins_trim//2 - 2*bpadj + 1
            h = nbins_trim//2 - 2*bpadj + 1

        qb = Rectangle(xy = bp, width = w, height = h,
                       fill = False,
                       ls = '--',
                       lw = 1.5,
                       edgecolor = '0.3')
        ax.add_patch(qb)
        
        #add the quadrant label
        ax.text(qlpos[b][0]-0.5, qlpos[b][1]-0.5, ['i','ii','iii','iv'][b], fontsize = 14, color = 'black', ha = 'center', va = 'center')
        


axes[0].set_ylabel(f'PC{whichpcs[1]}', fontsize = 24)

# adjust colorbar tick label size
cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(),fontsize=12)
cbar_ax.get_yaxis().labelpad = 18
cbar_ax.set_ylabel('Relative Dwell Time (sec)', fontsize = 16, rotation=270)


plt.tight_layout()

plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')



############## JITTER WITH QUADRANTS

tukeylist = []
diffdflist = []
ori_adj = np.array(origin) - bintrim
for i, d in enumerate(differences):
    c = countmap[i+1]
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
    
    quaddf['Treatment'] = treatments[i+1]
    diffdflist.append(quaddf)
    
    print(stats.f_oneway(*[q.diffs.to_list() for _, q in quaddf.groupby('quadrant')]))
    tukey = pairwise_tukeyhsd(endog=quaddf.diffs.values, groups=quaddf.quadrant.values, alpha=0.05)
    temptukey = pd.DataFrame(
        data=tukey._results_table.data[1:], 
        columns=tukey._results_table.data[0] 
    )
    temptukey['Treatment'] = treatments[i+1]
    tukeylist.append(temptukey)
    
tukeydf = pd.concat(tukeylist, ignore_index = True)
diffdf = pd.concat(diffdflist, ignore_index = True)



#set the color palette
colorlist = ['#9c836b','#faa7a7','#faf191']
sns.set_palette(palette=colorlist)
group_colors = {treatments[1]: colorlist[1], treatments[2]: colorlist[2]}
colors = diffdf['Treatment'].map(group_colors)


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
fig, ax = plt.subplots(figsize = (7,3))

ax.scatter(jittered_x, diffdf['diffs'], s=diffdf['counts']/3, c = colors, edgecolor = '0.4',
           alpha=0.3, zorder = 2)

### x axis labels
ax.set_xticks(unique_x_pos)
ax.set_xticklabels(['i','ii','iii','iv']*2)
### y axis fontsizes
ax.set_yticks(np.arange(-0.5,2.5,0.5))
ax.set_yticklabels(np.arange(-0.5,2.5,0.5))
ax.tick_params('both', labelsize =14)
# labelcolors = np.round(np.tile(np.linspace(0.2,0.8,4), 2), 2).astype(str)
# for label, color in zip(ax.get_xticklabels(), 'black'):
#     label.set_color(color)
    
### yaxis label
ax.set_ylabel('Relative Dwell Time (s)', fontsize = 18)


# ### label treaments
# ax.text(treat_positions.unique()[0], -1.3, 'Para-Nitro-Blebbistatin', fontsize = 12, ha = 'center')
# ax.text(treat_positions.unique()[1], -1.3, 'CK666', fontsize = 12, ha = 'center')


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
          bbox_to_anchor=[0.5,0.9],
          loc = 'center',
          borderpad=0.25,
          title_fontsize = 14,
          fontsize = 12,
          labelspacing=1.5,)

#remove box lines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


#plot significance stars
ticklabels = [x.get_text() for x in ax.xaxis.get_ticklabels()]
ymin, ymax = ax.get_ylim()
#build x position keys
pos_keys = {treat: {quad: unique_x_pos[j+4*i] for j, quad in enumerate(diffdf.quadrant.unique())} for i, treat in enumerate(treatments[1:])}
for r, row in tukeydf[tukeydf.reject].iterrows():
    
    pstar = get_stars(row['p-adj'])
    xp = np.sort(np.array([pos_keys[row.Treatment][row.group1],pos_keys[row.Treatment][row.group2]]))
    #increase the height if comparison isn't adjacent
    if (xp[0] == unique_x_pos[1]) | (xp[1] == unique_x_pos[1]):
        slv = 1
    elif (xp[0] == unique_x_pos[2]) | (xp[1] == unique_x_pos[2]):
        slv = 0
    elif np.diff(xp)[0] > quad_spacing+0.0001:
        slv = 1
    else:
        slv = 0
    starinc = (ymax-ymin)*0.001 if pstar == 'n.s.' else (ymax-ymin)*0.001
    barinc = (ymax-ymin)*0.08
    #star
    nsfs = 10 if pstar=='n.s.' else 12
    # ax.text(xp.mean(), ymax+starinc, pstar, fontsize = nsfs, ha='center')
    ax.text(xp.mean(), ymax+(barinc*slv), pstar, fontsize = nsfs, ha='center')
    #bar
    ax.plot([xp[0]+0.1,xp[1]-0.1], [ymax+(barinc*slv),ymax+(barinc*slv)], lw = 0.5, color = 'black')


#t-test for overall population different than zero
sig = [stats.ttest_1samp(dif.flatten(), popmean=0)[1] for dif in differences]
print(sig)
# #significance stars
# ax.text(treatment_to_x[treatments[1]], 4, get_stars(sig[0]), fontsize = 14,
#         color = 'black', va = 'center', ha='center')
# ax.text(treatment_to_x[treatments[2]], 4, get_stars(sig[1]), fontsize = 14,
#         color = 'black', va = 'center', ha='center')


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
secondcent = treatment_to_x[treatments[2]]
ax.plot([secondcent - quad_offsets[-1] * widthaug,secondcent + quad_offsets[-1] * widthaug], [weightedmeans[1]]*2, c='black')

#line at zero
xmin, xmax = ax.get_xlim()
ax.axhline(0,xmin,xmax, color = '0.7', ls = '--', zorder = 0)

plt.tight_layout()

plt.savefig(__file__.split('.')[0] + '_quad_dots.png', dpi = 500, bbox_inches='tight')








# ############# DOT PLOT VERSION ###################
# #change treatment list so that drugs are in the correct order visually
# treatments = ['DMSO','CK666','Para-Nitro-Blebbistatin']
# #change the order of the differences to match
# differences = differences[::-1]

# diflist = []
# for i, d in enumerate(differences):
#     diflist.append(pd.DataFrame({'diffs':list(d.flatten()),
#                              'Treatment':[treatments[int(i+1)]]*len(d.flatten()),
#                              'counts':(countmap[int(i+1)].flatten())}))
# diffdf = pd.concat(diflist)
    
# #print the results of the one-sample ttest
# sig = [stats.ttest_1samp(dif.flatten(), popmean=0)[1] for dif in differences]



# #set the color palette
# colorlist = ['#9c836b','#faa7a7','#faf191']
# sns.set_palette(palette=colorlist)
# group_colors = {treatments[1]: colorlist[2], treatments[2]: colorlist[1]}
# colors = diffdf['Treatment'].map(group_colors)


# #### start building the figure
# fig, ax = plt.subplots(figsize = (5,3))

# # Map treatment categories to x positions
# treatment_to_x = {t: i for i, t in enumerate(diffdf['Treatment'].unique())}
# x_positions = diffdf['Treatment'].map(treatment_to_x).astype(float)

# # Apply jitter to x positions (uniform noise)
# jitter_strength = 0.33  # smaller = less spread
# jittered_x = x_positions + np.random.uniform(-jitter_strength, jitter_strength, size=len(diffdf))

# ax.scatter(diffdf['diffs'], jittered_x, s=diffdf['counts']/3, c = colors, edgecolor = '0.4',
#            alpha=0.3, zorder = 2)

# #lines for the weighted means of the populations
# #calculate weighted means
# weightedmeans = []
# for t in diffdf.Treatment.unique():
#     wm = np.repeat(diffdf.loc[diffdf.Treatment==t,'diffs'].values,
#               diffdf.loc[diffdf.Treatment==t,'counts'].values.astype(int))
#     weightedmeans.append(np.mean(wm))
# #plot weighted mean lines
# ax.plot([weightedmeans[0]]*2,[-0.33,0.33], c='black')
# ax.plot([weightedmeans[1]]*2,[1-0.33,1+0.33], c='black')

# #line at zero
# ax.axvline(0,-1,2, color = '0.7', ls = '--', zorder = 1)


# #significance stars
# ax.text(3, 0, get_stars(sig[0]), fontsize = 12, va = 'center', ha='center')
# ax.text(3, 1, get_stars(sig[1]), fontsize = 12, va = 'center', ha='center')

# #adjust plot limits to center 0
# ax.set_ylim(-0.5,1.5)
# ax.set_yticks([0,1])
# ax.set_xlim(-3,3.05)
# # ax.set_xticks([-3,0,3])

# ax.set_yticklabels(treatments[1:])
# ax.set_yticklabels([x[:11]+'\n'+x[11:] if x == 'Para-Nitro-Blebbistatin' else x for x in treatments[1:]], fontsize = 14,
#                    rotation=90, va = 'center')

# ax.set_xlabel('Relative Dwell Time (s)', fontsize = 14)


# #legend sizes for the biggest and smallest dots on plot
# legend_sizes = [diffdf[(diffdf.diffs>-3) & (diffdf.diffs<3)].counts.max()/3,
#                 diffdf[(diffdf.diffs>-3) & (diffdf.diffs<3)].counts.min()/3]
# legend_labels = [str(int(s*3))+' samples' for s in legend_sizes]
# handles = [
#     Line2D([], [], marker='o', linestyle='None',
#            markersize=np.sqrt(s), label=label, color='gray', alpha=0.6)
#     for s, label in zip(legend_sizes, legend_labels)
# ]

# ax.legend(handles=handles,
#           title='Dot Size',
#           bbox_to_anchor=[0.175,0.5],
#           loc = 'center',
#           borderpad=0.75,
#           title_fontsize = 11,
#           fontsize = 8,
#           labelspacing=1.5,)


# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# plt.tight_layout()

# plt.savefig(__file__.split('.')[0] + '_dots.png', dpi = 500, bbox_inches='tight')



