

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from neutrophil_shape.CustomFunctions import utils
from neutrophil_shape.config.loader import load_config


config = load_config(microscope_type='lls')
config._alignment = 'trajectory'
time_interval = config.im_params.time_interval
whichpcs = (1,2)
ntrans = config.db_params.ntrans
basedir = config.common.savedir
datadir = basedir.joinpath('shape_data')
dbdir = basedir.joinpath('detailed_balance')
dbbsdir = dbdir.joinpath('separatedatabs')

FullFrame = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col = 0)
FullFrame['real_time'] = FullFrame.time.copy()

#open aers previously calculated
allaers = pd.read_csv(dbdir.joinpath(utils.whichpc_string(whichpcs) + '_raw_transition_aer_cf.csv'), index_col = 0)

#merge aer and cf info
TotalFrame = FullFrame.merge(allaers, on=['CellID','real_time','frame'] ,how='left')
TotalFrame = TotalFrame.sort_values(['CellID','time'])

#open bootstrapped gaps and fits with gaps
bsgaps = pd.read_csv(dbbsdir.joinpath(utils.whichpc_string(whichpcs) + f'_bootstrapped_{ntrans}_Area_Enclosing_Rates_gaps.csv'), index_col = 0)
bsdf = pd.read_csv(dbbsdir.joinpath(utils.whichpc_string(whichpcs) + f'_bootstrapped_{ntrans}_Area_Enclosed_Linear_Reg_gaps.csv'), index_col = 0)


#only use aers that are within the range of observed time of the real cells
minmaxtime = TotalFrame.groupby('CellID').time.max().min()
itertime = bsgaps.groupby('iter').real_time.max()
longiters = itertime[itertime>=minmaxtime].index.to_list()
bsaers_long = bsdf[bsdf.iter.isin(longiters)].copy()



#calculate aer and fit for real cells
dflist = []
for i, t in TotalFrame.groupby('CellID'):
    ### get ID info
    id_dict = {'CellID':i}
    #linear regression
    fitdict = utils.fit_rates_linear(t, ['aer','angular_velocity'])
    id_dict.update(fitdict)
    dflist.append(id_dict)
avgdf = pd.DataFrame(dflist)


### open bootstrapped data



#define the colors to make the meshes
set1 = plt.cm.Set1
set2 = plt.cm.Set2
set3 = plt.cm.Set3
cmap = matplotlib.colors.ListedColormap(list(set3.colors)+[set2.colors[-2]] + [set1.colors[-1]])
sns.set_palette(cmap.colors)

#truncate the greys color map for the density plot
greymap = plt.get_cmap('Greys')
new_cmap = matplotlib.colors.ListedColormap(
    greymap(np.linspace(0.15, 0.8, 256)))


### probability density proportions to use as levels for the kde plot
lvls = [0.05,0.2,0.4,0.6,0.8,1]

### plot the stuff
fig, ax = plt.subplots()
cbar_ax = fig.add_axes([.98, .24, .03, .6])

#individual dots
sns.scatterplot(y = 'aer_fit', x = 'aer_coeff', data = avgdf, hue = 'CellID',
                s = 100, edgecolor = '0.4', ax = ax, zorder = 2)

#density plot of the bootstrapped data
sns.kdeplot(data = bsaers_long, x = 'aer_coeff', y = 'aer_fit', levels = lvls, fill = True,
            cmap = new_cmap, cbar = True, cbar_ax = cbar_ax, ax = ax, zorder = 1)

ax.set_ylabel('Area Enclosing Rate R$^2$', fontsize = 18)
ax.set_xlabel('Area Enclosing Rate (PC units²/sec)', fontsize = 18)

#change fontsize on axis ticks
ax.tick_params(labelsize = 8)

ax.set_xlim(0,0.029)#(0.0085,0.0315)
ax.set_ylim(0,1.03)#(0.93,1.005)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend_ = None
# ax.set_aspect('equal')

#adjust the colobar stuff
cbar_ax.set_yticklabels(lvls,fontsize=8)
# cbar_ax.get_yaxis().labelpad = 18
cbar_ax.set_ylabel('Bootstrapped Density Proportion', fontsize = 10,
                   rotation=-90, labelpad = 13)


# plt.tight_layout()
# plt.show()



plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')

