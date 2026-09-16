

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse, Rectangle
from pathlib import Path
from neutrophil_shape.config.loader import load_config
from neutrophil_shape.CustomFunctions.DetailedBalance import load_and_fill_transition_counts
from neutrophil_shape.CustomFunctions.utils import whichpc_string

whichpcs = (4,5)

# inverse scale for arrows
scale = 0.0008

config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'
treatments = ['DMSO','Para-Nitro-Blebbistatin','CK666']
time_interval = config.im_params.time_interval
ntrans = config.db_params.ntrans
nbins = config.db_params.nbins

#get directories and open separated datasets
savedir = config.common.savedir
datadir = savedir / 'shape_data'
dbdir = savedir / 'detailed_balance'
dbbsdir = dbdir / 'separatedatabs'

#open the centers of the binned PCs
centers = pd.read_csv(datadir.joinpath('PC_bin_centers.csv'), index_col=0)
nbins = config.db_params.nbins

######## open all of the data
########### interpolate all transitions so that only individual transitions are made ###########
transdf_sep = pd.read_csv(dbdir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_interpolated_transitions.csv'), index_col=0)
transdf_sep = transdf_sep[transdf_sep.Treatment.isin(treatments)].copy()
#ensure that DMSO is the first in order
transdf_sep['Treatment'] = pd.Categorical(transdf_sep.Treatment, categories=treatments, ordered=True)
transdf_sep = transdf_sep.sort_values(by='Treatment')
############## get the counts of cells leaving
rates_path = dbdir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_binned_transition_rates.csv')
trans_rate_df_sep = load_and_fill_transition_counts(rates_path, nbins,)
trans_rate_df_sep = trans_rate_df_sep[trans_rate_df_sep.Treatment.isin(treatments)].copy()
#ensure that DMSO is the first in order
trans_rate_df_sep['Treatment'] = pd.Categorical(trans_rate_df_sep.Treatment, categories=treatments, ordered=True)
trans_rate_df_sep = trans_rate_df_sep.sort_values(by='Treatment')
############# open average bootstrapped currents ###################

all_bs_rates = pd.read_csv(dbbsdir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_bootstrapped_{ntrans}_transition_rates.csv'), index_col=0)



bsfield_sep = pd.read_csv(dbbsdir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_bootstrapped_{ntrans}_transitions_average_currents.csv'), index_col=0)
bsfield_sep = bsfield_sep[bsfield_sep.Treatment.isin(treatments)].copy()
#ensure that DMSO is the first in order
bsfield_sep['Treatment'] = pd.Categorical(bsfield_sep.Treatment, categories=treatments, ordered=True)
bsfield_sep = bsfield_sep.sort_values(by='Treatment')


### get some settings from config
nbins = config.db_params.nbins #how many bins in the x and y cgps axes
ntrans = config.db_params.ntrans #how many transitions to sample at each step
### open the data and fill sparse gaps with zeros to get real means
bsframe_sep_full = load_and_fill_transition_counts(
    dbbsdir.joinpath(f'{whichpc_string(whichpcs)}_bootstrapped_{ntrans}_transition_rates.csv'),
    nbins,
    'iter',
)






avgdmso = bsfield_sep[bsfield_sep.Treatment == 'DMSO']
justdmso = bsframe_sep_full[bsframe_sep_full.Treatment == 'DMSO']
justdmso['x_events'] = justdmso.x_minus_count + justdmso.x_plus_count
justdmso['y_events'] = justdmso.y_minus_count + justdmso.y_plus_count

highstate = justdmso[(justdmso.x == 3) & (justdmso.y == 12)]
highstate['bubble'] = 'high'
lowstate = justdmso[(justdmso.x == 3) & (justdmso.y == 15)]
lowstate['bubble'] = 'low'
big_and_small = pd.concat((highstate, lowstate), ignore_index = True)

fig, ax = plt.subplots()
sns.histplot(data=big_and_small, x='y_plus_count', hue = 'bubble',
             bins = 3, ax = ax)





#### estimate error in current field for this set of bootstrap realizations ######
####### this is for looking at data spread for the current field ############
full_index = pd.MultiIndex.from_product(
    [range(1, nbins + 1), range(1, nbins + 1)],
    names=['x', 'y']
)

bsfield = []
for m, mig in bsframe_sep_full.groupby('Treatment'):
    rows = []
    for (x, y), current in mig.groupby(['x', 'y']):
        js = np.column_stack([
            (current['x_plus_rate'].to_numpy() - current['x_minus_rate'].to_numpy()) / 2,
            (current['y_plus_rate'].to_numpy() - current['y_minus_rate'].to_numpy()) / 2,
        ])
        if js.shape[0] < 2:
            # not enough samples to estimate covariance/error at this bin
            evals = np.array([0.0, 0.0])
            evecs = np.eye(2)
        else:
            js_centered = js - js.mean(axis=0)
            avgjs = np.cov(js_centered.T)
            evals, evecs = np.linalg.eigh(avgjs)
        rows.append({'x':x+1,
                    'y':y+1,
                    'eval1':evals[1],
                    'eval2':evals[0],
                    'evec1x':evecs[0,1],
                    'evec1y':evecs[1,1],
                    'evec2x':evecs[0,0],
                    'evec2y':evecs[1,0],
                    'Treatment':m,
                    })
    default_row = {
        'eval1': 0,
        'eval2': 0,
        'evec1x': 0,
        'evec1y': 1,
        'evec2x': 1,
        'evec2y': 0,
        'Treatment':m,
    }
    df_m = pd.DataFrame(rows).set_index(['x', 'y'])
    df_m = df_m.reindex(full_index)
    df_m = df_m.fillna(value=default_row).reset_index()
    bsfield.append(df_m)


sparse_df = pd.read_csv(rates_path, index_col=0)
## get column names
dims = list(np.unique([c.split('_')[0] for c in sparse_df.columns if 'count' in c]))
value_cols = [c for c in sparse_df.columns if any([val in c for val in ['rate','count']])]

# build the full coordinate grid once
axes = [np.arange(1, nbins + 1)] * len(dims)
grid = np.meshgrid(*axes, indexing='ij')
coords = np.stack(grid, axis=-1).reshape(-1, len(dims))
coords_df = pd.DataFrame(coords, columns=dims)

### get non-value columns as list
groups = ['Treatment', group_factor] if group_factor else ['Treatment']
idx_cols = groups + dims

# unique factor combinations actually present in the data
factor_combos = sparse_df[groups].drop_duplicates()

# single cross join
coords_df['_key'] = 1
factor_combos = factor_combos.assign(_key=1)
full_index_df = factor_combos.merge(coords_df, on='_key').drop(columns='_key')
# index-based join
full_index_df = full_index_df.set_index(idx_cols)
sparse_indexed = sparse_df.set_index(idx_cols)[value_cols]
full_df = full_index_df.join(sparse_indexed, how='left')
full_df[value_cols] = full_df[value_cols].fillna(0)

full_df = full_df.reset_index().sort_values(by=idx_cols).reset_index(drop=True)