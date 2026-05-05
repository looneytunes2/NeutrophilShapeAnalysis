# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:33:38 2025

@author: Aaron
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.font_manager import FontProperties
from neutrophil_shape.config.loader import load_config

config = load_config(microscope_type = 'confocal')


####### load common directories and data
dirlist = ['Combined_37C_Confocal_PCA_shape','Combined_37C_Confocal_PCA_s5','Combined_37C_Confocal_PCA_planar']
alignlist = ['Shape Only','Trajectory + Shape', 'Trajectory Only']
colorlist = cm.Set2.colors[:3][::-1]

dflist = []
for d, ali in enumerate(['shape','trajectory_shape','trajectory']):
    #set alignment
    config._alignment = ali
    time_interval = config.im_params.time_interval
    ntrans = config.db_params.ntrans
    allorigins = config.db_params.origins
    savedir = config.common.savedir
    datadir = savedir / 'shape_data'
    centers = pd.read_csv(datadir.joinpath('PC_bin_centers.csv'), index_col=0)
    binlist = centers.columns.to_list()
    dbdir = savedir / 'detailed_balance'
    dbbsdir = dbdir / 'alldatabs'
    file_list = dbbsdir.glob(f'*bootstrapped_{ntrans}_Area_Enclosed_Linear_Reg.csv')
    for f in file_list:
        tempdf = pd.read_csv(f, index_col = 0)
        firstpc, secondpc = f.name.split('_')[0].split('-')
        tempdf['first_pc'] = firstpc
        tempdf['second_pc'] = secondpc
        tempdf['Alignment Method'] = alignlist[d]
        dflist.append(tempdf)

df = pd.concat(dflist, ignore_index = True)
##drop nans
df = df[~df.aer_fit.isna()]

## add cycle period not just angular velocity
df['cycle_period'] = 360/(df.angular_velocity_coeff.abs()*60) ## (degrees/cycle)/((degrees/sec)*(sec/min)) = min/cycle


#### function to shift IQR for abs median
def signed_median_stats(series):
    med = series.median()
    q25, q75 = series.quantile([0.25, 0.75])
    abs_med = abs(med)
    if med >= 0:
        # positive median: abs_med == med, IQR maps directly
        err_lo = abs_med - q25   # distance down to Q1
        err_hi = q75 - abs_med  # distance up to Q3
    else:
        # negative median: reflection flips Q1/Q3
        err_lo = abs_med - abs(q75)  # distance down to reflected Q3
        err_hi = abs(q25) - abs_med  # distance up to reflected Q1
    # clamp to zero in case of floating point issues
    return pd.Series({
        'median': abs_med,
        'err_lo': max(err_lo, 0),
        'err_hi': max(err_hi, 0),
    })

stats = df.groupby(['Alignment Method','first_pc','second_pc']).apply(lambda g: pd.concat([
    signed_median_stats(g['aer_coeff']).add_prefix('aer_'),
    signed_median_stats(g['cycle_period']).add_prefix('cycle_period_'),
])).reset_index()




#get the colors for the errorbars
colors = [colorlist[alignlist.index(t)] for t in stats['Alignment Method']]

#plot the scattered data
fig, ax = plt.subplots()
scat = sns.scatterplot(data = stats, x='aer_median', y = 'cycle_period_median', hue = 'Alignment Method',
                palette = colorlist, ax = ax, zorder = 2)


##### IQR as error bars for medians
ax.errorbar(
    stats.aer_median,
    stats.cycle_period_median,
    xerr=[stats.aer_err_lo, stats.aer_err_hi],
    yerr=[stats.cycle_period_err_lo, stats.cycle_period_err_hi],
    fmt='none',
    ecolor=colors,
    elinewidth=1,
    alpha = 0.6,
    # capsize=4,
    zorder=1
)


ymin, ymax = ax.get_ylim()
ax.set_ylim(0, ymax)

#Change legend title and size
leg = ax.legend_
leg.set_title(title = 'Alignment Method', prop = FontProperties(size=12))

ax.set_xlabel("|Median Area Enclosing Rate| (PC units²/sec)", fontsize = 16)         
ax.set_ylabel("Median Cycle Period (min/cycle)", fontsize = 16)

# remove upper and right box lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# plt.show()
plt.tight_layout()

plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi=500)