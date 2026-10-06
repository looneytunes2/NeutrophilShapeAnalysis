
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from neutrophil_shape.CustomFunctions.shapePCAtools import filter_extremes_based_on_percentile
from neutrophil_shape.config.loader import load_config
from neutrophil_shape.CustomFunctions.utils import whichpc_string
from scipy import stats

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

whichpcs = (1,2)
wpcstr = whichpc_string(whichpcs)
### define treatments and associated colors
treatments = ['DMSO','Para-Nitro-Blebbistatin','CK666']
colorlist = ['#d1b59b','#f7bebe','#faf191']
treat_color_dict = {t:c for t,c in zip(treatments, colorlist)}
#load config stuff
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'
ntrans = config.db_params.ntrans
time_interval = config.im_params.time_interval

savedir = config.common.savedir
dbbsdir = savedir / 'detailed_balance'

df = pd.read_csv(dbbsdir.joinpath(f'{wpcstr}_raw_transition_aer_cf.csv'), index_col = 0)
df = df[df.Treatment.isin(treatments)].copy()
df['Treatment'] = pd.Categorical(df.Treatment.to_list(), categories=treatments, ordered=True)


# print(f'{treatments[0]} AER mean is {df[df.Treatment == treatments[0]].aer_coeff.mean()}'
#           f' and median is {df[df.Treatment == treatments[0]].aer_coeff.median()}')
# print(f'{treatments[1]} AER mean is {df[df.Treatment == treatments[1]].aer_coeff.mean()}'
#           f' and median is {df[df.Treatment == treatments[1]].aer_coeff.median()}')
# print(f'{treatments[2]} AER mean is {df[df.Treatment == treatments[2]].aer_coeff.mean()}'
#           f' and median is {df[df.Treatment == treatments[2]].aer_coeff.median()}')


# avgdf_filtered = filter_extremes_based_on_percentile(
#     df,
#     ['aer_coeff','aer_fit'],
#     1)



############### CELL CYCLE ACTIVITY AVERAGES #################################
metrics_dict = {
    'aer':'Area Enclosing Rate (PC units²/sec)',
    'pc_speed': 'PC Speed (PC units/sec)',
    }

### get average aer
avgdf = df.groupby(['Treatment','CellID'])[list(metrics_dict.keys())].mean().reset_index()

for metric, ylabel in metrics_dict.items():

    ## perform stats
    tstat, pnbpval = stats.mannwhitneyu(
        avgdf[avgdf.Treatment == treatments[0]][metric].values,
        avgdf[avgdf.Treatment == treatments[1]][metric].values,
        )
    tstat, ck666pval = stats.mannwhitneyu(
        avgdf[avgdf.Treatment == treatments[0]][metric].values,
        avgdf[avgdf.Treatment == treatments[2]][metric].values,
        )
    stardict = {}
    stardict[treatments[1]] = pnbpval
    stardict[treatments[2]] = ck666pval

    ### plot figure
    fig, ax = plt.subplots(1, 1, figsize=(4,5))#, sharex=True)
    linewid = 2
    sns.swarmplot(data = avgdf, x = 'Treatment', y = metric,
                hue = 'Treatment', palette=treat_color_dict,
                size = 1.5, ax = ax, zorder = 1)
    sns.boxplot(data = avgdf, x = 'Treatment', y = metric, width = 0.5,
                boxprops={
                    'fill': False,
                    'linewidth': linewid,
                    'edgecolor': 'black',
                    'zorder':2
                    },
                medianprops={
                    'linewidth': linewid,
                    'color': 'black'
                    },
                whiskerprops={
                    'linewidth': 0,
                    'color': 'black'
                    },
                capprops={
                    'linewidth': 0,
                    'color': 'black'
                    },
                showfliers=False, ax = ax, zorder = 2)

    #set ylim min to zero
    # ax.set_ylim(0, ax.get_ylim()[1])
    #tick stuff
    ax.set_ylabel(ylabel, fontsize = 16)#, labelpad=-0.5)
    ax.set_xlabel('')
    xticklabels = [x[:11] + '\n' + x[11:] if x == 'Para-Nitro-Blebbistatin' else x for x in treatments]
    ax.set_xticks(range(len(treatments)))
    ax.set_xticklabels(xticklabels)
    # Turn off all spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #remove legends
    ax.legend_ = None


    #get plot extrema
    ymin,ymax = ax.get_ylim()

    #bar placement adjustment
    barinc = (ymax-ymin)*0.08
    for t, treat in enumerate(treatments[1:]):
        ### plot star or ns for DMSO-PNB
        pval = stardict[treat]
        #print
        print(f'{treat} pval for {metric} is {pval}')
        pstar = 'n.s.' if pval>0.05 else get_stars(pval)
        #use different font sizes for stars vs n.s.
        nsfs = 10 if pstar=='n.s.' else 12
        xp = np.array([0,t+1])
        starinc = (ymax-ymin)*0.02 if pstar == 'n.s.' else (ymax-ymin)*0.001

        #star
        ax.text(xp.mean(), ymax+(barinc*t)+starinc, pstar, fontsize = nsfs, ha='center')
        #bar
        ax.plot([xp[0]+0.1,xp[1]-0.1], [ymax+(barinc*t),ymax+(barinc*t)], color = 'black')


    plt.tight_layout()
    plt.savefig(__file__.split('.')[0] + f'_{metric}.png', dpi = 500, bbox_inches='tight')


