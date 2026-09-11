



import pandas as pd
import numpy as np
import scipy.stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm 
from statsmodels.formula.api import ols 
import matplotlib.pyplot as plt
import seaborn as sns
from neutrophil_shape.CustomFunctions import utils
from neutrophil_shape.config.loader import load_config
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

#get directories and open separated datasets
treatments = ['DMSO','Para-Nitro-Blebbistatin','CK666']
config = load_config(microscope_type='confocal')
time_interval = config.im_params.time_interval

#list of metrics constant across alignments
constant_metrics = [
    'image',
    'CellID',
    'cell',
    'structure',
    'frame',
    'x_raw',
    'y_raw',
    'z_raw',
    'xmincrop',
    'ymincrop',
    'zmincrop',
    'xmaxcrop',
    'ymaxcrop',
    'zmaxcrop',
    'Cell_intensity_mean',
    'Cell_intensity_std',
    'Cell_intensity_1pct',
    'Cell_intensity_99pct',
    'Cell_intensity_max',
    'Cell_intensity_min',
    'centroid_inside',
    'Cell_Major_Axis_Vec_X',
    'Cell_Major_Axis_Vec_Y',
    'Cell_Major_Axis_Vec_Z',
    'Cell_Median_Axis_Vec_X',
    'Cell_Median_Axis_Vec_Y',
    'Cell_Median_Axis_Vec_Z',
    'Cell_Minor_Axis_Vec_X',
    'Cell_Minor_Axis_Vec_Y',
    'Cell_Minor_Axis_Vec_Z',
    'time',
    'Trajectory_Vec_X',
    'Trajectory_Vec_Y',
    'Trajectory_Vec_Z',
    'x',
    'y',
    'z',
    'Prev_Trajectory_Vec_X',
    'Prev_Trajectory_Vec_Y',
    'Prev_Trajectory_Vec_Z',
    'Next_Trajectory_Vec_X',
    'Next_Trajectory_Vec_Y',
    'Next_Trajectory_Vec_Z',
    'Turn_Angle',
    'persistence',
    'activity',
    'speed',
    'directional_autocorrelation',
    'Euler_Angles_X',
    'Euler_Angles_Y',
    'Euler_Angles_Z',
    'Cell_Volume',
    'Cell_SurfaceArea',
    'Cell_Sphericity',
    'Cell_Major_Axis_Min',
    'Cell_Major_Axis_Max',
    'Cell_Median_Axis_Min',
    'Cell_Median_Axis_Max',
    'Cell_Minor_Axis_Min',
    'Cell_Minor_Axis_Max',
    'Cell_Major_Axis_Length',
    'Cell_Median_Axis_Length',
    'Cell_Minor_Axis_Length',
    'Cell_Aspect_Ratio',
    'OriginaltoReconError',
    'RecontoOriginalError',
    'Date',
    'Experiment',
    'Treatment',
    ]

alldflist = []
for align in ['shape','trajectory_shape', 'trajectory']:
    config._alignment = align
    datadir = config.common.savedir / 'shape_data'
    #open data for this alignment
    FullFrame = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col=0)
    #limit data to the Para-Nitro-Blebbistatin experiments
    TotalFrame = FullFrame[FullFrame.Treatment.isin(treatments)].copy()
    TotalFrame['Treatment'] = pd.Categorical(TotalFrame.Treatment.to_list(), categories=treatments, ordered=True)
    ###filter the data for only cells that I have 10 or more frames of
    TotalFrame_filtered = TotalFrame[TotalFrame['CellID'].map(TotalFrame['CellID'].value_counts()) >= 10].copy()
    ## add alignment suffix
    TotalFrame_filtered = TotalFrame_filtered.add_suffix('_'+align)
    #get rid of suffix for constant columns
    TotalFrame_filtered = TotalFrame_filtered.rename(columns = {
        x+'_'+align:x for x in constant_metrics
    })
    alldflist.append(TotalFrame_filtered)

#merge all data
TotalFrame_merged = pd.concat(alldflist, axis = 1)
#remove duplicate ID columns
TotalFrame_merged = TotalFrame_merged.loc[:,~TotalFrame_merged.columns.duplicated()].copy()
#get cell averages
avgdf_filtered = TotalFrame_merged.groupby(['Treatment','CellID']).mean(numeric_only=True).reset_index(level='Treatment')
# #change the rear length to abs
# avgdf_filtered.loc[:,'LengthAlongTrajectoryRear'] = avgdf_filtered.LengthAlongTrajectoryRear.abs()

############### get list of metrics that are significant ttest of CELL AVERAGES ############
substrings = ['Axis_Vec', 'Axis_Length', '_Ratio', 'AlongTrajectory',]
single_metric_additions = ['Treatment','Cell_Volume','Cell_SurfaceArea','Cell_Sphericity','speed']
group_metrics = [s for s in TotalFrame_merged.columns if any(sub in s for sub in substrings)]
#remove the shape aligned axes vecs
group_metrics = [x for x in group_metrics if not all(y in x for y in ['_Axis_Vec_','_shape'])]
metriclist = single_metric_additions + group_metrics
## get just the straight PC values as well
pclist = [x for x in TotalFrame_merged.columns.to_list() if 'PC' in x and 'bin' not in x]
includelist = metriclist + pclist

#iterate through remaining columns and do two-way ttest between each drug and control
reslist = []
for col in includelist:
    for t in treatments[1:]:
        if col not in ['Treatment']:
            tempframe = avgdf_filtered[['Treatment', col]].dropna()
            if tempframe.loc[tempframe.Treatment=='DMSO', col].std() < 0.000001:
                print(col)
            tstat, pval = scipy.stats.ttest_ind(
                tempframe.loc[tempframe.Treatment=='DMSO', col].values,
                tempframe.loc[tempframe.Treatment==t, col].values
                )
            reslist.append({'metric': col, 'Treatment': t, 'pvalue': pval})
pvdf = pd.DataFrame(reslist).sort_values('Treatment')


#separately test statistics by treatment and metrics vs PCs
bothtreatsig = []
for tr in treatments[1:]:
    tpvdf = pvdf[pvdf.Treatment == tr].copy().reset_index(drop=True)
    metricdf = tpvdf[tpvdf.metric.isin(metriclist)].copy()
    pcdf = tpvdf[tpvdf.metric.isin(pclist)].copy()
    #correct pvalues for metrics
    metric_reject, metrics_pvcorr = multipletests(metricdf['pvalue'],method='fdr_bh')[:2]
    #correct pvalues for PCs
    PC_reject, PC_pvcorr = multipletests(pcdf['pvalue'],method='fdr_bh')[:2]
    #combine rejected hypotheses
    tempsigframe = pd.concat((metricdf[metric_reject], pcdf[PC_reject]), ignore_index=True)
    #add corrected PCs
    tempsigframe['pvcorr'] = np.concatenate((metrics_pvcorr[metric_reject], PC_pvcorr[PC_reject]))
    bothtreatsig.append(tempsigframe)
## combine treatments
sigframe = pd.concat(bothtreatsig, ignore_index=True)

#all significant comparisons
allsiglist = sigframe.metric.unique()


print(allsiglist)

siglist = allsiglist.copy()#[
    # 'speed',
    # 'Cell_MajorAxis_Vec_X',
    # 'Cell_MajorAxis_Vec_Y',
    # 'PC1',
    # 'PC2',
    # 'PC3',
    # 'Volume_Front_Ratio',
    # ]

ylabels = siglist.copy()#[
    # 'Instantaneous\nSpeed (µm/s)',
    # 'Major Axis X\nComponent',
    # 'Major Axis Y\nComponent',
    # 'PC1',
    # 'PC2',
    # 'PC3',
    # 'Front-Rear\nVolume Ratio',
    # ]

############### CELL AVERAGES OF SIGNIFICANT METRICS #################################
colorlist = ['#9c836b','#faa7a7','#faf191']
# sns.set_palette(palette=colorlist)

scale = int(len(siglist)/2)
linewid= 1.2

sq = int(np.ceil(np.sqrt(len(siglist))))

fig, axes = plt.subplots(sq,sq,figsize=(3*sq,4*sq))
flatax = axes.flatten()
for i, sig in enumerate(siglist):
    ax = flatax[i]
    sns.swarmplot(data = avgdf_filtered, x = 'Treatment', y = sig, hue = 'Treatment', palette=colorlist,
                    size = 1.3, ax = ax, zorder = 1)
    sns.boxplot(data = avgdf_filtered, x = 'Treatment', y = sig, width = 0.5,
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
    ax.set_ylabel(ylabels[i], fontsize = 16)#, labelpad=-0.5)
    ax.set_xlabel('')
    ax.set_xticklabels([x[:11]+'\n'+x[11:] if x == 'Para-Nitro-Blebbistatin' else x for x in treatments])
    # ax.set_xticks([])
    # Turn off all spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    #remove legends
    ax.legend_ = None


    #get plot extrema
    ymin,ymax = ax.get_ylim()
    
    #get only the significantly different comparisons
    starframe = sigframe[sigframe.metric == sig].reset_index(drop=True)


    #bar placement adjustment
    barinc = (ymax-ymin)*0.08
    for t, treat in enumerate(treatments[1:]):
        ### plot star or ns for DMSO-PNB
        row = starframe[starframe.Treatment==treat]
        #print
        print(f'{treat} pval for {ylabels[i]} is {pvdf[(pvdf.metric == sig) & (pvdf.Treatment == treat)].pvalue.iloc[0]}')
        pstar = 'n.s.' if row.empty else get_stars(row['pvcorr'].values[0])
        #use different font sizes for stars vs n.s.
        nsfs = 10 if pstar=='n.s.' else 12
        xp = np.array([0,t+1])
        starinc = (ymax-ymin)*0.02 if pstar == 'n.s.' else (ymax-ymin)*0.001

        #star
        ax.text(xp.mean(), ymax+(barinc*t)+starinc, pstar, fontsize = nsfs, ha='center')
        #bar
        ax.plot([xp[0]+0.1,xp[1]-0.1], [ymax+(barinc*t),ymax+(barinc*t)], color = 'black')

for a in range(i+1, len(flatax)):
    ax = flatax[a]
    ax.remove()

plt.tight_layout()


plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')

