
import numpy as np
import pandas as pd
from neutrophil_shape.CustomFunctions import utils
from neutrophil_shape.config.loader import load_config
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy import stats
from pathlib import Path

#get directories and open separated datasets
treatments = ['DMSO','Para-Nitro-Blebbistatin','CK666']
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'
time_interval = config.im_params.time_interval
datadir = config.common.savedir / 'shape_data'


FullFrame = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col=0)
#limit data to the Para-Nitro-Blebbistatin experiments
TotalFrame = FullFrame[FullFrame.Treatment.isin(treatments)].copy()
TotalFrame['Treatment'] = pd.Categorical(TotalFrame.Treatment.to_list(), categories=treatments, ordered=True)


#define the maximum time lag
maxlag = int(60/time_interval*5) # 5 minutes


plist = []
for c, cell in TotalFrame.groupby(['Treatment','CellID']):
    cell, runs = utils.get_consecutive_timepoints(cell, 'frame', 1)
    fmin = cell.frame.min()
    fmax = cell.frame.max()
    # galvdict[c[0]][c[1]] = {}
    plist.append({'Treatment':c[0],'CellID':c[1],'lag_min':0,'dot_prod':1})
    for lag in range(2,int(maxlag + 2)):
        if len(cell)>lag:
            frames = np.arange(fmin, fmax, step = lag)
            # plist = []
            for f in frames:
                traj = cell[cell.frame.isin([f,int(f+lag-1)])][['Trajectory_X','Trajectory_Z','Trajectory_Z']].values
                if len(traj)==2:
                    # Normalize vectors to get unit direction vectors
                    unitvecs = traj/np.linalg.norm(traj, axis = 1)[:, np.newaxis]
                    # Calculate dot products of consecutive unit vectors with the given lag
                    dot_products = np.sum(unitvecs[:-1] * unitvecs[1:], axis=1)
                    plist.append({'Treatment':c[0],'CellID':c[1],'lag_min':(lag-1)*time_interval/60,'dot_prod':dot_products[0]})
df = pd.DataFrame(plist)
df['Treatment'] = pd.Categorical(df.Treatment.to_list(), categories=treatments, ordered=True)


print(f'n = {df[df.lag_min==time_interval/60].groupby("Treatment").apply(lambda x: x.shape[0])} \
      track segments for time lag {time_interval/60}')
print(f'n = {df[df.lag_min==maxlag*time_interval/60].groupby("Treatment").apply(lambda x: x.shape[0])} \
      track segments for time lag {maxlag*time_interval/60}')



### run kruskal wallace at each time point
testlist = []
for i, l in df.groupby('lag_min'):
    if i != 0:
        _,pval = stats.kruskal(*[d.to_list() for _,d in l.groupby('Treatment').dot_prod])
        testlist.append({'lag':i,'pvalue':pval})
kwdf = pd.DataFrame(testlist)
#run BH multiple comparisons correction
reject, pvcorr = multipletests(kwdf['pvalue'],method='fdr_bh')[:2]
sigkw = kwdf[reject]

#set color palette
colorlist = ['#9c836b','#faa7a7','#faf191']
sns.set_palette(palette=colorlist)

##########  plot random versus galv
fig, ax = plt.subplots()
sns.lineplot(data = df, x = 'lag_min',y = 'dot_prod', hue = 'Treatment',  lw = 3, ax = ax)


ax.set_ylabel('Directional Autocorrelation', fontsize = 22)
ax.set_xlabel('Time lag (min)', fontsize = 22)

### make the legend larger
leg = ax.legend(loc = [0.4, 0.7])
for line in leg.get_lines():
    line.set_linewidth(3)
for text in leg.get_texts():
    text.set_fontsize(16)
    

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()


plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)





