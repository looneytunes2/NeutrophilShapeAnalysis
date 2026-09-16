
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse, Rectangle
from neutrophil_shape.config.loader import load_config
from neutrophil_shape.CustomFunctions.DetailedBalance import load_and_fill_transition_counts
from neutrophil_shape.CustomFunctions.utils import whichpc_string

whichpcs = (4,5)
wpcs = whichpc_string(whichpcs)

# inverse scale for arrows
scale = 0.0012

#get directories and open separated datasets
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'
treatments = ['Galvanotaxis']
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
transdf_sep = pd.read_csv(dbdir.joinpath(f'{wpcs}_interpolated_transitions.csv'), index_col=0)
transdf_sep = transdf_sep[transdf_sep.Treatment.isin(treatments)].copy()
#ensure that DMSO is the first in order
transdf_sep['Treatment'] = pd.Categorical(transdf_sep.Treatment, categories=treatments, ordered=True)
transdf_sep = transdf_sep.sort_values(by='Treatment')
############## get the counts of cells leaving
rates_path = dbdir.joinpath(f'{wpcs}_binned_transition_rates.csv')
trans_rate_df_sep = load_and_fill_transition_counts(rates_path, nbins,)
trans_rate_df_sep = trans_rate_df_sep[trans_rate_df_sep.Treatment.isin(treatments)].copy()
#ensure that DMSO is the first in order
trans_rate_df_sep['Treatment'] = pd.Categorical(trans_rate_df_sep.Treatment, categories=treatments, ordered=True)
trans_rate_df_sep = trans_rate_df_sep.sort_values(by='Treatment')
############# open average bootstrapped currents ###################
bsfield_sep = pd.read_csv(dbbsdir.joinpath(f'{wpcs}_bootstrapped_{ntrans}_transitions_average_currents.csv'), index_col=0)
bsfield_sep = bsfield_sep[bsfield_sep.Treatment.isin(treatments)].copy()
#ensure that DMSO is the first in order
bsfield_sep['Treatment'] = pd.Categorical(bsfield_sep.Treatment, categories=treatments, ordered=True)
bsfield_sep = bsfield_sep.sort_values(by='Treatment')




# combine fake error data with real transition data
elldf = bsfield_sep.merge(trans_rate_df_sep,left_on = ['x','y'], right_on = ['x','y'])



fig, ax = plt.subplots(figsize=(10,10))
#single colorbar axis
cbar_ax = fig.add_axes([.92, .082, .042, .784])

ttot = transdf_sep.time_elapsed.sum()
#make numpy array with heatmap data
bighm = np.zeros((nbins,nbins))
#get total time observed in the system

for x in range(nbins):
    for y in range(nbins):
        current =  transdf_sep[(transdf_sep['from_x'] == x+1) & (transdf_sep['from_y'] == y+1)]
        if current.empty:
            bighm[y,x] = 0
        else:
            bighm[y,x] = current.time_elapsed.sum()/ttot
#plot heatmap with seaborn
sns.heatmap(
    bighm,
    vmin=0, vmax=bighm.max(), #center=0,
    cmap='rocket',
    square=True,
    xticklabels = True,
    yticklabels = True,
    ax = ax,
#     cbar=i==0,
    cbar_ax = cbar_ax,
#         cbar_kws=cbar_kws
)


    
for x in range(1,nbins+1):
    for y in range(1,nbins+1):
        current = elldf[(elldf['x'] == x) & (elldf['y'] == y)]
        xcurrent = ((current.x_plus_rate - current.x_minus_rate)/2).iloc[0]
        ycurrent = ((current.y_plus_rate - current.y_minus_rate)/2).iloc[0]

        #add flux current arrow        
        ax.quiver(x-0.5,
                   y-0.5, 
                   xcurrent,
                   ycurrent,
                  angles = 'xy',
                  scale_units = 'xy',
                  scale = scale,
                  color = 'white',
                    zorder = 3 * 5)



        #determine ellipse width, height and angle
        #always set eval1 to width and adjust angle accordingly
        ex = xcurrent*(1/scale)
        ey = ycurrent*(1/scale)
        eh = np.sqrt(abs(current.eval2.iloc[0]))*(2/scale)
        ew = np.sqrt(abs(current.eval1.iloc[0]))*(2/scale)
        evec = current[['evec1x','evec1y']].values[0]
        evec = evec if evec[1]>0 else -evec
        eang = np.degrees(np.arctan2(evec[1],evec[0]))

        ell = Ellipse(xy=(x-0.5+ex,y-0.5+ey),
                        width=ew,
                        height=eh,
                        angle=eang,
                        color = 'lightblue',
                        alpha = 0.12,
                        zorder = 2)
        ax.add_artist(ell)
        

    

#         print(x, x+(xcurrent.values*scale),y,  y+(ycurrent.values*scale))
# axis label stuff
ax.set_xlabel(f'PC{whichpcs[0]}', fontsize = 45)
ax.set_ylabel(f'PC{whichpcs[1]}', fontsize = 45)
ax.xaxis.set_label_coords(0.47,-0.01)
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_xlim(0,nbins+1)
ax.set_ylim(0,nbins+1)
ax.set_title('Electrotaxis', fontsize = 50, loc = 'center')#,pad = -100)


# adjust colorbar tick label size
cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(),fontsize=18)
cbar_ax.set_ylabel('Probability', fontsize = 32, rotation = -90, labelpad = 33)


########## add scale for the vectors ##########
#legend background
lxp = 0.35
lyp = 0.35
legh = 1.4
legw = 4.4
rect = Rectangle((lxp, lyp), legw, legh, linewidth=1, edgecolor='black', facecolor='#80858a')
ax.add_patch(rect)
rect.set_zorder(4 * 5)
scalevalue = 0.0017
#x-axis legend arrow
#position of arrow in middle of box
arrxp = lxp + legw/2 - (scalevalue/scale)/2 
ax.quiver(arrxp,lyp+1.05,scalevalue,0,angles = 'xy',scale_units = 'xy',scale = scale,color = "white",zorder = 4 * 5)
#x-axis legend text
xsc = f'{scalevalue:.1e}'
xsc = xsc.split('e')[0] + 'x10$^{' +  str(int(xsc.split('e')[1])) + '}$'
ax.text(lxp+0.18,lyp+0.15,xsc+' $s^{-1}$', color = 'white', fontsize = 22, fontweight = 'bold',zorder = 4 * 5)

plt.tight_layout()



plt.savefig(__file__.split('.')[0] + '.png', bbox_inches='tight', dpi =500)
