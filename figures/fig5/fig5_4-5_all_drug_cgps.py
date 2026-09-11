


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse, Rectangle
from pathlib import Path
from neutrophil_shape.config.loader import load_config
from neutrophil_shape.CustomFunctions.DetailedBalance import load_and_fill_transition_counts


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
trans_rate_df_sep = load_and_fill_transition_counts(rates_path, nbins, 'Treatment')
trans_rate_df_sep = trans_rate_df_sep[trans_rate_df_sep.Treatment.isin(treatments)].copy()
#ensure that DMSO is the first in order
trans_rate_df_sep['Treatment'] = pd.Categorical(trans_rate_df_sep.Treatment, categories=treatments, ordered=True)
trans_rate_df_sep = trans_rate_df_sep.sort_values(by='Treatment')
############# open average bootstrapped currents ###################
bsfield_sep = pd.read_csv(dbbsdir.joinpath(f'PC{whichpcs[0]}-PC{whichpcs[1]}_bootstrapped_{ntrans}_transitions_average_currents.csv'), index_col=0)
bsfield_sep = bsfield_sep[bsfield_sep.Treatment.isin(treatments)].copy()
#ensure that DMSO is the first in order
bsfield_sep['Treatment'] = pd.Categorical(bsfield_sep.Treatment, categories=treatments, ordered=True)
bsfield_sep = bsfield_sep.sort_values(by='Treatment')



########### PDFs AND PROBABILITY FLUX OF THE SEPARATED Treatments #############

# combine error data with real transition data
elldf = bsfield_sep.merge(trans_rate_df_sep,left_on = ['x','y','Treatment'], right_on = ['x','y','Treatment'])


################ heatmap of probability density #############
bighm = np.zeros((len(treatments),nbins,nbins))
for i,(t, mdf) in enumerate(transdf_sep.groupby('Treatment')):
    #get total time observed in the system
    ttot = mdf.time_elapsed.sum()
    for x in range(nbins):
        for y in range(nbins):
            current =  mdf[(mdf['from_x'] == x+1) & (mdf['from_y'] == y+1)]
            if current.empty:
                bighm[i,y,x] = 0
            else:
                bighm[i,y,x] = current.time_elapsed.sum()/ttot
                
                
                
            

####### figure stuff
fig, graphaxes = plt.subplots(1,len(treatments),figsize=(7.5+(10*(len(treatments)-1)),10))
#single colorbar axis
cbar_ax = fig.add_axes([.98, .033, .02, .81])
                
                
for i, ax in enumerate(graphaxes):
    #plot heatmap with seaborn
    sns.heatmap(
        bighm[i],
        vmin=0, vmax=bighm.max(), #center=0,
        cmap='rocket',
        square=True,
        ax = ax,
        cbar=i==0,
        cbar_ax = None if i else cbar_ax,
#         cbar_kws=cbar_kws
    )
  
    ######################### vector map of probability flux ################
    mm = elldf.Treatment.unique()[i]
    mdf = elldf[elldf.Treatment==mm]
    
    
    for x in range(1,nbins+1):
        for y in range(1,nbins+1):
            current = mdf[(mdf['x'] == x) & (mdf['y'] == y)]
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
                        zorder = 3)


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


            
                    


    # axis label stuff
    if i==1:    
        ax.set_xlabel('PC1', fontsize = 45, ha = 'center', labelpad = 10)
    else:
        ax.set_xlabel('')
    # ax.xaxis.set_label_coords(0.46,-0.05)
    # if i == 0:
    #     ax.set_yticks(np.arange(0.5,nbins+0.5))
    #     ax.set_yticklabels([round(x,1) for x in centers.PC7.to_list()], fontsize = 19)
    # else:
    #     ax.set_yticks([])
    #     ax.set_yticklabels([])
        
        
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    ax.set_xlim(0,nbins+1)
    ax.set_ylim(0,nbins+1)
    ax.set_title(mm if mm!='Para-Nitro-Blebbistatin' else mm[:11]+'\n'+mm[11:], fontsize = 50, loc = 'center')#,pad = -100)
    # ax.title.set_position([0.5, 1.05])
    # ax.title.set_position([0.45, -100])
    # ax.text(nbins/2, nbins, mm, fontsize=50, ha='center')
    
# adjust colorbar tick label size
cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(),fontsize=18)
cbar_ax.set_ylabel('Probability', fontsize = 32, rotation = -90, labelpad = 33)
#set axis title
graphaxes[0].set_ylabel('PC7', fontsize = 45, labelpad = -20)
graphaxes[0].yaxis.set_label_coords(-0.05, 0.465)


########## add scale for the vectors ##########
#legend background
lxp = 0.35
lyp = 0.35
legh = 1.4
legw = 4.4
rect = Rectangle((lxp, lyp), legw, legh, linewidth=1, edgecolor='black', facecolor='#80858a')
graphaxes[0].add_patch(rect)
rect.set_zorder(4 * 5)
scalevalue = 0.0017
#x-axis legend arrow
#position of arrow in middle of box
arrxp = lxp + legw/2 - (scalevalue/scale)/2 
graphaxes[0].quiver(arrxp,lyp+1.05,scalevalue,0,angles = 'xy',scale_units = 'xy',scale = scale,color = "white",zorder = 4 * 5)
#x-axis legend text
xsc = f'{scalevalue:.1e}'
xsc = xsc.split('e')[0] + 'x10$^{' +  str(int(xsc.split('e')[1])) + '}$'
graphaxes[0].text(lxp+0.1,lyp+0.15,xsc+' $s^{-1}$', color = 'white', fontsize = 24, fontweight = 'bold',zorder = 4 * 5)


plt.tight_layout()




plt.savefig(__file__.split('.')[0]+'.png', bbox_inches='tight', dpi = 500)



