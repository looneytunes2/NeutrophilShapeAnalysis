# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:33:16 2024

@author: Aaron
"""

############## plot vectors of PC1/2 transitions from different migration modes #################

from cmocean import cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from CustomFunctions.DetailedBalance import contour_coords

# inverse scale for arrows
scale = 0.0005


# combine error data with real transition data
elldf = bsfield_sep.merge(trans_rate_df_sep,left_on = ['x','y','Treatment'], right_on = ['x','y','Treatment'])


#do color scales stuff
norm = matplotlib.colors.Normalize()
norm.autoscale([0,360])
cmm = cm.phase
color_scale = pd.DataFrame({'color':list(sns.diverging_palette(220, 20, n=180).as_hex()),
              'value':list(np.arange(0,180,1))})


meshdir = datadir + 'PC_Meshes/'

proj=[0,1]
PCkey = [1,2]
binkey = [1,5]      

fig = plt.figure(figsize=(10+(8*(len(elldf.Treatment.unique())-1)),10))

graphaxes, axes = PCvisualization.get_contours_for_axes(meshdir,
                proj,
                PCkey,
                binkey,
                graphnum = len(elldf.Treatment.unique()))


for i, ax in enumerate(graphaxes):
    mm = elldf.Treatment.unique()[i]
    mdf = elldf[elldf.Treatment==mm]
    #add "grid lines" first 
    for h in np.linspace(0.5, nbins+0.5, nbins+1):
        ax.axhline(h, linestyle='-', color='grey', alpha=0.3) # horizontal lines
        ax.axvline(h, linestyle='-', color='grey', alpha=0.3) # vertical lines

#     #add contou line if desired
#     uple = [3,8]
#     lori = [9,3]
#     contourcoords = contour_coords(uple,lori)
#     #actually plot the contour
#     ax.plot(np.array(contourcoords)[:,0], np.array(contourcoords)[:,1], lw=8,color='black',alpha=0.3)

        
    for x in range(1,nbins+1):
        for y in range(1,nbins+1):
            current = mdf[(mdf['x'] == x) & (mdf['y'] == y)]
            xcurrent = (current.x_plus_rate - current.x_minus_rate)/2
            ycurrent = (current.y_plus_rate - current.y_minus_rate)/2

            ell = Ellipse(xy=(x+(xcurrent.values*(1/scale)),y+(ycurrent.values*(1/scale))),
                    width=np.sqrt(abs(current.eval1))*(1/scale)*2,
                      height=np.sqrt(abs(current.eval2))*(1/scale)*2,
                    angle=np.arctan2(current.evec1y,current.evec1x))
            ax.add_artist(ell)
            ell.set_alpha(0.2)

    for x in range(1,nbins+1):
        for y in range(1,nbins+1):
            current = mdf[(mdf['x'] == x) & (mdf['y'] == y)]
            xcurrent = (current.x_plus_rate - current.x_minus_rate)/2
            ycurrent = (current.y_plus_rate - current.y_minus_rate)/2
            anglecolor = (np.arctan2(xcurrent,ycurrent) *180/np.pi)+180
            ax.quiver(x,
                       y, 
                       xcurrent,
                       ycurrent,
                      angles = 'xy',
                      scale_units = 'xy',
                      scale = scale,
#                       width = 0.012,
#                       minlength = 0.8,
                      color = cmm(norm(anglecolor)))


    #         print(x, x+(xcurrent.values*scale),y,  y+(ycurrent.values*scale))
    ax.set_xlabel('PC1', fontsize = 45)

    ax.set_xticks(list(range(1,nbins+1)),[round((PC1bins[i+1]+x)/2,1) for i,x in enumerate(PC1bins[:-1])], fontsize = 22)
    ax.set_yticks(list(range(1,nbins+1)),[round((PC8bins[i+1]+x)/2,1) for i,x in enumerate(PC8bins[:-1])], fontsize = 22)
    ax.set_xlim(0,nbins+1)
    ax.set_ylim(0,nbins+1)
    ax.set_title(mm, fontsize = 30)
    
    
axes[0].set_ylabel('PC8', fontsize = 45)
plt.tight_layout()


########### color wheel
# __a__=np.arange(0,1000*np.pi, np.pi/1.61803398875)
# __r__=0.3+np.log(1+np.arange(0, len(__a__))/len(__a__))
# ax.scatter(1-20+1*__r__*np.cos(__a__), 1+10*__r__*np.sin(__a__),5,c=np.mod(0.5-__a__/np.pi,1),cmap=cm.phase)
# ax.scatter(1.5+__r__*np.cos(__a__), 14+__r__*np.sin(__a__),5,c=np.mod(0.5-__a__/np.pi,1),cmap=cm.phase)


plt.savefig(savedir + 'PC1_PC8 Vector map absolute angle colored separated.png', bbox_inches='tight')
