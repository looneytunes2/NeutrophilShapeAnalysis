# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 14:18:38 2025

@author: Aaron
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

time_interval = 10
meshframes = np.arange(66, 92,3)
#get the discrete times in minutes
times = (meshframes-meshframes.min())*time_interval/60
#get the time reversed as strings
timelabels = -times[::-1]
timelabels[timelabels == 0] = 0
#define colorscale boundaries
boundaries = np.linspace(0,times.max(),len(times)+1)
tick_locs = (boundaries[:-1] + boundaries[1:]) / 2  # centers

cmap = matplotlib.cm.get_cmap('rainbow', len(times))
norm = matplotlib.colors.BoundaryNorm(boundaries=np.arange(len(times) + 1) - 0.5, ncolors=len(times))

norm = matplotlib.colors.Normalize(0,times.max())

fig, ax = plt.subplots()
# cbar_ax = fig.add_axes([0.211, 0.09, 0.603, 0.013]) 
fig.subplots_adjust(bottom=0.45, top=0.55)
cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ticks = tick_locs,
                    orientation = 'horizontal',
                    cax=ax)


cbar.set_label('Time (min)', fontsize=26)
cbar.ax.xaxis.set_label_position('bottom')
cbar.ax.set_xticklabels(timelabels,fontsize=16)




plt.savefig(__file__.split('.')[0] + '.png', dpi = 500, bbox_inches='tight')