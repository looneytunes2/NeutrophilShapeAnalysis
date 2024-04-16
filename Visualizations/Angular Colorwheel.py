# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:10:42 2024

@author: Aaron
"""
import matplotlib.pyplot as plt
import numpy as np
from cmocean import cm


savedir = ''

fig, ax = plt.subplots(figsize=(10,10))
__a__=np.arange(0,20000*np.pi, np.pi/1.61803398875)
__r__=0.5+np.log(1+np.arange(0, len(__a__))/len(__a__))
# ax.scatter(1-20+1*__r__*np.cos(__a__), 1+10*__r__*np.sin(__a__),5,c=np.mod(0.5-__a__/np.pi,1),cmap=cm.phase)
ax.scatter(__r__*np.cos(__a__), __r__*np.sin(__a__),5,c=np.mod(0.5-__a__/np.pi,1),cmap=cm.phase)
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
plt.savefig(savedir + 'COLORWHEEL.png', bbox_inches='tight', dpi=500)