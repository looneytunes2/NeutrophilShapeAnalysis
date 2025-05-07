


import os 
import numpy as np
import pandas as pd
from CustomFunctions import segment_cells2short
import math
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter




#get some directories
basedir = 'E:/Aaron/Galvanotaxis_Confocal_40x_37C_10s/'
meshdir = basedir+'Meshes/'
infodir = basedir+'processed_data/'
cellname = '20231116_488EGFP-CAAX_3mA_37C_1_cell_42'
direct = f"//10.158.28.37/ExpansionHomesA/avlnas/HL60 Galv/20231116/{cellname.split('_cell')[0]}/Default/"


imshape = [361,150,1024,1024]
xyres = 0.3394 #um / pixel .2285
zstep = 0.7 # um
xybuffer = 50
zbuffer = 20
#get all the position and trajectory info
df = []
for x in os.listdir(infodir):
    if cellname in x:
        df.append(pd.read_csv(infodir+x, index_col = 0))
df = pd.concat(df).sort_values('frame').reset_index(drop=True)
#frame gaps
diffs = df.frame.diff()
gaps = diffs[diffs>1]
trimdf = df.iloc[72:121]#int(gaps.iloc[0])]
#go from microns back to pixels
trimdf['x_raw'] = trimdf['x_raw']/xyres
trimdf['y_raw'] = trimdf['y_raw']/xyres
trimdf['z_raw'] = trimdf['z_raw']/zstep
coords = trimdf[['x_raw','y_raw','z_raw']].values
mincoords = np.min(coords, axis = 0)
maxcoords = np.max(coords, axis = 0)
#expand coords with buffer
xmincrop = max(0,math.floor(mincoords[0]-xybuffer))
ymincrop = max(0,math.floor(mincoords[1]-xybuffer))
zmincrop = max(0,math.floor(mincoords[2]-zbuffer))
xmaxcrop = min(imshape[-1]-1,math.ceil(maxcoords[0]+xybuffer))
ymaxcrop = min(imshape[-1]-1,math.ceil(maxcoords[1]+xybuffer))
zmaxcrop = min(imshape[-3]-1,math.ceil(maxcoords[2]+zbuffer))
croparr = np.array([xmincrop,xmaxcrop,ymincrop,ymaxcrop,zmincrop,zmaxcrop])
timerange = range(72,121)

croppedmovie = np.zeros((len(timerange),
                         len(range(croparr[4],croparr[5])),
                         len(range(croparr[2],croparr[3])),
                         len(range(croparr[0],croparr[1]))
                         ))

for i,t in enumerate(timerange):
    #open the full zstack at this movie frame
    frameim = segment_cells2short.MM_slicetostack_reader(direct, t, imshape[-3:], range(croparr[4],croparr[5]))
    #crop frame to the cell
    croppedmovie[i] = frameim[
                    :,
                    croparr[2]:croparr[3],
                    croparr[0]:croparr[1]]


OmeTiffWriter.save(croppedmovie, __file__.split('.')[0]+'.ome.tiff', dim_order='TZYX')


