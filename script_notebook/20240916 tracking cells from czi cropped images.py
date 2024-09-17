# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:46:32 2024

@author: Aaron
"""

import os
import pandas as pd
import numpy as np
import re
from CustomFunctions.track_functions import tracking_track
from CustomFunctions.metadata_funcs import get_sec

direct = '//10.158.28.37/ExpansionHomesA/avlnas/LLS/crop_csvs/'

csvlist = [x for x in os.listdir(direct) if 'DMSO' in x]



readcsvs = [pd.read_csv(direct + x) for x in csvlist]

colnames = readcsvs[0].columns.to_list()
newcolnames = dict([[c,c.split('::')[0]] for c in colnames])
timecols = [newcolnames[str(x)] for x in newcolnames if "Time" in x]

newcsvs = []
for r in readcsvs:
    temp = r.rename(columns = newcolnames)
    temp = temp.drop(0)
    temp['ImageCell'] = [re.findall('.*cell\d(?=-\d*)', temp.ImageDocumentName.iloc[0])[0]]*len(temp)
    print(len(temp.columns))
    newcsvs.append(temp)
df = pd.concat(newcsvs)


['ImageStageXPosition','ImageStageYPosition','ImageFocusPosition']

df.iloc[0]['CenterX3']

for i, im in df.groupby('ImageCell'):
    attime = [get_sec(x.split('T')[1][:8]) for x in im.ImageAcquisitionTime.to_list()]
    reltime = [get_sec(x[:8]) for x in im.ImageRelativeTime.to_list()]
    im['CombinedTime'] = np.array(attime)+np.array(reltime)
    #add stage position to the image positions
    im['x'] = im.CenterX3.astype(float) + im.ImageStageXPosition.astype(float)
    im['y'] = im.CenterY3.astype(float) + im.ImageStageYPosition.astype(float)
    im['z'] = im.CenterZ3.astype(float) + im.ImageFocusPosition.astype(float)
    im = im.rename(columns = {'ID':'cell'})
    uni = list(im.CombinedTime.unique())
    im['frame'] = [uni.index(x) for x in im.CombinedTime.to_list()]
    im = im.sort_values('CombinedTime').reset_index(drop = True)
    result = tracking_track(im)

list(range(len(im.CombinedTime.unique())))
uni = list(im.CombinedTime.unique())
[uni.index(x) for x in im.CombinedTime.to_list()]
