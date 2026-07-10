


import numpy as np
import pandas as pd
from neutrophil_shape.CustomFunctions import utils
from neutrophil_shape.config.loader import load_config
import tifffile
import multiprocessing


config = load_config(microscope_type='lls')
config._alignment = 'trajectory'
time_interval = config.im_params.time_interval
whichpcs = (1,2)
ntrans = config.db_params.ntrans
nbins = config.db_params.nbins
origins = config.db_params.origins
basedir = config.common.savedir
datadir = basedir.joinpath('shape_data')
dbdir = basedir.joinpath('detailed_balance')
dbbsdir = dbdir.joinpath('separatedatabs')
serverdir = config.experiment.lls.serverdir
localdir = config.experiment.lls.localdir
posdir = localdir / 'position_info'
singlecelldir = localdir / 'singlecells'
xyres = config.im_params.xyres



FullFrame = pd.read_csv(datadir.joinpath('All_Data_with_CGPS_bins.csv'), index_col = 0)
FullFrame['real_time'] = FullFrame.time.copy()

#open aers previously calculated
allaers = pd.read_csv(dbdir.joinpath(utils.whichpc_string(whichpcs) + '_raw_transition_aer_cf.csv'), index_col = 0)

#merge aer and cf info
TotalFrame = pd.merge(FullFrame, allaers,on=['CellID','real_time','frame'],how='left')
TotalFrame = TotalFrame.sort_values(['CellID','time'])



def pad_3d(
        im,
        pad_shape,
        ):
    im_shape = np.array(im.shape)
    pads = pad_shape - im_shape
    
    x1 = pads[-1] // 2
    x2 = pad_shape[-1] - x1 - im_shape[-1]
    y1 = pads[-2] // 2
    y2 = pad_shape[-2] - y1 - im_shape[-2]
    z1 = pads[-3] // 2
    z2 = pad_shape[-3] - z1 - im_shape[-3]
    
    return np.pad(im, pad_width = ((z1,z2), (y1,y2), (x1,x2)), mode = 'constant')


    
# cell = TotalFrame.iloc[0].cell
# trajectory_eulers = TotalFrame[['Euler_angles_X', 'Euler_angles_Y', 'Euler_angles_Z']].iloc[0].values
# normal_angle = TotalFrame['Width_Rotation_Angle'].iloc[0]

# rotated = align_raw_image(
#         cell,
#         trajectory_eulers,
#         normal_angle,
#         config,
#         )



#### iterate through all LLS cells and make 1/2 scale aligned videos

for cellid, celldf in TotalFrame.groupby('CellID'):
    celldf = celldf.sort_values('time').reset_index(drop = True)
    # imlist = []
    arglist = []
    for r, row in celldf.iterrows():
        trajectory_eulers = row[['Euler_angles_X', 'Euler_angles_Y', 'Euler_angles_Z']].values.astype(float)
        normal_angle = row['Width_Rotation_Angle']
        arglist.append((
                row.cell,
                trajectory_eulers,
                normal_angle,
                config,
                ))
        # imlist.append(rotated)
        
    with multiprocessing.Pool(processes = 60) as pool:
        results = list(pool.imap(utils.align_raw_image_imap, arglist))
    
    ### pad images and concatenate into one array
    # get dimensions of all images
    all_shapes = np.array([i.shape for i in results])
    # get largest dimensions
    pad_shape = np.max(all_shapes, axis = 0)
    #fill empty array with padded, rotated images
    full_im = np.zeros(np.insert(pad_shape, 0, len(all_shapes)))
    for i, im in enumerate(results):
        for c in range(pad_shape[-4]):
            full_im[i,c] = pad_3d(
                im[c],
                pad_shape[-3:])
   
    ## trim down the image to minimal size necessary
    thresh = 200
    wh = np.where(full_im[:,0] > thresh)
    mins = np.min(wh, axis = 1)
    maxs = np.max(wh, axis = 1)
    maxdiffs = pad_shape[-3:] - maxs[-3:]
    crop_int = np.concatenate((mins[-3:], maxdiffs)).min() - 1
    
    cropped = full_im[:,:,
                      
                      crop_int:pad_shape[-3]-crop_int,
                      crop_int:pad_shape[-2]-crop_int,
                      crop_int:pad_shape[-1]-crop_int,
                      ]
            
            
    tifffile.imwrite(singlecelldir.joinpath(cellid, cellid+'_traj_aligned.ome.tiff'),
                     cropped,
                     metadata={'axes': 'TCZYX'})
            
            
    

    
    
    