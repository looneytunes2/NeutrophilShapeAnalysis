
import dataclasses
import numpy as np
import pandas as pd
import re
import tifffile
import skimage.measure
from scipy import interpolate
from scipy.ndimage import affine_transform
from sklearn.linear_model import LinearRegression
from ..aicssegmentation.core.utils import hole_filling
from .persistence_activity import get_pa, DA_3D
from scipy.spatial.transform import Rotation as R
from ..config.models import Config

def running_mean_withna(x, N):
    means = []
    for i, r in enumerate(x):
        if np.isnan(r):
            means.append(np.nan)
        elif i<N:
            #get the window to average
            wind = x[:int(i+1)]
            #remove nan
            wind = wind[~np.isnan(wind)]
            #get average
            means.append(np.mean(wind))
        else:
            #get the indicies around the target value
            first = i - N//2+N%2
            second = first + N
            wind = x[first:second]
            #remove nan
            wind = wind[~np.isnan(wind)]
            #get average
            means.append(np.mean(wind))

    return np.array(means)



def get_consecutive_timepoints(
        df, #dataframe
        column: str, #string column to get consecutive timepoints from
        interval: int, #expected interval of "column"
        ):
    #sort the dataframe based on the column
    df_ = df.copy()
    df_sorted = df_.sort_values(column).reset_index(drop = True)
    #get differences over the column
    diff = df_sorted[column].diff()
    #create a list of all the places with time jumps starting with 0
    difflist = [0]
    difflist.extend(diff[diff>interval].index.to_list())
    if difflist[-1] < len(df_sorted):
        difflist.append(len(df_sorted))
    #make a list of lists with the indices of consecutive time points
    runs = [list(range(difflist[x], difflist[x+1])) for x in range(len(difflist)-1)]
    return df_sorted, runs
    

def get_consecutive_transitions(
        cell, #a dataframe with info for a single cell including a "real_time" column
        ):
    #sort data and get continuous transitions in order
    cell = cell.sort_values('real_time').reset_index(drop = True)
    ### identify indicies where the time_elapsed doesn't match the
    ### change in cumulative time, these are data gaps
    gap_mask = cell.cumulative_time.diff() != cell.time_elapsed
    gaps_inds = cell[gap_mask].index.to_list()
    if gaps_inds[-1] < len(cell):
        gaps_inds.append(len(cell))
    #make a list of lists with the indices of consecutive time points
    runs = [list(range(gaps_inds[x], gaps_inds[x+1])) for x in range(len(gaps_inds)-1)]
    return cell, runs



def get_smooth_trajectory(
    df: pd.DataFrame,
    smooth_factor: float,
    time_interval: float,
    ):
    """
    Take a dataframe with continuous timepoints and generate a smoothened
    trajectory and trajectory-related data. 
    """

    # set the k order for interpolation to the max possible
    if len(df) < 6:
        kay = len(df)-1
    else:
        kay = 5

    # do speed and trajectory stuff
    pos = df[['x_raw', 'y_raw', 'z_raw']]
    dupes = pos[pos.duplicated()].index.tolist()
    if bool(dupes):
        ######### FIND CELL TRAJECTORY AND EULER ANGLES ################
        # drop dupes before processing
        pos_drop = pos.drop(dupes, axis=0)
        # if dropping the duplicates leads to less that three positions,
        # just continue with the duplicates but don't smoothen
        if pos_drop.shape[0] < 3:
            traj = pos.to_numpy().copy()
            possmo = pos.to_numpy().copy()
        else:
            # get trajectories without the duplicates
            tck, u = interpolate.splprep(
                pos_drop.to_numpy().T, k=kay, s=smooth_factor)
            yderv = interpolate.splev(u, tck, der=1)
            # get smoothened trajectory
            traj = np.vstack(yderv).T
            # get smoothened position
            ysmo = interpolate.splev(u, tck, der=0)
            possmo = np.vstack(ysmo).T
            # re-insert duplicate row that was dropped
            for d, dd in enumerate(dupes):
                traj = np.insert(traj, dd, traj[dd-1, :], axis=0)
                possmo = np.insert(
                    possmo, dd, possmo[dd-1, :], axis=0)

    else:
        ######### FIND CELL TRAJECTORY AND EULER ANGLES ################
        # no duplicate positions
        # interpolate and get tangent at midpoint
        tck, b = interpolate.splprep(
            pos.to_numpy().T, k=kay, s=smooth_factor)
        yderv = interpolate.splev(b, tck, der=1)
        traj = np.vstack(yderv).T
        # get smoothened trajectory
        ysmo = interpolate.splev(b, tck, der=0)
        possmo = np.vstack(ysmo).T

    ## before we start adding to the dataframe, remove everything but identifiers
    df = df[['cell','time']]

    #### add smoothened positions and trajectory
    #normalize trajectory first
    unit_traj = traj / np.linalg.norm(traj, axis = 1, keepdims = True)
    df['Trajectory_Vec_X'] = unit_traj[:,0]
    df['Trajectory_Vec_Y'] = unit_traj[:,1]
    df['Trajectory_Vec_Z'] = unit_traj[:,2]
    df['x'] = possmo[:, 0]
    df['y'] = possmo[:, 1]
    df['z'] = possmo[:, 2]
    # add previous and next trajectory rows
    df['Prev_Trajectory_Vec_X'] = df['Trajectory_Vec_X'].shift()
    df['Prev_Trajectory_Vec_Y'] = df['Trajectory_Vec_Y'].shift()
    df['Prev_Trajectory_Vec_Z'] = df['Trajectory_Vec_Z'].shift()
    df['Next_Trajectory_Vec_X'] = df['Trajectory_Vec_X'].shift(-1)
    df['Next_Trajectory_Vec_Y'] = df['Trajectory_Vec_Y'].shift(-1)
    df['Next_Trajectory_Vec_Z'] = df['Trajectory_Vec_Z'].shift(-1)
    # calculate all turn angles between previous and current frames
    df['Turn_Angle'] = angle3D(df[['Trajectory_Vec_X','Trajectory_Vec_Y','Trajectory_Vec_Z']].values,
                    df[['Prev_Trajectory_Vec_X','Prev_Trajectory_Vec_Y','Prev_Trajectory_Vec_Z']].values,)

    ############## Bayesian persistence and activity #################
    persistence, activity, speed = get_pa(df, time_interval)
    df['persistence'] = np.concatenate(
        [np.array([np.nan]*2), persistence])
    df['activity'] = np.concatenate(
        [np.array([np.nan]*2), activity])
    df['speed'] = np.concatenate([np.array([np.nan]), speed])

    # add directional autocorrelations
    df['directional_autocorrelation'] = DA_3D(
        df[['x', 'y', 'z']].to_numpy())
    
    return df

def smooth_trajectory_wrapper(args):
    return get_smooth_trajectory(*args)


#get distance between two points in 3d
def dist_nd(p1,p2):
    return np.sqrt(np.sum((p2-p1)**2))


def get_pc_distance(
        df,
        config,
        n_timepoints = 1,
        ):
    df = df.copy()
    npcs = config.common.npcs
    time_interval = config.im_params.time_interval
    pc_cols = [f'PC{n+1}' for n in range(npcs)]
    dflist = []
    for cell, celldf in df.groupby('CellID'):
        celldf, runs = get_consecutive_timepoints(celldf, 'time', time_interval)
        for r in runs:
            if len(r)<n_timepoints:
                continue
            rundf = celldf.iloc[r]
            points = rundf[pc_cols].to_numpy()
            #empty array to fill with distances
            pc_dists = np.full(len(points), np.nan)
            #get start and end points and measure distance
            start_points = points[:-n_timepoints]
            end_points = points[n_timepoints:]
            pc_dists[n_timepoints:] = dist_nd(start_points, end_points)

            rundf['PC_Distance'] = pc_dists
            dflist.append(rundf)
    
    return pd.concat(dflist, ignore_index = True)



#project vector a onto vector b
def project_vector(a, b):
    b_norm_sq = np.dot(b, b)
    if b_norm_sq == 0:
        raise ValueError("Cannot project onto a zero vector.")
    projection = (np.dot(a, b) / b_norm_sq) * b
    return projection


### ensure all vectors are pointed in a similar direction
def remove_vector_flips(
        vectors: np.array, # (N,3)
        ):
    # dot products between consecutive vectors
    dots = np.sum(vectors[:-1] * vectors[1:], axis=1)
    # get signs
    dot_signs = np.where(dots < 0, -1, 1)
    # add first position and get cum prod
    s = np.concatenate(([1], np.cumprod(dot_signs)))
    # correct signs of the actual vectors
    vecs_aligned = vectors * s[:, np.newaxis]
    return vecs_aligned

#### sorts data for an individual cell and adds the raw speed projected onto
#### the smoothened trajectory
def project_raw_smooth(
        df, #dataframe of a cell with raw and smoothened x,y,z positions
        image_interval, #time between frames
        timespan, #integer number of image intervals to calculate velocity 
        ):
    
    cell, runs = get_consecutive_timepoints(df, 'time', image_interval)
    #iterate through consecutive frames
    speeds = []
    velocities = []
    for r in runs:
        rundf = cell.iloc[r].copy().reset_index(drop=True)
        for i in range(len(rundf)):
            #get rows of current and previous timepoints
            cur = rundf.iloc[i]
            prev = rundf.shift(-timespan).iloc[i]
            #add nan if there's no data for this timepoint
            if all(prev.isna()):
                velocities.append(np.nan)
                speeds.append(np.nan)
            else:
                #get smooth and raw trajectory vectors
                smoothvec = np.array([cur.x-prev.x, cur.y-prev.y, cur.z-prev.z])
                rawvec = np.array([cur.x_raw-prev.x_raw, cur.y_raw-prev.y_raw, cur.z_raw-prev.z_raw])
                #project the raw vector and get the distance
                rawproj = project_vector(rawvec, smoothvec)
                projdist = dist_nd([0,0,0], rawproj)
                if (smoothvec[0]>0) and (rawproj[0]<0):
                    projdist *= -1
                elif (smoothvec[0]<0) and (rawproj[0]>0):
                    projdist *= -1
                velocities.append(projdist/(image_interval*timespan))
                speeds.append(dist_nd([0,0,0], smoothvec)/(image_interval*timespan))
            
    cell.loc[:,f'velocity_span_{timespan}'] = velocities
    cell.loc[:,f'speed_span_{timespan}'] = speeds
    
    return cell
            
      
            
def filename_match_llscellid(
        cellid, #CellID of cell in question
        lst, #list of file names
        ):
    movie = '_'.join(cellid.split('_')[:-1])
    cellinmovie = cellid.split('_')[-1]
    filematches = []
    for l in lst:
        if movie in l:
            if re.search(r'\d+', l.split('Subset-')[-1])[0] == cellinmovie:
                filematches.append(l)
    return filematches


def smoothen_aer(
        cell, #dataframe with 'time', 'aer', and 'time_elapsed' columns for a single cell
        s = 30, #splprep s factor
        ):
    #ensure the cell is in time order
    cell_ = cell.sort_values('real_time').reset_index(drop=True)
    #get rid of NA in aer which will ruin cumulative sums etc.
    cellnona = cell_[~cell_.aer.isna()].copy()
    #get area enclosed from aer
    cellnona['area_enclosed'] = cellnona.aer * cellnona.time_elapsed.values
    #### weight the points near gaps more
    _, runs = get_consecutive_transitions(cellnona)
    #get the indicies before and after jumps
    gaps = np.array([[r[0],r[-1]] for r in runs]).flatten()
    #add the weights
    w = np.ones(cellnona.shape[0])
    w[gaps] = 3

    ####interpolation method
    #interpolate for smoothening
    tck, u = interpolate.splprep(np.array((cellnona.real_time.values,
                                            cellnona.area_enclosed.cumsum().values)),
                                    k=3, s = s, w = w)#k=1, s=2, w = w)
    #get the derivative of the smoothened curve
    dx, dy = interpolate.splev(u, tck, der=1)
    #get derivative in correct units of time (area enclosed / sec)
    deriv = dy/dx#(cellnona.time.max() - cellnona.time.min())

    return pd.Series(deriv, index=cellnona.index), tck, w




####### threshold smoothened area enclosing rate 
def get_aer_state(
        df, #a dataframe with 'real_time', 'aer', and 'time_elapsed' columns
        whichpcs, #(x,y) PCs that define the CGPS of this cycle
        config,
        group_factor, #factor column to group dataframe by for determining consecutive transitions
        ):


    ### get threshold info from config
    thresh_dict = config.db_params.cycle_thresh[whichpc_string(whichpcs)]
    low_thresh = thresh_dict['low']
    negative_thresh = -low_thresh
    high_thresh = thresh_dict['high']
    #thresholds for aer in decreasing order
    thresholds = [high_thresh, low_thresh, negative_thresh] 
    #labels for state above each threshold, last is default
    state_labels = ['high', 'low', 'zero', 'neg']
    #splprep smoothing factor
    smooth = thresh_dict['smooth'] 
    
    df_ = df.reset_index(drop = True)

    ### get smoothened aer
    smooth_aer = []
    for idd, cell in df_.groupby(['Treatment', group_factor]):
        s_a ,_,_ = smoothen_aer(cell, s=smooth)
        smooth_aer.extend(s_a.to_list())
    smooth_array = np.array(smooth_aer)
    #threshold with np.select
    threshs = [smooth_array >=x for x in thresholds]
    statethresh = np.select(threshs, state_labels[:-1], default = state_labels[-1])
    #add new values to dataframe
    df_.loc[:,'aer_smooth'] = smooth_array
    df_.loc[:,'aer_state'] = statethresh

    return df_



##### assign unique identifiers to a dataframe with aer_state chunks
def get_aer_state_chunk_ids(
    df, # dataframe with 'aer_state' and 'chunk_id' columns
    group_factor = 'CellID', #factor that separates group of interest
    ):
    ### ensure the dataframe is sorted by cell and time
    df_ = df.sort_values(['Treatment',group_factor,'real_time']).reset_index(drop=True)
    ### get where aer state changes or cell changes
    run_change = pd.Series(False, index=df_.index)
    for col in ['aer_state','Treatment',group_factor]:
        run_change |= (df_[col] != df_[col].shift())
    runs = run_change.cumsum()

    ### add chunk_id
    df_['chunk_id'] = runs

    #### add chunk run info
    g = df_.groupby('chunk_id')['time_elapsed']
    df_['chunk_run_time'] = g.cumsum()
    df_['chunk_run_time_norm'] = df_['chunk_run_time'] / g.transform('sum')

    return df_


######### 
def get_observed_aer_state_chunk_starts_stops(
    df, # dataframe with 'aer_state' and 'chunk_id' columns
    group_factor = 'CellID', #factor that separates group of interest
    ):

    """
    function to detect OBSERVED state starts

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'aer_state' and 'chunk_id' columns.

    Returns
    -------
    observedstarts : list
        List of chunk_ids in the input dataframe where observed states start.
    observedstops : list
        List of chunk_ids in the input dataframe where observed states stop.
    Other parameters
    ----------------
    group_factor : str, optional
        Factor that separates groups of interest for creating the
        mesh, default is 'CellID'.

    Notes
    -----

    """
    
    df = df.sort_values(['chunk_id']).reset_index(drop=True)

    observedstarts = []
    observedstops = []
    for _, cell in df.groupby(['Treatment',group_factor]):
        #get all cell states
        state = cell.aer_state

        # starts: state changed from previous row (excluding the very first row)
        changed_fwd = state != state.shift(1)
        changed_fwd.iloc[0] = False
        observedstarts.extend(cell.chunk_id[changed_fwd].tolist())

        # stops: state changes vs next row (excluding the very last row)
        changed_bwd = state != state.shift(-1)
        changed_bwd.iloc[-1] = False
        observedstops.extend(cell.chunk_id[changed_bwd].tolist())

    ### return chunk_ids where states start and stop
    return observedstarts, observedstops


def get_whole_chunk_df(
        df,
        group_factor,
        ):
    """
    run get_observed_aer_state_chunk_starts_stops and use them to return
    a dataframe with only whole chunk_ids
    df, # dataframe with 'aer_state' and 'chunk_id' columns
    group_factor = 'CellID', #factor that separates group of interest
    """
    observedstarts, observedstops = get_observed_aer_state_chunk_starts_stops(
        df,
        group_factor,
        )
    whole_chunks = set(observedstarts) & set(observedstops)
    return df[df.chunk_id.isin(whole_chunks)].copy()

######## calculate average metrics over time in minutes
def calculate_rates(
        df, # dataframe of transitions with 'time' in seconds
        group_factor, # factor column
        rate_cols = ['aer','angular_velocity','pc_speed'], #iterable with column names of rate quantities to fit with lr
        ):
    #make sure data is sorted by time
    time_col = 'real_time' if 'real_time' in df.columns else 'time'
    df, runs = get_consecutive_transitions(df)
    dropdf = df[~df[rate_cols[0]].isna()]
    ### make dict to update
    rate_fit_dict = {
        'Treatment': df.iloc[0].Treatment,
        group_factor: df.iloc[0][group_factor],
    }
    for rc in rate_cols:
        #get value per time instead of per sec
        time_value_col = 'time_interval_'+rc
        value_cumsum_col = time_value_col + '_cumsum'
        dropdf[time_value_col] = dropdf[rc].values*dropdf['time_elapsed'].values
        dropdf[value_cumsum_col] = dropdf[time_value_col].cumsum()
        #linear regression
        total_change = 0
        total_time = 0
        for r in runs:
            total_change += dropdf[value_cumsum_col].iloc[r[-1]] - dropdf[value_cumsum_col].iloc[r[0]]
            total_time += dropdf[time_col].iloc[r[-1]] - dropdf[time_col].iloc[r[0]]
        avg_rate = total_change / total_time
        #add to dictionary of metrics
        rate_fit_dict.update({
            rc+'_avg': avg_rate,
            })
    
    return rate_fit_dict


######## perform regression on metrics over time in minutes
def fit_rates_linear(
        df, # dataframe with 'time' in seconds
        rate_cols, #iterable with column names of rate quantities to fit with lr
        ):
    #make sure data is sorted by time
    time_col = 'real_time' if 'real_time' in df.columns else 'time'
    df = df.sort_values(time_col).reset_index(drop=True)
    ### make dict to update
    rate_fit_dict = {}
    for rc in rate_cols:
        #drop na
        dropdf = df[~df[rc].isna()].copy()
        #get value per time instead of per sec
        dropdf['value_per_time'] = dropdf[rc].values*dropdf['time_elapsed'].values
        #linear regression
        reg = LinearRegression().fit(dropdf[time_col].values.reshape(-1, 1),
                                        dropdf.value_per_time.cumsum().values.reshape(-1, 1))
        resid = reg.score(dropdf[time_col].values.reshape(-1, 1),
                                dropdf.value_per_time.cumsum().values.reshape(-1, 1))
        rate_fit_dict.update({
            rc+'_coeff': reg.coef_[0][0],
            rc+'_fit': resid,
            })
    
    return rate_fit_dict



#### bootstrap a confidence interval similar to seaborn
def bs_ci(values, #distribution to sample from
          iterations = 1000, #how many times to sample
          ):
    
    if type(values) != np.ndarray:
        values = np.array(values)
    #remove nan
    values = values[~np.isnan(values)]
    leng = len(values)
    iters = np.zeros((iterations))
    for i in range(iterations):
        sample_inds = np.random.randint(0,leng,leng)
        sample = values[sample_inds]
        iters[i] = sample.mean()
    #calculate 95% percentile interval
    lower = np.percentile(iters, 2.5)
    upper = np.percentile(iters, 97.5)
    
    return lower, upper



###### get raw intensity features from a "seg" mask
###### stolen from Allen Institute for Cell Science
def get_intensity_features(img, seg):
    features = {}
    input_seg = seg.copy()
    input_seg = (input_seg>0).astype(np.uint8)
    input_seg_lcc = skimage.measure.label(input_seg)
    for mask, suffix in zip([input_seg, input_seg_lcc], ['', '_lcc']):
        values = img[mask>0].flatten()
        if values.size:
            features[f'intensity_mean{suffix}'] = values.mean()
            features[f'intensity_std{suffix}'] = values.std()
            features[f'intensity_1pct{suffix}'] = np.percentile(values, 1)
            features[f'intensity_99pct{suffix}'] = np.percentile(values, 99)
            features[f'intensity_max{suffix}'] = values.max()
            features[f'intensity_min{suffix}'] = values.min()
        else:
            features[f'intensity_mean{suffix}'] = np.nan
            features[f'intensity_std{suffix}'] = np.nan
            features[f'intensity_1pct{suffix}'] = np.nan
            features[f'intensity_99pct{suffix}'] = np.nan
            features[f'intensity_max{suffix}'] = np.nan
            features[f'intensity_min{suffix}'] = np.nan
    return features


#### fill 2D holes sequentially
def twodholefill(thresh, hole_min, hole_max):
    YZ = thresh.swapaxes(0,2)
    YZ_fill = hole_filling(YZ, hole_min, hole_max, fill_2d=True)
    YZrev = YZ_fill.swapaxes(2,0)
    XZ = YZrev.swapaxes(0,1)
    XZ_fill = hole_filling(XZ, hole_min, hole_max, fill_2d=True)
    XZrev = XZ_fill.swapaxes(1, 0)
    XY = hole_filling(XZrev, hole_min, hole_max, fill_2d=True)
    return XY




### angle between two vectors in degrees
def angle3D(v1, v2):
    """
    v1, v2: (N, 3) numpy arrays
    Returns: (N,) array of angles in degrees between corresponding rows
    """
    d = np.sum(v1 * v2, axis=1)
    e1 = np.linalg.norm(v1, axis=1)
    e2 = np.linalg.norm(v2, axis=1)
    d = np.clip(d / (e1 * e2), -1, 1)

    return np.degrees(np.arccos(d))



### align a vector to the x axis and get the euler rotations to do so
def align_vec_to_xaxis_euler(
        vec, #iterable in XYZ order
        return_rotation_object:bool = False, #whether to return the scipy rotation object
        ):
    #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    xaxis = np.array([[1,0,0], [0,1,0], [0,0,1]]).astype('float64')
    upnorm = np.cross(vec,[1,0,0])
    sidenorm = np.cross(upnorm, vec)
    current_vec = np.stack((vec, sidenorm, upnorm), axis = 0)
    rotationthing = R.align_vectors(xaxis, current_vec)
    #below is actual rotation matrix if needed
    #rot_mat = rotationthing[0].as_matrix()
    rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)
    euler_angles = np.array([rotthing_euler[0], rotthing_euler[1], rotthing_euler[2]])
    
    return (euler_angles, rotationthing) if return_rotation_object else euler_angles 


### takes an interable and reformats to a PC string
def whichpc_string(whichpcs):
    return '-'.join(f"PC{w}" if w>0 else f"PC{abs(w)}_abs" for w in whichpcs)


#### takes integer number of seconds and returns MM:SS for movies
def format_seconds(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02}:{secs:02}"


# Permutation: numpy axis order is ZYX, scipy/rotation is XYZ
P = np.array([[0, 0, 1],   # Z → X
              [0, 1, 0],   # Y → Y
              [1, 0, 0]])  # X → Z
def to_numpy_basis(mat_xyz):
    """Convert an XYZ rotation matrix to ZYX (numpy) basis."""
    return P @ mat_xyz @ P.T   # P.T == P here since P is symmetric


### rotate LLS image to shape alignment frame
def align_cropped_image(
        cellser: pd.Series,
        config: Config,
        down_factor: int = 1,
        im_type: str = 'raw',
        ):
    #get directory
    configdict = dataclasses.asdict(config)
    localdir = configdict['experiment'][cellser.Experiment]['localdir']
    imdir = localdir / 'processed_images'
    #open image
    if cellser.Experiment == 'lls':
        im = tifffile.imread(imdir.joinpath(cellser.cell + f'_{im_type}.ome.tiff'))
    else:
        im = tifffile.imread(imdir.joinpath(cellser.cell + f'_{im_type}.tiff'))
        #scale these images to so that z is proportional to xy
        xyres = config.im_params.xyres
        zstep = config.im_params.zstep
        im = skimage.transform.rescale(im, [zstep/xyres, 1, 1], preserve_range=True)

    ## add a dimension if there's only 3
    if len(im.shape)<4:
        im = im[np.newaxis, ...]
    #optionally shrink image
    if down_factor>1:
        downlist = []
        for c in range(im.shape[-4]):
            downlist.append(skimage.transform.rescale(im[c], 1/down_factor, preserve_range=True))
        im = np.stack(downlist)

    # get image centroid (which should be cell centroid) to calculate offset for rotation
    # also get some array shape info
    center = (np.array(im.shape[-3:])-1)/2
    imshape = np.array(im.shape)
    maxdim = np.repeat(np.max(imshape[-3:])*1.5,3).astype(np.uint16)
    maxcenter = (maxdim - 1)/ 2
    

    ###### get rotation matrix
    euler_cols = [x for x in cellser.index if 'Euler' in x]
    trajectory_eulers = cellser[euler_cols].values
    matrix = to_numpy_basis(R.from_euler('xyz', trajectory_eulers,  degrees=True).as_matrix()).T
    offset = center - matrix @ maxcenter

    rotated_img = np.zeros(np.insert(maxdim, 0, imshape[-4]))
    ### rotate each channel individually
    for c in range(imshape[0]):
        ### apply rotation
        rotated_img[c] = affine_transform(
            im[c],
            matrix,
            offset = offset,
            output_shape = maxdim,
            order=0,
            mode='constant',
            cval=0,
        )
    
    return rotated_img #CZYX




#wrapper for get_shape_info_nonuc for imap
def align_cropped_image_imap(args):
    return align_cropped_image(*args)


## quick image normalization
def normalize_channel(ch, pmin=0.5, pmax=99.5):
    lo, hi = np.percentile(ch, (pmin, pmax))
    ch = np.clip((ch - lo) / (hi - lo), 0, 1)
    return ch

## convert two-channel image into rgb
def multichannel_to_rbg(
        img,
        color1: np.array,
        color2: np.array,
        ):
    
    ch1 = img[0].astype(float)
    ch2 = img[1].astype(float)
    
    ## adjust channel values and 
    ch1_n = normalize_channel(ch1)
    ch2_n = normalize_channel(ch2)

    
    rgb = (ch1_n[..., np.newaxis] * color1) + (ch2_n[..., np.newaxis] * color2)
    rgb = np.clip(rgb, 0, 1)
    
    rgb_8bit = (rgb * 255).astype(np.uint8)
    
    return rgb_8bit

def multichannel_to_rbg_imap(args):
    return multichannel_to_rbg(*args)



p_ax = ['Major','Median','Minor']
dim = ['X','Y','Z']
from . import shparam_mod
def measure_axes(cellpath, xyres, zstep):
    #open image
    im = tifffile.imread(cellpath)
    if len(im.shape)>3:
        im = im[0]
    #get vectors
    cell_evecs = shparam_mod.extract_object_principal_axes(
        im,
        xyres,
        zstep,
        )
    #unpack vectors into a dictionary
    p_ax_dict = {
        f'Cell_{axis}_Axis_Vec_{d}': cell_evecs[e, v]
        for e, axis in enumerate(p_ax)
        for v, d in enumerate(dim)
        }
    return p_ax_dict

def measure_axes_imap(args):
    return measure_axes(*args)


from .shtools_mod import read_polydata
def measure_mesh_axes(meshfl,):

    mesh = read_polydata(meshfl)

    cell_evecs = shparam_mod.extract_mesh_principal_axes(
        mesh
        )
    #unpack vectors into a dictionary
    p_ax_dict = {
        f'Cell_{axis}_Axis_Vec_{d}': cell_evecs[e, v]
        for e, axis in enumerate(p_ax)
        for v, d in enumerate(dim)
        }
    return p_ax_dict



