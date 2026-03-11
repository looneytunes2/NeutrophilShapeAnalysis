
import math
import numpy as np
import re
import skimage.measure
from scipy import interpolate
from sklearn.linear_model import LinearRegression
from aicssegmentation.core.utils import hole_filling
from scipy.spatial.transform import Rotation as R

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
    df_sorted = df.sort_values(column).reset_index(drop = True)
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


#get distance between two points in 3d
def dist_3d(p1,p2):
    return np.sqrt(np.sum((p2-p1)**2))


#project vector a onto vector b
def project_vector(a, b):
    b_norm_sq = np.dot(b, b)
    if b_norm_sq == 0:
        raise ValueError("Cannot project onto a zero vector.")
    projection = (np.dot(a, b) / b_norm_sq) * b
    return projection


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
                projdist = dist_3d([0,0,0], rawproj)
                if (smoothvec[0]>0) and (rawproj[0]<0):
                    projdist *= -1
                elif (smoothvec[0]<0) and (rawproj[0]>0):
                    projdist *= -1
                velocities.append(projdist/(image_interval*timespan))
                speeds.append(dist_3d([0,0,0], smoothvec)/(image_interval*timespan))
            
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




def get_aer_state(
        cell, #dataframe of the cell to be thresholded
        time_interval, #imaging time interval
        derivthresh = 0.0007, #derivative threshold to call increasing or decreasing aer
        ):
    
    #ensure the cell is in time order
    cell = cell.sort_values('time').reset_index(drop=True)
    #get rid of NA in aer which will ruin cumulative sums etc.
    cellnona = cell[~cell.aer.isna()].copy()
    #get area enclosed from aer
    cellnona['area_enclosed'] = cellnona.aer*time_interval
    #### weight the points near gaps more
    _, runs = get_consecutive_transitions(cellnona)
    #get the indicies before and after jumps
    gaps = np.array([[r[0],r[-1]] for r in runs]).flatten()
    #add the weights
    w = np.ones(cellnona.shape[0])
    w[gaps] = 3


    ####interpolation method
    #interpolate for smoothening
    tck, u = interpolate.splprep(np.array((cellnona.time.values,
                                           cellnona.area_enclosed.cumsum().values)),
                                 k=3, s = 15, w = w)#k=1, s=2, w = w)
    #get the derivative of the smoothened curve
    dx, dy = interpolate.splev(u, tck, der=1)
    #get derivative in correct units of time (area enclosed / sec)
    deriv = dy/(cellnona.time.max() - cellnona.time.min())

    #threshold with np.select
    threshs = [dy>=derivthresh, dy<=-derivthresh]
    choices = ['increasing', 'decreasing']
    statethresh = np.select(threshs, choices, default = 'unchanging')
    #add new values to dataframe
    cell.loc[cellnona.index,'aer_smooth'] = deriv
    cell.loc[cellnona.index,'aer_state'] = statethresh

    return cell, tck, w



######## perform regression on AE over time in minutes
def fit_rates_linear(
        df, # dataframe with 'time' in seconds
        time_interval, #frame rate of 'time' data in df
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
        #get value per time interval instead of per sec
        dropdf['value_per_time'] = dropdf[rc]*time_interval
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
def angle3D(a1, b1, c1, a2, b2, c2):
    d = ( a1 * a2 + b1 * b2 + c1 * c2 )
    e1 = math.sqrt( a1 * a1 + b1 * b1 + c1 * c1)
    e2 = math.sqrt( a2 * a2 + b2 * b2 + c2 * c2)
    d = d / (e1 * e2)
    if d>1:
        d = 1
    elif d<-1:
        d = -1
    A = math.degrees(math.acos(d))
    return A



### align a vector to the x axis and get the euler rotations to do so
def align_vec_to_xaxis_euler(
        vec, #iterable in XYZ order
        return_rotation_object:bool = False, #whether to return the scipy rotation object
        ):
    #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    xaxis = np.array([[1,0,0], [0,1,0], [0,0,1]]).astype('float64')
    upnorm = np.cross(vec,[1,0,0])
    sidenorm = np.cross(vec,upnorm)
    current_vec = np.stack((vec, sidenorm, upnorm), axis = 0)
    rotationthing = R.align_vectors(xaxis, current_vec)
    #below is actual rotation matrix if needed
    #rot_mat = rotationthing[0].as_matrix()
    rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)
    euler_angles = np.array([rotthing_euler[0], rotthing_euler[1], rotthing_euler[2]])
    
    return (euler_angles, rotationthing) if return_rotation_object else euler_angles 