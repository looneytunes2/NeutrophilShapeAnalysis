# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 15:27:28 2025

@author: Aaron
"""

import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from aicssegmentation.core.MO_threshold import MO
from aicsimageio.readers.tiff_reader import TiffReader
from CustomFunctions import utils
import skimage.measure
from pathlib import Path

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


whichpcs = [1,7]
time_interval = 5
mind = 'E:/Aaron/Combined_37C_Confocal_PCA_s5_LLS_Apply/'
datadir = mind + 'Data_and_Figs/'
randir = mind + 'random/'
moviedir = 'E:/Aaron/random_lls/singlecells/'


runlength = 13 #how many frames per video
chunksize = 62 #pixel size in each dimension to crop
chunknum = 4 #square root of the number of chunks to show
chunksq = chunknum ** 2 #number of chunks to show

FullFrame = pd.read_csv(datadir + 'All_Data_with_CGPS_bins.csv', index_col = 0)
# open aers
allaers = pd.read_csv(randir+f'PC{whichpcs[0]}-PC{whichpcs[1]}_raw_transition_aer_cf.csv', index_col = 0)

#merge aer and cf info
TotalFrame = FullFrame.merge(allaers[['aer','angular_velocity','cell']],left_on='cell',right_on='cell')
TotalFrame = TotalFrame.sort_values(['CellID','time'])



############ find top 5 minutes with positive and negative AER for each unique cell
topruns = []
bottomruns = []
aertime = np.arange(1,runlength)*time_interval #constant time range based on time window for AER fit
for i, cell in TotalFrame.groupby('CellID'):
    aerruns = [] #list to append dicts
    ######### get AER for every window of the specified length
    cs, runs = utils.get_consecutive_timepoints(cell, 'time', time_interval)
    for r in runs:
        c = cs.iloc[r]
        #only look at runs that are long enough
        if len(c)>=runlength:
            allshifts = np.arange(0,len(c)-runlength)
            stateruns = []
            for n in allshifts:
                #get snippet
                tempc = c.iloc[n:int(n+runlength)].copy()
                #find fit
                aerresid, aercoef = utils.fit_AER(tempc[1:],time_interval,'aer')
                #get relevant info for this stretch
                aerdict = {
                    'CellID':tempc.CellID.iloc[0],
                    'cell':tempc.cell.values,
                    'time_range': tempc.time.values,
                    'aer': aercoef,
                    }
                aerruns.append(aerdict)
    #make dataframe from this cell
    cellaerframe = pd.DataFrame(aerruns)
    #copy this dataframe and find the top runs
    topaerframe = cellaerframe.copy()
    for ae in range(5):
        sortedaers = topaerframe.sort_values('aer')
        #append the top
        temptop = sortedaers.iloc[[-1]]
        topruns.append(temptop)
        #remove any overlapping time windows
        overlaps = [temptop.index[0]]
        overlapmin = min(temptop.time_range.values[0])
        overlapmax = max(temptop.time_range.values[0])
        for oc in topaerframe.itertuples():
            #if there's any overlap append to overlap list
            if any([all((t>=overlapmin,t<=overlapmax)) for t in oc.time_range]):
                overlaps.append(oc.Index)
        #drop the overlaps from the aerframe and repeat
        topaerframe = topaerframe.drop(overlaps)
    
    ### copy the dataframe and find the bottom
    botaerframe = cellaerframe.copy()
    for ae in range(5):
        sortedaers = botaerframe.sort_values('aer')
        #append the top
        tempbot = sortedaers.iloc[[0]]
        bottomruns.append(tempbot)
        #remove any overlapping time windows
        overlaps = [tempbot.index[0]]
        overlapmin = min(tempbot.time_range.values[0])
        overlapmax = max(tempbot.time_range.values[0])
        for oc in botaerframe.itertuples():
            #if there's any overlap append to overlap list
            if any([all((t>=overlapmin,t<=overlapmax)) for t in oc.time_range]):
                overlaps.append(oc.Index)
        #drop the overlaps from the aerframe and repeat
        botaerframe = botaerframe.drop(overlaps)
        
#get the top chunksq number of the top and bottom
toprunframe = pd.concat(topruns).sort_values('aer', ascending=False)[:chunksq].reset_index(drop = True)
toprunframe['topbottom'] = 'top'
bottomrunframe = pd.concat(bottomruns).sort_values('aer')[:chunksq].reset_index(drop = True)
bottomrunframe['topbottom'] = 'bottom'

### combine the dataframes, but keep duplicated indexes for building the image arrays below
topbotframe = pd.concat((toprunframe,bottomrunframe))

######### if you haven't already, build the image arrays of movie snippets
######### cell by cell
if os.path.exists(os.getcwd()+'/topbottom_state_movie_array_data.csv'):
    ###### open data and numpy array
    df = pd.read_csv(os.getcwd()+'/topbottom_state_movie_array_data.csv', index_col = 0)
    
    topchunks = np.load(os.getcwd()+'/topaer_movie_array.npy')
    bottomchunks = np.load(os.getcwd()+'/bottomaer_movie_array.npy')
else:
    chunkinfo = []
    topchunks = np.zeros((chunksq,runlength,chunksize,chunksize,chunksize))
    bottomchunks = np.zeros((chunksq,runlength,chunksize,chunksize,chunksize))
    #start with cell
    for i, cell in topbotframe.groupby('CellID'):

        #open 0.25 size cell movie
        movpath = Path(moviedir, i, i+'_full_movie.ome.tiff')
        cellim = TiffReader(movpath).data[:,0,:,:,:]
        imshape = cellim.shape
        #get the actual frames included in this movie
        framelist = pd.read_csv(moviedir+f'{i}/{i}_framelist.csv', index_col = 0)
        framelist = framelist['0']
        
        for row in cell.itertuples():
            #threshold and crop the cell from the miniature cell movie
            #get the frames in the video of this chunk
            frameind = framelist.index[framelist.isin(row.cell)].to_list()
            runcell = cellim[frameind]
            cropinfo = []
            for rc in runcell:
                mothresh = MO(rc, global_thresh_method = 'triangle', object_minArea = 100)
                im_labeled, n_labels = skimage.measure.label(
                                          mothresh, background=0, return_num=True,  )
            
                im_props = skimage.measure.regionprops(im_labeled)
                tempdf = []
                for count, prop in enumerate(im_props):
                    z,y,x = np.array(prop.centroid)
                    thebox = np.array(prop.bbox)
                    area = prop.area
                    intensity = np.mean(rc[im_labeled==int(count+1)])
                    td = {'cell':count, 'z_min':thebox[0], 'y_min':thebox[1], 
                            'x_min':thebox[2],'z_max':thebox[3], 'y_max':thebox[4], 'x_max':thebox[5],
                           'z':z, 'y':y, 'x': x, 'area':area, 'intensity':intensity}
                    tempdf.append(td)
                tempdf = pd.DataFrame(tempdf).sort_values(['area','intensity'])
                cropinfo.append(tempdf.iloc[-1])
            
            chunktrack = pd.DataFrame(cropinfo)
            avgcent = chunktrack[['z','y','x']].mean().values.astype(np.uint16)
            #make empty array
            empty = np.zeros((runlength,chunksize,chunksize,chunksize))
            #get the actual indices to crop
            cropmins = avgcent-chunksize/2
            cropmins = [max(c,0) for c in cropmins]
            cropmaxs = avgcent+chunksize/2
            cropmaxs = [min(c,cropmaxs[n]) for n,c in enumerate(imshape[-3:])]
            # #ensure you're not cropping outside the image bounds
            # under = np.where(cropmins<0)[0]
            # over = np.where(cropmaxs>imshape[-3:])[0]
            # for o in over:
            #     cropmaxs[o] = imshape[o+1]
            #     cropmins[o] = imshape[o+1]-chunksize
            # for u in under:
            #     cropmins[u] = 0
            #     cropmaxs[u] = chunksize
            cropmins = np.array(cropmins).astype(np.uint16)
            cropmaxs = np.array(cropmaxs).astype(np.uint16)
            
            #add other chunk info
            chunktrack['CellID'] = i
            chunktrack.loc[:,'cell'] = framelist[frameind].to_list()
            chunktrack['aer'] = row.aer
            
            #add an identifier to the chunktrack and append
            chunktrack['chunk_index'] = row.Index
            chunktrack['topbottom'] = row.topbottom
            chunkinfo.append(chunktrack)
            
            #add the actual chunk to a list
            empty[
                :,
                :int(cropmaxs[0]-cropmins[0]),
                :int(cropmaxs[1]-cropmins[1]),
                :int(cropmaxs[2]-cropmins[2])
                ] = runcell[
                :,
                cropmins[0]:cropmaxs[0],
                cropmins[1]:cropmaxs[1],
                cropmins[2]:cropmaxs[2],
                ]
                    
            #insert the cropped movie into the full arrays
            if row.topbottom=='top':
                topchunks[row.Index] = empty
            elif row.topbottom=='bottom':
                bottomchunks[row.Index] = empty

            
            print(f'finished {row.topbottom} chunk {row.Index}')
        print('finished cell '+ i)

        

    df = pd.concat(chunkinfo, ignore_index = True)
    df.to_csv(os.getcwd()+'/topbottom_state_movie_array_data.csv')
    
    np.save(os.getcwd()+'/topaer_movie_array.npy', topchunks)
    np.save(os.getcwd()+'/bottomaer_movie_array.npy', bottomchunks)


#set "thresholds" for flipping the image
anglebins = [-180,-150,-65,65,150,180]
flips = {int(i+1): x for i,x in enumerate(np.arange(-2,3))}


#add aer derivative to the runs
merged = df.merge(TotalFrame[['cell','speed']], on = 'cell')


######### loop through each state and make the movies
for tp in ['top','bottom']:
    curdf = merged[merged.topbottom == tp]
    #get the 
    if tp == 'top':
        curchunks = topchunks.copy()
        ast = 'Highest AER'
        views = ['xz','xy','xz','xy',
                 'xy','xy','xz','xy',
                 'yz','xz','xz','xy',
                 'xy','xz','yz','xy']
    elif tp == 'bottom':
        curchunks = bottomchunks.copy()
        ast = 'Lowest AER'
        views = ['yz','xy','yz','yz',
                 'xy','xy','yz','yz',
                 'xy','xy','yz','yz',
                 'xz','xy','xy','xz']



    #make a movie for each projection
    # for persp in ['xy','xz','yz']:

    
    fig, axes = plt.subplots(chunknum,chunknum, figsize = (chunknum*2,chunknum*2))
    fig.patch.set_facecolor('black')
    axvids = []
    included_chunks = np.zeros((chunksq, curchunks.shape[1], curchunks.shape[-2], curchunks.shape[-1]))
    
    
    #add state label to the figure
    fig.text(0.1,0.9, ast, color = 'white', fontsize = 14)
    
    for i, image in enumerate(curchunks):
        
        persp = views[i]
        print(persp)
        if persp == 'xy':
            axproj = 1
        elif persp == 'xz':
            axproj = 2
        elif persp == 'yz':
            axproj = 3
        
        #roughly align trajectories to the right
        chunktemp = TotalFrame[TotalFrame.cell.isin(curdf[curdf.chunk_index == i].cell)]
        chunkvec = chunktemp[['Trajectory_X','Trajectory_Y','Trajectory_Z']].mean().values
        # chunkvec = [chunktemp.iloc[-1].x-chunktemp.iloc[0].x,
        #             chunktemp.iloc[-1].y-chunktemp.iloc[0].y,
        #             chunktemp.iloc[-1].z-chunktemp.iloc[0].z]
        #get angle between the vector and the planes
        xzang = angle3D(chunkvec[0], 0, chunkvec[2], 1, 0, 0)
        # xyang = angle3D(chunkvec[0], chunkvec[1], chunkvec[2], chunkvec[0], 0, chunkvec[2])
        xyang = angle3D(chunkvec[0], chunkvec[1], 0, 1, 0, 0)
        #make sure the directionality is correct
        xzang = xzang if chunkvec[2]>0 else -1*xzang
        xyang = xyang if chunkvec[1]>0 else -1*xyang
        
        #determine how many times to flip image along each axis
        angthresh = np.digitize([xyang,xzang], anglebins)
        xyflips = flips[angthresh[0]]
        xzflips = flips[angthresh[1]]
        

        #rotate to point right
        if (abs(xzflips) == 2) and (abs(xyflips) == 2):
            rotim = np.rot90(image, xyflips, axes = (2,3))
        elif abs(xzflips)>abs(xyflips):
            rotim = np.rot90(image, xzflips, axes = (1,3))
            rotim = np.rot90(rotim, xyflips, axes = (2,3))
        else:
            rotim = np.rot90(image, xyflips, axes = (2,3))
            rotim = np.rot90(rotim, xzflips, axes = (1,3))
        
        
        ### extra rotations to fix ones that don't work well automatically
        if (tp == 'top') and (i == 2):
            rotim = np.rot90(rotim, -1, axes = (1,3))
        elif (tp == 'top') and (i == 14):
            rotim = np.rot90(rotim, 1, axes = (1,2))
        elif (tp == 'top') and (i == 8):
            rotim = np.rot90(rotim, -1, axes = (1,2))
        
        #make the max projection
        proj = np.max(rotim, axis = axproj)
        
        #get the axis
        ax = axes.flatten()[i]
        
        #bleaching correction/brightness adjustment
        proj_bc = np.zeros(proj.shape)
        if np.max(proj)/np.mean(proj[proj>100])>4.5:
            corrval = np.percentile(proj[proj<np.percentile(proj, 99.5)],99.9)
            print(i)
        else:
            corrval = np.percentile(proj, 99.9)
        # corrval = np.mean(proj)+np.std(proj)*4
        for n in range(len(proj)):
            #all images have a min of zero so just divide by max
            proj_bc[n] = proj[n]/corrval
    
        
        #all the images
        imsh = ax.imshow(proj_bc[0], cmap = 'gray', animated = True, zorder = 0)
        
        #append images and ax object
        axvids.append(imsh)
        included_chunks[i] = proj_bc
        
        
        if i == 0:
            scalebar_x_displacement = proj_bc.shape[-1]-10
            scalebar_y_displacement = proj_bc.shape[-2]-10
            scalebar_length = 10
            resolution = 0.145*4 #um / pixel
            
            #scalebar
            sb = ax.plot([scalebar_x_displacement-(scalebar_length/resolution), scalebar_x_displacement],
                    [scalebar_y_displacement, scalebar_y_displacement],
                    lw = 3,
                    color = 'white',
                    zorder=2)
            
            #scalbar text
            sb_label = ax.text(scalebar_x_displacement-(scalebar_length/resolution)-1.5,
                                scalebar_y_displacement + 7,
                                f'{scalebar_length} μm',
                                color = 'white',
                                fontdict = {'fontsize': 10})
        
        
        # ### time label
        # timer = ax2.text(3,18,'00:00', color = 'white', fontdict = {'fontsize': 24})
        
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(1)
        
        for ax in axes.flatten()[i:]:
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
        
        # make function for updating point position
        def animate(i,):
            #set the current set of data
            avs = []
            for ic, av in zip(included_chunks, axvids):
                av.set_data(ic[i])
                avs.append(av)
            
            # #animate the vline on the aer plot
            # vtime = times[i]/60
            # vline.set_xdata(vtime)
            
            ## scale bar info
            scalebar_x_displacement = proj_bc.shape[-1]-10
            scalebar_y_displacement = proj_bc.shape[-2]-10
            scalebar_length = 10
            resolution = 0.145*4 #um / pixel
            #scalebar "animation"
            sb[0].set_data([scalebar_x_displacement-(scalebar_length/resolution), scalebar_x_displacement],
                    [scalebar_y_displacement, scalebar_y_displacement])
            #scalebar label "animation
            sb_label.set_text(f'{scalebar_length} μm')
            
            # #timer animation
            # timer.set_text(format_seconds(times[i]))
            
            artists = avs + [sb[0], sb_label]
            return artists #xyimsh, xzimsh, yzimsh, timer, vline#, aerplot
        
        
        
        #add two to the frame count to adjust the range function and to add a blank frame at the beginning
        
        ani = FuncAnimation(fig, animate, repeat=True, blit=True, 
                            frames=curchunks.shape[1],)
        # plt.show()
        
        
        writer = FFMpegWriter(fps=4) #, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
        ani.save(__file__.split('.')[0] + f'_{ast}_animated.mp4', writer = writer, dpi = 200)#, extra_args=['-vcodec', 'libx264'])
        
        # plt.close()
        
