
"""
Modified from Allen Cell aicsshparam  
"""

import re
import vtk
import warnings
import pyshtools
import numpy as np
import pandas as pd
from vtk.util import numpy_support
from skimage import transform as sktrans
import skimage.measure
from scipy import signal
from scipy import interpolate as spinterp
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
from pathlib import Path

from . import shtools_mod #, cytoparam_mod
from .utils import align_vec_to_xaxis_euler, angle3D





#surface area as a ratio of surface area of the a similar volume sphere
def get_sphericity(
        surf, #object surface area
        vol, #object volume
        ):
    # r = ((3*vol)/(4*math.pi)) ** (1/3)
    # SA = 4*math.pi*r
    return (np.pi**(1/3)*(6*vol)**(2/3))/surf #SA/surf


def get_long_axis_eulers_img(
        im, #binary segmented image
        xyres, #xy resolution of the image in microns
        zstep, #z resolution of the image in microns
        return_rotation_object:bool = False, #whether to return the scipy rotation object
        ):
    ## get zyx cell coords from segmented image
    cell_coords = np.stack(np.where(im>0))
    ## center coords
    cell_coords = cell_coords - np.mean(cell_coords,axis = 1,keepdims=True)
    ## adjust coordinates to micron resolution
    pixel_res = np.array([zstep, xyres, xyres])[..., np.newaxis]
    cell_coords *= pixel_res


    #get covariance matrix and find eigenvalues and vectors
    cov = np.cov(cell_coords)
    cell_evals, cell_evecs = np.linalg.eigh(cov)
    #make sure that the eigenvalues and vectors are in the order of highest to lowest
    idx = np.argsort(cell_evals)[::-1]
    cell_evals = cell_evals[idx]
    cell_evecs = cell_evecs[:,idx]

    ### PC1 eigen vector in xyz order
    eig1 = cell_evecs[:,0][::-1]
    #always get the x-positive direction of the vector
    if eig1[0]<0:
        eig1 *= -1

    ## actually get euler angles to align the long axis to the x axis
    euler_angles, rotationthing = align_vec_to_xaxis_euler(eig1, True)
    
    return (euler_angles, rotationthing) if return_rotation_object else euler_angles

def get_long_axis_eulers_mesh(
        mesh, #vtk polydata object
        return_rotation_object:bool = False, #whether to return the scipy rotation object
        ):
    ## get xyz cell coords from segmented image
    cell_coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    ### flip to zyx order
    cell_coords = np.flip(cell_coords, axis = 1)
    ## center coords
    cell_coords = cell_coords - np.mean(cell_coords,axis = 0,keepdims=True)

    #get covariance matrix and find eigenvalues and vectors
    cov = np.cov(cell_coords.T)
    cell_evals, cell_evecs = np.linalg.eigh(cov)
    #make sure that the eigenvalues and vectors are in the order of highest to lowest
    idx = np.argsort(cell_evals)[::-1]
    cell_evals = cell_evals[idx]
    cell_evecs = cell_evecs[:,idx]

    ### PC1 eigen vector in xyz order
    eig1 = cell_evecs[:,0][::-1]
    #always get the x-positive direction of the vector
    if eig1[0]<0:
        eig1 *= -1

    ## actually get euler angles to align the long axis to the x axis
    euler_angles, rotationthing = align_vec_to_xaxis_euler(eig1, True)
    
    return (euler_angles, rotationthing) if return_rotation_object else euler_angles


# find the widest part of the cell relative to the x axis
def find_normal_width_peaks(
        impath,
        csvdir,
        align_method: str = 'None',
        ):

    
    #get cell name from impath
    cell_name = impath.name.split('/')[-1].split('_cell_mesh')[0]
    #read mesh
    mesh = shtools_mod.read_polydata(impath)

    #get euler angles to align the provided vector to the +x axis
    if type(align_method) == np.ndarray:
        vec = align_method.copy()
        Euler_Angles = align_vec_to_xaxis_euler(vec)
    #get euler angles to align the trajectory vector to the +x axis
    elif align_method == 'trajectory':
        #if the csvdir is a string read the csv file, if it's a dict turn it into a DataFrame
        if isinstance(csvdir, Path):
            infopath = csvdir.joinpath(cell_name + '_cell_info.csv')
            info = pd.read_csv(infopath, index_col=0)
        elif type(csvdir)==dict:
            info = pd.DataFrame(csvdir, index=[0])
        vec = np.array([info.Trajectory_X[0], info.Trajectory_Y[0], info.Trajectory_Z[0]])
        Euler_Angles = align_vec_to_xaxis_euler(vec)
    #get euler angles to align the long axis of the cell to the x axis   
    elif align_method == 'long_axis':
        Euler_Angles = get_long_axis_eulers_mesh(mesh, False)
        
    #rotate mesh
    mesh = shtools_mod.rotate_and_scale_mesh(
            mesh,
            rotations = Euler_Angles,
            )
    
    
    
    #rotate around the x axis until you find the widest distance in y
    angles = np.arange(0,360,0.5)
    widths = np.empty(len(angles))
    for i, a in enumerate(angles):
        
        rotatedmesh = shtools_mod.rotate_and_scale_mesh(
                mesh,
                rotations = np.array([a,0,0]),
                )
        
        coords = numpy_support.vtk_to_numpy(rotatedmesh.GetPoints().GetData())
        #store the average of the negative y coordinates
        widths[i] = coords[np.where(coords[:,1]<0)][:,1].mean()
    
    #get the angle that rotates the least to achieve a "width" peak
    both = np.concatenate((widths, widths))
    peaks, properties = signal.find_peaks(abs(both),prominence=0.11, width=55)
    angpeaks = np.concatenate((angles,angles))[peaks]
    tangpeaks = angpeaks.copy()
    tangpeaks = list(set(tangpeaks))
    # tangpeaks[tangpeaks>180] -= 360

    return [cell_name, tangpeaks]


### measure the volume of a mesh in the positive and negative directions along
### a particular axis
def measure_volume_half(
        mesh,
        domain,
        ):
    
    #turn off warnings
    vtk.vtkObject.GlobalWarningDisplayOff()
    
    #Create a plane in the domain from the input
    plane = vtk.vtkPlane()

    if domain == 'x':
        plane.SetNormal(1,0,0)
    if domain == 'y':
        plane.SetNormal(0,1,0)
    if domain == 'z':
        plane.SetNormal(0,0,1)
    

    clip = vtk.vtkClipPolyData()
    clip.SetClipFunction(plane)
    clip.SetInputData(mesh)
    clip.Update()
    clipped = clip.GetOutput(0)
    
    #get the volume of the intersection
    CellMassProperties = vtk.vtkMassProperties()
    CellMassProperties.SetInputData(clipped)
    
    return CellMassProperties.GetVolume()


# get the spherical harmonic coefficients from a mesh
def get_shcoeffs_mesh(
        mesh,
        lmax,):
    
    coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    # Translate and update mesh normals
    mesh = shtools_mod.update_mesh_points(mesh, x, y, z)

    # Cartesian to spherical coordinates convertion
    rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    lat = np.arccos(np.divide(z, rad, out=np.zeros_like(rad), where=(rad != 0)))
    lon = np.pi + np.arctan2(y, x)

    # Creating a meshgrid data from (lon,lat,r)
    points = np.concatenate(
        [np.array(lon).reshape(-1, 1), np.array(lat).reshape(-1, 1)], axis=1
    )

    grid_lon, grid_lat = np.meshgrid(
        np.linspace(start=0, stop=2 * np.pi, num=256, endpoint=True),
        np.linspace(start=0, stop=1 * np.pi, num=128, endpoint=True),
    )

    # Interpolate the (lon,lat,r) data into a grid
    grid = spinterp.griddata(points, rad, (grid_lon, grid_lat), method="nearest")

    # Fit grid data with SH. Look at pyshtools for detail.
    coeffs = pyshtools.expand.SHExpandDH(grid, sampling=2, lmax_calc=lmax)

    # Reconstruct grid. Look at pyshtools for detail.
    grid_rec = pyshtools.expand.MakeGridDH(coeffs, sampling=2)

    # Resize the input grid to match the size of the reconstruction
    grid_down = sktrans.resize(grid, output_shape=grid_rec.shape, preserve_range=True)

    # Create (l,m) keys for the coefficient dictionary
    lvalues = np.repeat(np.arange(lmax + 1).reshape(-1, 1), lmax + 1, axis=1)

    keys = []
    for suffix in ["C", "S"]:
        for (l, m) in zip(lvalues.flatten(), lvalues.T.flatten()):
            keys.append(f"shcoeffs_L{l}M{m}{suffix}")

    coeffs_dict = dict(zip(keys, coeffs.flatten()))

    return (coeffs_dict, grid_rec), (grid_down)


#### get shcoeffs from an image
def get_shcoeffs_image(
    image: np.array,
    img_name: str,
    lmax: int,
    xyres: float,
    zstep: float,
    Euler_Angles: np.array,
    sigma: float,
    normal_rotation: float,
    compute_lcc: bool = True,
    ):

    """Compute spherical harmonics coefficients that describe an object stored as
    an image.

        Calculates the spherical harmonics coefficients that parametrize the shape
        formed by the foreground set of voxels in the input image. The input image
        does not need to be binary and all foreground voxels (background=0) are used
        in the computation. Foreground voxels must form a single connected component.
        If you are sure that this is the case for the input image, you can set
        compute_lcc to False to speed up the calculation. In addition, the shape is
        expected to be centered in the input image.

        Parameters
        ----------
        image : ndarray
            Input image. Expected to have shape ZYX.
        lmax : int
            Order of the spherical harmonics parametrization. The higher the order
            the more shape details are represented.
        zstep : float
            Z step of the image
        xyres : float
            microns/pixel resolution
        Euler_Angles : numpy array
            (3,) array of angles for rotation of the shape
        normal_rotation_method: str
            "widest" is longest axis parallel to trajectory
        
        Returns
        -------
        coeffs_dict : dict
            Dictionary with the spherical harmonics coefficients and the mean square
            error between input and its parametrization
        grid_rec : ndarray
            Parametric grid representing sh parametrization
        image_ : ndarray
            Input image after pre-processing (lcc calculation, smooth and binarization).
        mesh : vtkPolyData
            Polydata representation of image_.
        grid_down : ndarray
            Parametric grid representing input object.
        transform : tuple of floats
            (xc, yc, zc, angle) if alignment_2d is True or
            (xc, yc, zc) if alignment_2d is False. (xc, yc, zc) are the coordinates
            of the shape centroid after alignment; angle is the angle used to align
            the image

        Other parameters
        ----------------
        sigma : float, optional
            The degree of smooth to be applied to the input image, default is 0 (no
            smooth)
        compute_lcc : bool, optional
            Whether to compute the largest connected component before appliying the
            spherical harmonic parametrization, default is True. Set compute_lcc to
            False in case you are sure the input image contains a single connected
            component. It is crucial that parametrization is calculated on a single
            connected component object.

        Notes
        -----
        Alignment mode '2d' allows for keeping the z axis unchanged which might be
        important for some applications.

        Examples
        --------
            import numpy as np
            from aicsshparam import shparam, shtools

            img = np.ones((32,32,32), dtype=np.uint8)

            (coeffs, grid_rec), (image_, mesh, grid, transform) =
                shparam.get_shcoeffs(image=img, lmax=2)
            mse = shtools.get_reconstruction_error(grid, grid_rec)

            print('Coefficients:', coeffs)
        >>> Coefficients: {'shcoeffs_L0M0C': 18.31594310878251, 'shcoeffs_L0M1C': 0.0,
        'shcoeffs_L0M2C': 0.0, 'shcoeffs_L1M0C': 0.020438775421611564, 'shcoeffs_L1M1C':
        -0.0030960466571801513, 'shcoeffs_L1M2C': 0.0, 'shcoeffs_L2M0C':
        -0.0185688727281408, 'shcoeffs_L2M1C': -2.9925077712704384e-05,
        'shcoeffs_L2M2C': -0.009087503958673892, 'shcoeffs_L0M0S': 0.0,
        'shcoeffs_L0M1S': 0.0, 'shcoeffs_L0M2S': 0.0, 'shcoeffs_L1M0S': 0.0,
        'shcoeffs_L1M1S': 3.799611612562637e-05, 'shcoeffs_L1M2S': 0.0,
        'shcoeffs_L2M0S': 0.0, 'shcoeffs_L2M1S': 3.672543904347801e-07,
        'shcoeffs_L2M2S': 0.0002230857005948496}
            print('Error:', mse)
        >>> Error: 2.3738182456948795
    """

    #make an empty variable to return if there's no exceptions
    exceptions_list = None

    if len(image.shape) != 3:
        raise ValueError(
            "Incorrect dimensions: {}. Expected 3 dimensions.".format(image.shape)
        )

    if image.sum() == 0:
        # raise ValueError("No foreground voxels found. Is the input image empty?")
        warnings.warn(
            "No foreground voxels found. Is the input image empty?" + str(img_name)
        )
        exceptions_list = ["No foreground voxels found.", img_name]

    # Binarize the input. We assume that everything that is not background will
    # be use for parametrization
    image_ = image.copy()
    image_[image_ > 0] = 1

    # Converting the input image into a mesh using regular marching cubes
    mesh, image_, first_center = shtools_mod.get_mesh_from_image(image=image_, sigma=sigma)
    
    #rotate and scale mesh
    mesh = shtools_mod.rotate_and_scale_mesh(
            mesh,
            rotations = Euler_Angles,
            scale = np.array([xyres, xyres, zstep]), 
            )
    
 
    #################### normal rotation by provided angle ###############
    if normal_rotation!=0:        
        #rotate mesh by chosen angle
        mesh = shtools_mod.rotate_and_scale_mesh(
                mesh,
                rotations = np.array([normal_rotation,0,0]),
                )
        
    
    
    if not image_[tuple([int(u) for u in first_center[::-1]])]:
        warnings.warn(
            "Mesh centroid seems to fall outside the object. This indicates\
        the mesh may not be a manifold suitable for spherical harmonics\
        parameterization." + str(img_name)
        )
        exceptions_list = ["Mesh centroid seems to fall outside the object", img_name]

        
    # Get coordinates of mesh points
    coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    #get the centroid
    second_center = coords.mean(axis=0, keepdims=True)
    #subtract centroid from coordinates
    coords -= second_center
    #separate the coordinates
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    transform = tuple((np.array(first_center).squeeze(), second_center.squeeze()))

    # Translate and update mesh normals
    mesh = shtools_mod.update_mesh_points(mesh, x, y, z)

    # get shcoeffs from mesh
    (coeffs_dict, grid_rec), (grid_down) = get_shcoeffs_mesh(
                                                            mesh,
                                                            lmax,)

    return (coeffs_dict, grid_rec, exceptions_list), (image_, mesh, grid_down, transform)




def axis_order(ax):
    if ax == 'X':
        return [1, 0, 0]
    if ax == 'Y':
        return [0, 1, 0]
    if ax == 'Z':
        return [0, 0, 1]


####### measure the distance between the reconstructed mesh and the original
def recondistancesmesh(
        meshdir,
        img_name,
        l_order_num,
        ):

    ###### makes spherical harmonic reconstructions of the cell at various values of l 
    ###### starting with the surface mesh of the cell
    
    mesh = vtk.vtkXMLPolyDataReader()
    mesh.SetFileName(meshdir + img_name)
    mesh.Update()
    mesh = mesh.GetOutput()
    
    OtRdistances = {}
    RtOdistances = {}
    grid_se = {}
    
    ########### do a bunch of reconstructions for the cell #################
    for l_order in range(l_order_num):
        ordernumber = 'Lmax '+ str(l_order)
        (coeffs, grid_rec), (grid) = get_shcoeffs_mesh(
            mesh = mesh,
            lmax = l_order,
        )
        #get standard error of reconstruction from grid
        grid_se[ordernumber] = shtools_mod.get_reconstruction_error(grid, grid_rec)

        #put all the coefficients into a pandas dataframe
        cell_coeffs = pd.DataFrame([coeffs.values()],
                                  index = [img_name.replace('_segmented.tiff','')],
                                  columns = list(coeffs.keys()))

        #split up all the cofficients so they can be fed to the get_reconstruction_from_coeffs function
        coeff_names = list(cell_coeffs.columns)
        L_coeffs = len(np.unique(re.findall(r'L\d*', ''.join(coeff_names))))
        M_coeffs = len(np.unique(re.findall(r'M\d*', ''.join(coeff_names))))
        #reconstruct from coefficients
        cell_recon, grid_recon = shtools_mod.get_reconstruction_from_coeffs(np.array(cell_coeffs).reshape(2,L_coeffs,M_coeffs))
        #get average nearest distance for this particular reconstruction
        tree = KDTree(numpy_support.vtk_to_numpy(mesh.GetPoints().GetData()))
        d, idx = tree.query(numpy_support.vtk_to_numpy(cell_recon.GetPoints().GetData()))
        OtRdistances[ordernumber] = np.mean(d)


        #get average nearest distance for this particular reconstruction
        tree = KDTree(numpy_support.vtk_to_numpy(cell_recon.GetPoints().GetData()))
        d, idx = tree.query(numpy_support.vtk_to_numpy(mesh.GetPoints().GetData()))
        RtOdistances[ordernumber] = np.mean(d)

    return([img_name, OtRdistances, RtOdistances, grid_se])

def get_pilr_stuct_string(
    cell_name:str,
    ):
    if 'actin' in cell_name:
        str_name = 'actin'
    elif ('Hoechst' in cell_name) or ('DNA' in cell_name):
        str_name = 'nucleus'
    elif 'myosin' in cell_name:
        str_name = 'myosin'
    else:
        str_name = ''
    return str_name

# def get_pilr_nonuc(
#     impath: Path,
#     savedir: str,
#     xyres: float,
#     zstep: float,
#     normal_rotation_method: str,
#     l_order: int,
#     nisos: int,
#     pilr_method: str,
#     sigma: float = 0,
#     align_method: str = 'None',
#     ):

#     #get cell name from impath
#     cell_name = impath.name.split('/')[-1].split('_segmented')[0]

#     #determine the type of structure
#     str_name = get_pilr_stuct_string(cell_name)

#     #read image
#     im = TiffReader(impath)
#     #determind image dimensions
#     if im.shape[0]>2:
#         ci = im.data[0,:,:,:]
#         si = im.data[1:,:,:,:]
#     else:
#         ci = im.data[0,:,:,:]
#         si = im.data[1,:,:,:][np.newaxis]

#     ##### if we're going to get a PILR get shchoeffs and mesh in terms of pixels
#     (coeffs_mem, grid_rec, widestangle, exceptions_list), (image_, cell_mesh, grid, centroid_mem) = get_shcoeffs_mod(        
#         image = ci,
#         img_name= cell_name,
#         lmax = l_order,
#         xyres = xyres/xyres, #use pixels and not microns because I'll need pixel dimensions for PILRs
#         zstep = zstep/xyres,
#         Euler_Angles = euler_angles,
#         sigma = sigma,
#         normal_rotation_method = normal_rotation_method,
#         )


#     #create inner sphere
#     sphereSource = vtk.vtkSphereSource()
#     sphereSource.SetCenter(0.0, 0.0, 0.0)
#     sphereSource.SetRadius(nisos[0]/2)
#     # Make the surface smooth.
#     sphereSource.SetPhiResolution(100)
#     sphereSource.SetThetaResolution(100)
#     sphereSource.Update()
#     spherepoly = sphereSource.GetOutput()


#     (sphere_coeffs, grid_rec), (grid_down) = get_shcoeffs_mesh(
#             spherepoly,
#             lmax= l_order)

    
#     images_to_probe = []
#     if pilr_method == 'threshold':
#         #set levels strings for structure thrsholds
#         if len(si)>1:
#             levels = ['low','mid','high']
#         else:
#             levels = ['mid']
#         #for each structure threshold try to get a PILR
#         for n, s in enumerate(si):

#             #provide escape for cells with no signal in the "structure channel"
#             if np.max(s) == 0:
#                 #get voxelized intracellular structure image
#                 img, origin = cytoparam_mod.voxelize_meshes([cell_mesh])
            
#                 #get rotated segmentented str signal alone
#                 strimg = img.copy()
#                 strimg[strimg>0]=255
            
#             else:
#                 #get structure mesh
#                 str_mesh, _, cent = shtools_mod.get_mesh_from_image(
#                     image = s,
#                     translate_to_origin=False,
#                     lcc = False,
#                     center = np.array(centroid_mem)[0]
#                     )
#                 #euler rotation and scaling
#                 str_mesh = shtools_mod.rotate_and_scale_mesh(str_mesh,
#                                                     rotations = euler_angles,
#                                                     scale = np.array([xyres, xyres, zstep])/xyres)
#                 #widest angle rotation
#                 str_mesh = shtools_mod.rotate_and_scale_mesh(str_mesh,
#                                                     rotations = np.array([widestangle,0,0]))
#                 #adjust the structure center to the final position of the cell after rotation
#                 coords = numpy_support.vtk_to_numpy(str_mesh.GetPoints().GetData())
#                 coords -= np.array(centroid_mem)[1]
#                 str_mesh = shtools_mod.update_mesh_points(str_mesh, coords[:, 0], coords[:, 1], coords[:, 2])
            
#                 #get voxelized intracellular structure image
#                 img, origin = cytoparam_mod.voxelize_meshes([cell_mesh,str_mesh])
                
#                 #get rotated segmentented str signal alone
#                 strimg = img.copy()
#                 strimg[strimg<2]=0
#                 strimg[strimg>0]=255
#                 images_to_probe.append([str_name,strimg])
                
#                 #scale structure mesh
#                 #set transform and apply
#                 meshf = savedir.joinpath('meshes')
#                 str_mesh = shtools_mod.rotate_and_scale_mesh(
#                         str_mesh,
#                         scale = np.array([xyres, xyres, xyres]),
#                         )

#                 #save str mesh
#                 writer = vtk.vtkXMLPolyDataWriter()
#                 writer.SetFileName(meshf.joinpath(cell_name + f'_str_mesh_{levels[n]}.vtp'))
#                 writer.SetInputData(str_mesh)
#                 writer.Write()
                
#         # since threshold is used for discrete structures, get the centroid
#         # of that structure in an aligned cell
#         struct_coords = numpy_support.vtk_to_numpy(str_mesh.GetPoints().GetData())
#         struct_centroid = struct_coords.mean(axis=0, keepdims=True)[0]

#     elif pilr_method == 'raw':
#         ######### translate coordinates to membrane centroid
#         #open the raw data
#         rawpath = impath.parent.absolute().joinpath(impath.name.split('_segmented')[0] + '_raw.tiff')
#         #read image
#         raw = TiffReader(rawpath).data
#         memseg = np.where(ci>0)
#         intensities = np.tile(raw[1][memseg],3)
#         #add some half points
#         memsegmore = []
#         for m in memseg:
#             memsegmore.append(np.concatenate((m,m-0.25,m+0.25)))
#         memcent = np.mean(memsegmore, axis = 1)
#         centcoords = [memsegmore[i]-m for i, m in enumerate(memcent)]
#         ########## rotate coordinates
#         #first rotate toward trajectory (coords are flipped from zyx to xyz)
#         rotcoords = rotationthing[0].apply(np.flip(np.array(centcoords).T, axis = 1))
#         #then do width rotation (coords are flipped back to zyx from xyz)
#         widrot = R.from_euler('xyz',np.array([widestangle,0,0]), degrees = True)
#         widrotcoords = np.flip(widrot.apply(rotcoords), axis = 1)
#         ######### move coordinates to origin as 0,0,0 (size of image plus pad 1)
#         rotimg, origin = cytoparam_mod.voxelize_meshes([cell_mesh])
#         rotseg = np.where(rotimg>0)
#         rotcent = np.mean(rotseg, axis = 1)
#         zerocoords = np.subtract(widrotcoords, -rotcent)#origin[::-1])
#         ######### turn coordinates into int
#         zerointcoords = np.round(zerocoords).astype(np.int16)
#         ######### apply scalars to new array
#         strimg = np.zeros(rotimg.shape)
#         #combine coords with intensities
#         coordint = np.hstack((zerointcoords,intensities.reshape(-1,1)))
#         for z in range(strimg.shape[-3]):
#             zcoords = coordint[np.where(coordint[:,0] == z)]
#             for y in np.unique(zcoords[:,1]):
#                 ycoords = zcoords[np.where(zcoords[:,1] == y)]
#                 for x in np.unique(ycoords[:,2]):
#                     xcoords = ycoords[np.where(ycoords[:,2] == x)]
#                     #fill in the new value if it fits in a real position
#                     if all((z,y,x)<=np.array(strimg.shape[-3:])-1):
#                         strimg[z,y,x] = np.mean(xcoords[:,-1])
#         ####### normalize the strimg
#         strimg = strimg-intensities.min()
#         strimg = strimg/strimg.max()
#         strimg[strimg<0] = 0
#         images_to_probe.append([str_name,strimg])

#     #########parameterize cell
#     aicstif = cytoparam_mod.cellular_mapping(
#         coeffs_mem = coeffs_mem,
#         centroid_mem = abs(origin)[0],
#         coeffs_nuc = sphere_coeffs,
#         centroid_nuc = abs(origin)[0],
#         nisos = nisos,
#         images_to_probe = images_to_probe,
#         )
            
#     #Save PILR
#     pilrf = savedir.joinpath('PILRs')
#     if pilrf.joinpath(cell_name+'_PILR.ome.tiff').exists():
#         pilrf.joinpath(cell_name+'_PILR.ome.tiff').unlink()
#     OmeTiffWriter.save(aicstif.get_image_data('CZYX', S=0, T=0), pilrf.joinpath(cell_name+'_PILR.ome.tiff'), dim_order='CZYX', channel_names=aicstif.channel_names)
    



def get_shape_info(
        mesh_path: Path,
        xyres: float,
        zstep: float,
        normal_rotation: float,
        l_order: int,
        align_method: str,
        ):

    """
        Parameters
        ----------
        mesh_path : Path
            Path to the mesh file
        xyres : float
            microns/pixel resolution of the image
        zstep : float
            Z step of the image
        normal_rotation_method : str
            "widest" is longest axis parallel to trajectory
        str_name : str
            String detailing the name of the intracellular structure in the image
        l_order : int
            l order for SH transformation
        
        Returns
        -------
        coeffs_dict : dict
            Dictionary with the spherical harmonics coefficients and the mean square
            error between input and its parametrization
        grid_rec : ndarray
            Parametric grid representing sh parametrization
        image_ : ndarray
            Input image after pre-processing (lcc calculation, smooth and binarization).
        mesh : vtkPolyData
            Polydata representation of image_.
        grid_down : ndarray
            Parametric grid representing input object.
        transform : tuple of floats
            (xc, yc, zc, angle) if alignment_2d is True or
            (xc, yc, zc) if alignment_2d is False. (xc, yc, zc) are the coordinates
            of the shape centroid after alignment; angle is the angle used to align
            the image

        Other parameters
        ----------------
        sigma : float, optional
            The degree of smooth to be applied to the input image, default is 0 (no
            smooth)
        compute_lcc : bool, optional
            Whether to compute the largest connected component before appliying the
            spherical harmonic parametrization, default is True. Set compute_lcc to
            False in case you are sure the input image contains a single connected
            component. It is crucial that parametrization is calculated on a single
            connected component object.
        alignment_2d : bool
            Wheather the image should be aligned in 2d. Default is True.
        make_unique : bool
            Set true to make sure the alignment rotation is unique. 
            """

    cell_name = mesh_path.name.split('_cell_mesh')[0]

    ### read the mesh
    mesh = shtools_mod.read_polydata(mesh_path)
    
    #if align_method is a numpy array, use that as the vector to align to
    if type(align_method) == np.ndarray:
        vec = align_method.copy()
        euler_angles = align_vec_to_xaxis_euler(vec, False) 
    elif align_method == 'trajectory':
        #read euler angles for alignment
        infopath = mesh_path.parents[1].joinpath('smooth_traj', cell_name + '_cell_info.csv')
        info = pd.read_csv(infopath, index_col=0)
        vec = np.array([info.Trajectory_X[0], info.Trajectory_Y[0], info.Trajectory_Z[0]])
        euler_angles = align_vec_to_xaxis_euler(vec, False) 
    elif align_method == 'long_axis':
        euler_angles = get_long_axis_eulers_mesh(mesh, False)

    
    
    #rotate mesh
    mesh = shtools_mod.rotate_and_scale_mesh(
            mesh,
            rotations = euler_angles,
            )
    #################### normal rotation by provided angle ###############
    if normal_rotation!=0:        
        #rotate mesh by chosen angle
        mesh = shtools_mod.rotate_and_scale_mesh(
                mesh,
                rotations = np.array([normal_rotation,0,0]),
                )
    
    ### enforce mesh center at origin after rotations
    # Get coordinates of mesh points
    coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    #get the centroid
    second_center = coords.mean(axis=0, keepdims=True)
    #subtract centroid from coordinates
    coords -= second_center
    #separate the coordinates
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    # Translate and update mesh normals
    cell_mesh = shtools_mod.update_mesh_points(mesh, x, y, z)

    # get shcoeffs from mesh
    (coeffs_mem, _), (_) = get_shcoeffs_mesh(
                                mesh,
                                l_order,)
        
    #get reconstruction errors both ways
    cell_recon, _ = shtools_mod.get_reconstruction_from_coeffs(np.array(list(coeffs_mem.values())).reshape(2,l_order+1,l_order+1))
    #get average nearest distance from original mesh to reconstruction
    tree = KDTree(numpy_support.vtk_to_numpy(cell_mesh.GetPoints().GetData()))
    d, idx = tree.query(numpy_support.vtk_to_numpy(cell_recon.GetPoints().GetData()))
    OriginaltoReconError = np.mean(d)
    #get average nearest distance from reconstruction to original mesh
    tree = KDTree(numpy_support.vtk_to_numpy(cell_recon.GetPoints().GetData()))
    d, idx = tree.query(numpy_support.vtk_to_numpy(cell_mesh.GetPoints().GetData()))
    RecontoOriginalError = np.mean(d)
    
    
    #Get physical properties of cell
    CellMassProperties = vtk.vtkMassProperties()
    CellMassProperties.SetInputData(cell_mesh)
    Cell_Volume = CellMassProperties.GetVolume()
    Cell_SurfaceArea = CellMassProperties.GetSurfaceArea()
    Cell_Sphericity = get_sphericity(Cell_SurfaceArea, Cell_Volume)
    #measure the volume of the cell only within the positive x, y, and z domains
    FrontVolume = measure_volume_half(cell_mesh, 'x')
    LeftVolume = measure_volume_half(cell_mesh, 'y')
    TopVolume = measure_volume_half(cell_mesh, 'z')
    
    
    #get cell major, median, and minor axes using the aligned mesh
    cell_coords = numpy_support.vtk_to_numpy(cell_mesh.GetPoints().GetData())
    alignlenfront = np.max(cell_coords[:,0])
    alignlenrear = np.min(cell_coords[:,0])
    alignwidleft = np.max(cell_coords[:,1])
    alignwidright = np.min(cell_coords[:,1])
    alignheighttop = np.max(cell_coords[:,2])
    alignheightbottom = np.min(cell_coords[:,2])
    ##### measure length of cell along the trajectory axis
    alignlen = alignlenfront-alignlenrear
    alignwid = alignwidleft-alignwidright
    alignheight = alignheighttop-alignheightbottom
    #remove duplicate coordinates
    duplicates = pd.DataFrame(cell_coords).duplicated().to_numpy()
    mask = np.ones(len(cell_coords), dtype=bool)
    mask[duplicates] = False
    cell_coords = cell_coords[mask,:]
    #get covariance matrix and find eigenvalues and vectors
    cov = np.cov(cell_coords.T)
    cell_evals, cell_evecs = np.linalg.eig(cov)
    #make sure that the eigenvalues and vectors are in the order of highest to lowest
    idx = np.argsort(cell_evals)[::-1]
    cell_evals = cell_evals[idx]
    cell_evecs = cell_evecs[:,idx]
    ### enforce consistent directionality of the eigenvectors
    # major axis points +x
    if cell_evecs[0,0]<0:
        cell_evecs[:,0] *= -1
    # median axis points -y
    if cell_evecs[1,1]>0:
        cell_evecs[:,1] *= -1
    # minor axis is right handed to the other two
    righth = np.cross(cell_evecs[:,0], cell_evecs[:,1])
    if (cell_evecs[2,2] * righth[2]) < 0:
        cell_evecs[:,2] *= -1
            

    #rotate the cell coordinates to align the major axis with the x, the median axis to the y and the minor axis to the z
    rotationthing = R.align_vectors(np.array([[1,0,0],[0,1,0]]), cell_evecs.T[:2,:])
    cell_coords = rotationthing[0].apply(cell_coords)
    #get lengths of the cell's absolute axes
    Cell_MajorAxis_Length = np.max(cell_coords[:,0])-np.min(cell_coords[:,0])
    Cell_MedianAxis_Length = np.max(cell_coords[:,1])-np.min(cell_coords[:,1])
    Cell_MinorAxis_Length = np.max(cell_coords[:,2])-np.min(cell_coords[:,2])


    ######### Build dict of angles between principle axes relative to the alignment axis #############
    ax_angle_dict = {}
    ax_names = ['Major','Median','Minor']
    for a, arr in enumerate(cell_evecs.T):
        #get angle between the vector and the planes
        XYAngle = angle3D(arr[0], arr[1], 0, 1, 0, 0)
        XZAngle = angle3D(arr[0], 0, arr[2], 1, 0, 0)
        YZAngle = angle3D(0, arr[1], arr[2], 0, 1, 0)
        TotalAngle = angle3D(arr[0], arr[1], arr[2], 1, 0, 0)
        #make sure the directionality is correct
        XYAngle = XYAngle if arr[1]>0 else -1*XYAngle
        XZAngle = XZAngle if arr[2]>0 else -1*XZAngle
        YZAngle = YZAngle if arr[2]>0 else -1*YZAngle
        ax_angle_dict.update({
            'Cell_'+ax_names[a]+'Axis_TotalAngle': TotalAngle, # absolute angle between principal axis and cell's alignment axis
            'Cell_'+ax_names[a]+'Axis_XYAngle': XYAngle, # X-Y angle between principal axis and cell's alignment axis
            'Cell_'+ax_names[a]+'Axis_XZAngle': XZAngle, # X-Z angle between principal axis and cell's alignment axis
            'Cell_'+ax_names[a]+'Axis_YZAngle': YZAngle, # Y-Z angle between principal axis and cell's alignment axis
            'Cell_'+ax_names[a]+'Axis_Vec_X': arr[0], #vector of the cell shape's absolute longest axis
            'Cell_'+ax_names[a]+'Axis_Vec_Y': arr[1], #vector of the cell shape's absolute longest axis
            'Cell_'+ax_names[a]+'Axis_Vec_Z': arr[2], #vector of the cell shape's absolute longest axis
            })
        

    #Shape stats dict
    Shape_Stats = {'cell': cell_name,
                   'Euler_angles_X': euler_angles[0],
                   'Euler_angles_Y':euler_angles[1],
                   'Euler_angles_Z':euler_angles[2],
                   'Width_Rotation_Angle': normal_rotation,
                   'Cell_Volume': Cell_Volume,
                    'Cell_Volume_Front': FrontVolume,
                    'Cell_Volume_Left': LeftVolume,
                    'Cell_Volume_Top': TopVolume,
                    'Volume_Front_Ratio': FrontVolume/Cell_Volume,
                    'Volume_Left_Ratio': LeftVolume/Cell_Volume,
                    'Volume_Top_Ratio': TopVolume/Cell_Volume,
                   'Cell_SurfaceArea': Cell_SurfaceArea,
                   'Cell_Sphericity': Cell_Sphericity,
                   'Cell_MajorAxis_Length': Cell_MajorAxis_Length,
                   'Cell_MedianAxis_Length': Cell_MedianAxis_Length,
                   'Cell_MinorAxis_Length': Cell_MinorAxis_Length,
                   'Cell_Aspect_Ratio': Cell_MajorAxis_Length/Cell_MinorAxis_Length,
                   'OriginaltoReconError': OriginaltoReconError,
                   'RecontoOriginalError': RecontoOriginalError,
                   'LengthAlongTrajectory': alignlen,
                   'LengthAlongTrajectoryFront': alignlenfront,
                   'LengthAlongTrajectoryRear': alignlenrear,
                   'WidthAlongTrajectory': alignwid,
                   'WidthAlongTrajectoryLeft': alignwidleft,
                   'WidthAlongTrajectoryRight': alignwidright,
                   'HeightAlongTrajectory': alignheight,
                   'HeightAlongTrajectoryTop': alignheighttop,
                   'HeightAlongTrajectoryBottom': alignheightbottom,
                    }

    #add the principal axes angles
    Shape_Stats.update(ax_angle_dict)
    #add the shcoeffs to the dict I just built
    Shape_Stats.update(coeffs_mem)
    
    
    return Shape_Stats



#wrapper for get_shape_info_nonuc for imap
def shape_info_imap(args):
    return get_shape_info(*args)