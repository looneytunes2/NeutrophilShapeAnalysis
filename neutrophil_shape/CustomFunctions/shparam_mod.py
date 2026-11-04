
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
from scipy import interpolate as spinterp
from scipy.sparse import coo_matrix
from scipy.spatial import KDTree, ConvexHull, QhullError
from scipy.spatial.transform import Rotation as R
from collections import deque
from dataclasses import dataclass
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

def orient_by_mass(
        coords,
        evec,
        ):
    """Flip evec if needed so its points toward the side with more mass."""
    proj = evec @ coords          # project all voxels onto this axis
    if np.median(proj) < 0:
        evec = -evec
    return evec


def calculate_coordinate_principal_axes(
    coords: np.array,
    ):
    """
    Calculate principal axes from coordinates (PCA)
    coords: np.array with shape (3,N) in zyx order
    """
    #get covariance matrix and find eigenvalues and vectors
    cov = np.cov(coords)
    cell_evals, cell_evecs = np.linalg.eigh(cov)
    #make sure that the eigenvalues and vectors are in the order of highest to lowest
    idx = np.argsort(cell_evals)[::-1]
    cell_evals = cell_evals[idx]
    cell_evecs = cell_evecs[:,idx]

    #orient the top two towards the greater mass
    for i in range(3):
        cell_evecs[:,i] = orient_by_mass(coords, cell_evecs[:,i])

    ### return eigenvectors Major->Minor along axis 0
    ### and xyz along axis 1
    return cell_evecs[::-1].T


def extract_object_principal_axes(
    im: np.array, #binary segmented image
    xyres: float, #xy resolution of the image in microns
    zstep: float, #z resolution of the image in microns
    ):
    
    ## get zyx cell coords from segmented image
    cell_coords = np.stack(np.where(im>0))
    ## center coords
    cell_coords = cell_coords - np.mean(cell_coords,axis = 1,keepdims=True)
    ## adjust coordinates to micron resolution
    pixel_res = np.array([[zstep], [xyres], [xyres]])
    cell_coords *= pixel_res

    return calculate_coordinate_principal_axes(cell_coords)

def extract_mesh_principal_axes(
        mesh, #vtk polydata object
        ):
    
    # ## open mesh
    # mesh = shtools_mod.read_polydata(mesh_path)
    ## get xyz cell coords from mesh
    cell_coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    ## get only unique coordinates
    cell_coords = np.unique(cell_coords, axis = 0)
    ### flip to zyx order
    cell_coords = np.flip(cell_coords, axis = 1)
    ## center coords
    cell_coords = cell_coords - np.mean(cell_coords,axis = 0,keepdims=True)

    return calculate_coordinate_principal_axes(cell_coords.T)



def calculate_orthogonal_mass_vector(
    coords: np.array,  # xyz coordinates (N,3) 
    vec: np.array,   # xyz vector to look orthogonally around
    ):
    """Find the vector with greatest mass perpendicular to a provided vector,
    computed directly in the original reference frame."""

    #make sure vector is normalized
    vec = vec / np.linalg.norm(vec)

    # build an orthonormal basis {u, w} spanning the plane orthogonal to vec
    # pick a helper vector not parallel to vec
    helper = np.array([1.0, 0.0, 0.0]) if abs(vec[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = helper - np.dot(helper, vec) * vec
    u /= np.linalg.norm(u)
    w = np.cross(vec, u)  

    # project cell coordinates onto u and w
    coords_2d = np.stack([coords @ u, coords @ w])

    # PCA in the 2D orthogonal-plane basis
    cov = np.cov(coords_2d)
    evals, evecs = np.linalg.eigh(cov)
    top = evecs[:, -1]  # (coeff along u, coeff along w)

    #get directionality with more mass
    top = orient_by_mass(coords_2d,top)

    # map back into the ORIGINAL 3D frame
    v2 = top[0] * u + top[1] * w
    v2 /= np.linalg.norm(v2)

    return v2 ## xyz 

def get_mesh_orthogonal_mass_vector(
    mesh_path: Path,
    vec: np.array,
    ):
    ## open mesh
    mesh = shtools_mod.read_polydata(mesh_path)
    ## get xyz cell coords from mesh
    cell_coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    ## get only unique coordinates
    cell_coords = np.unique(cell_coords, axis = 0)
    ## center coords
    cell_coords = cell_coords - np.mean(cell_coords,axis = 0,keepdims=True)

    return calculate_orthogonal_mass_vector(cell_coords, vec)

def get_orthogonal_mass_vector_imap(args):
    return get_mesh_orthogonal_mass_vector(*args)

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
    smooth_iters: int,
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
        smooth_iters : int
            Iterations of surface smoothing of the mesh made from the image, 0 for
            none (see shtools_mod.get_mesh_from_image)
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

    # the largest connected component is what the mesh is made from, and what the centroid is checked against
    if compute_lcc:
        image_ = shtools_mod.largest_connected_component(image_)

    # Converting the input image into a mesh using marching cubes, in microns (spacing is z, y, x), left where it is in the image
    mesh = shtools_mod.get_mesh_from_image(
        image=image_,
        lcc=False,
        spacing=[zstep, xyres, xyres],
        smooth_iters=smooth_iters,
        translate_to_origin=False,
        )
    #centroid of the mesh in image coordinates (x, y, z, in voxels, to index image_ and for the returned transform)
    first_center = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float64).mean(axis=0) / np.array([xyres, xyres, zstep])

    #rotate the mesh about its centroid
    mesh = shtools_mod.rotate_mesh(
            mesh,
            rotations = Euler_Angles,
            center = 'centroid',
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
    #the SH expansion is about the origin and the saved meshes are in their real-space positions
    mesh, _ = shtools_mod.translate_mesh_to_origin(mesh)

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
#                 #(in units of xyres: spacing is z, y, x, and the offset, x, y, z, subtracts the centroid)
#                 str_mesh = shtools_mod.get_mesh_from_image(
#                     image = s,
#                     spacing = [zstep/xyres, 1, 1],
#                     translate_to_origin=False,
#                     lcc = False,
#                     offset = -np.array(centroid_mem)[0] * np.array([1, 1, zstep/xyres])
#                     )
#                 #euler rotation
#                 str_mesh = shtools_mod.rotate_mesh(str_mesh,
#                                                     rotations = euler_angles)
#                 #widest angle rotation
#                 str_mesh = shtools_mod.rotate_mesh(str_mesh,
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
#                 coords = numpy_support.vtk_to_numpy(str_mesh.GetPoints().GetData())
#                 coords *= xyres
#                 str_mesh = shtools_mod.update_mesh_points(str_mesh, coords[:, 0], coords[:, 1], coords[:, 2])

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
    

ax_names = ['Major','Median','Minor']
dims = ["X", "Y", "Z"]
trajectory_columns = ['Trajectory_Vec_X','Trajectory_Vec_Y','Trajectory_Vec_Z']


#settings for protrusion detection and classification, see measure_protrusions_from_mesh
@dataclass(frozen=True)
class ProtrusionParams:
    """
    Settings for protrusion detection and classification (see
    measure_protrusions_from_mesh).

    Detection
    ---------
    lrec : degree of the low-order "body" reconstruction used as the smoothed
        envelope that the actual surface is compared against.
    smooth_iters : number of mesh-neighbor-averaging passes applied to the
        residual before peak-seeding/ordering, to keep single-vertex noise from
        being picked up as its own peak. All reported metrics use the raw
        (unsmoothed) residual regardless.
    prominence_sigmas : a basin must have prominence greater than this many
        robust-sigma (median absolute deviation scaled to approximate a
        standard deviation) above the residual's noise floor to be treated as
        a genuinely distinct protrusion.
    grow_sigmas : once a peak is confirmed significant, its region is grown
        outward (flood-fill) until the surface drops back to this many
        robust-sigma above the noise floor.
    min_area_verts : drop any surviving region smaller than this many vertices.

    Classification
    --------------
    aspect_ratio_median_thresh : protrusions with major/median axis length
        (aspect_ratio_median) at or above this, and a minor axis length below
        minor_axis_length_thresh, are "high aspect ratio": filopodia or
        lamellipodia depending on the orientation of the major axis. Others go
        on to the minor-axis and convex hull checks below.
    minor_axis_length_thresh : maximum minor axis length (um) for a protrusion
        to count as "high aspect ratio" above, i.e. thin enough to be a
        filopodium or an elongated lamellipodium.
    aspect_ratio_minor_thresh : protrusions that miss
        aspect_ratio_median_thresh but have a major/minor axis length
        (aspect_ratio_minor) at or above this are lamellipodia, i.e. flat
        sheets that are not elongated within their own plane.
    radial_alignment_thresh : high-aspect-ratio protrusions whose major axis
        has at least this dot product with the direction away from the cell
        centroid are "oriented away from the origin" (filopodia); below it they
        are lamellipodia. Default cos(45 degrees).
    residual_volume_ratio_thresh : protrusions that pass none of the aspect
        ratio checks and whose peak_residual_um / volume_um3 (1/um^2) is at or
        above this are tall and thin for their volume: 'pseudopod_filopodia'.
    hull_area_ratio_thresh : protrusions that pass none of the checks above
        and whose 'hull_area_ratio' is at or above this are pseudopods; below
        it (a large mismatch between the protrusion and its convex hull) they
        are lamellipodia.
    """
    lrec: int = 3
    smooth_iters: int = 5
    prominence_sigmas: float = 1.4
    grow_sigmas: float = 0.55
    min_area_verts: int = 50
    aspect_ratio_median_thresh: float = 1.6
    minor_axis_length_thresh: float = 3.0
    aspect_ratio_minor_thresh: float = 2.5
    radial_alignment_thresh: float = float(np.cos(np.pi / 4))
    residual_volume_ratio_thresh: float = 0.1
    hull_area_ratio_thresh: float = 0.6


def get_aligned_mesh(
        df: pd.DataFrame,  # row with 'cell' and the Euler angles for alignment
        meshdir: Path,  # directory containing '{cell}_cell_mesh.vtp' files
        ):
    """Read a cell's mesh, rotate it into its alignment frame and center it
    on the origin (its vertex centroid). The saved meshes are in their real-space
    positions (microns), so the mesh is rotated about its own centroid, which
    makes the aligned mesh independent of where the cell was in the image."""
    mesh_path = meshdir.joinpath(df.cell + '_cell_mesh.vtp')
    mesh = shtools_mod.read_polydata(mesh_path)

    euler_angles = df[['Euler_Angles_X','Euler_Angles_Y','Euler_Angles_Z']].values
    mesh = shtools_mod.rotate_mesh(
            mesh,
            rotations = euler_angles,
            center = 'centroid',
            )

    ### put the mesh center at the origin (also normalizes the position of a mesh that was saved already at the origin)
    mesh, _ = shtools_mod.translate_mesh_to_origin(mesh)
    return mesh


def get_aligned_vectors(
        df: pd.DataFrame,  # row with the Euler angles, principal axes and trajectory vectors
        ):
    """Rotate the cell's principal axes (rows: major, median, minor), trajectory
    vector and next trajectory vector into the alignment frame, i.e. the frame
    that get_aligned_mesh puts the mesh in. The axes are not yet given a
    consistent directionality (see orient_cell_axes)."""
    euler_angles = df[['Euler_Angles_X','Euler_Angles_Y','Euler_Angles_Z']].values
    euler_rotation = R.from_euler('xyz', euler_angles, degrees = True)

    cell_evecs = np.array([[df[f"Cell_{axis}_Axis_Vec_{dim}"] for dim in dims] for axis in ax_names])
    cell_evecs_aligned = euler_rotation.apply(cell_evecs)
    trajectory_aligned = euler_rotation.apply(df[trajectory_columns].values)
    next_trajectory_aligned = euler_rotation.apply(df[['Next_'+x for x in trajectory_columns]].values)

    return cell_evecs_aligned, trajectory_aligned, next_trajectory_aligned


def orient_cell_axes(
        cell_evecs_aligned: np.ndarray,  # (3,3) aligned principal axes, rows = major, median, minor
        ):
    """Give the aligned principal axes the consistent directionality used for
    the Cell_Aligned_*_Axis_Vec columns: major points +x, median points -y,
    minor points +z. Returns a flipped copy."""
    cell_evecs_aligned = cell_evecs_aligned.copy()
    if cell_evecs_aligned[0,0]<0:
        cell_evecs_aligned[0,:] *= -1
    if cell_evecs_aligned[1,1]>0:
        cell_evecs_aligned[1,:] *= -1
    if cell_evecs_aligned[2,2]<0:
        cell_evecs_aligned[2,:] *= -1
    return cell_evecs_aligned


#order of the SH coefficients that get_shape_info calculates and saves, whatever the l_order in the
#config is. Everything that recalls the coefficients (PCA, reconstructions, ...) uses the config's
#l_order, see shapePCAtools.load_shcoeffs
SHAPE_INFO_L_ORDER = 20


def get_shape_info(
        df: pd.DataFrame,
        meshdir: Path,
        measure_protrusions: bool = True,
        protrusion_params: ProtrusionParams = None,
        ):

    """
        Parameters
        ----------
        df : pd.Series
            Row with the cell name, the euler angles to align the mesh to the
            alignment frame, and the trajectory and principal axis vectors
            (see get_aligned_vectors).
        meshdir : Path
            Path to the mesh directory
        measure_protrusions : bool, optional
            Whether to also detect and measure surface protrusions on the aligned
            mesh, reusing the mesh and SH coefficients computed here. Set False to
            skip (protrusion metrics depend on the alignment only through their
            directional measures, so it can be worth measuring them for just one
            alignment). Default True.
        protrusion_params : ProtrusionParams, optional
            Protrusion detection and classification settings, default
            ProtrusionParams().

        Returns
        -------
        Shape_Stats : dict
            Shape measurements of the cell. The reconstruction errors are those of
            the SHAPE_INFO_L_ORDER fit.
        shcoeffs : dict
            The cell's name ('cell') and its SH coefficients up to order
            SHAPE_INFO_L_ORDER, kept apart from Shape_Stats so they can be saved
            separately and combined with the shape metrics on 'cell' when needed.
        protrusion_rows : list of dict
            One dict per detected protrusion (see measure_protrusions_from_mesh);
            empty if measure_protrusions is False or none were found.
            """

    cell_name = df.cell

    ### read the mesh, rotate it into the alignment frame and center it
    cell_mesh = get_aligned_mesh(df, meshdir)
    #copy, since get_shcoeffs_mesh below swaps out the mesh's point array
    coords = numpy_support.vtk_to_numpy(cell_mesh.GetPoints().GetData()).copy()

    # get shcoeffs from mesh
    (coeffs_mem, _), (_) = get_shcoeffs_mesh(
                                cell_mesh,
                                SHAPE_INFO_L_ORDER,)

    #get reconstruction errors both ways
    cell_recon, _ = shtools_mod.get_reconstruction_from_coeffs(np.array(list(coeffs_mem.values())).reshape(2,SHAPE_INFO_L_ORDER+1,SHAPE_INFO_L_ORDER+1))
    #get average nearest distance from original mesh to reconstruction
    tree = KDTree(numpy_support.vtk_to_numpy(cell_mesh.GetPoints().GetData()))
    d, idx = tree.query(numpy_support.vtk_to_numpy(cell_recon.GetPoints().GetData()))
    OriginaltoReconError = np.mean(d)
    #get average nearest distance from reconstruction to original mesh
    tree = KDTree(numpy_support.vtk_to_numpy(cell_recon.GetPoints().GetData()))
    d, idx = tree.query(numpy_support.vtk_to_numpy(cell_mesh.GetPoints().GetData()))
    RecontoOriginalError = np.mean(d)
    
    
    #Get surface and volume properties of cell
    CellMassProperties = vtk.vtkMassProperties()
    CellMassProperties.SetInputData(cell_mesh)
    Cell_Volume = CellMassProperties.GetVolume()
    Cell_SurfaceArea = CellMassProperties.GetSurfaceArea()
    Cell_Sphericity = get_sphericity(Cell_SurfaceArea, Cell_Volume)
    #measure the volume of the cell only within the positive x, y, and z domains
    FrontVolume = measure_volume_half(cell_mesh, 'x')
    LeftVolume = measure_volume_half(cell_mesh, 'y')
    TopVolume = measure_volume_half(cell_mesh, 'z')
    
    
    #get coordinates of the cell in the aligned frame
    cell_coords = numpy_support.vtk_to_numpy(cell_mesh.GetPoints().GetData())
    #remove duplicate coordinates
    cell_coords = np.unique(cell_coords, axis = 0)
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
    ## also get std of the coordinates along dimensions in the aligned frame
    Cell_Aligned_X_Stdev = np.std(cell_coords[:,0])
    Cell_Aligned_Y_Stdev = np.std(cell_coords[:,1])
    Cell_Aligned_Z_Stdev = np.std(cell_coords[:,2])

    ### rotate the principal axes and trajectory vectors into the alignment frame
    cell_evecs_aligned, trajectory_aligned, next_trajectory_aligned = get_aligned_vectors(df)

    #rotate the ORIGINAL cell coordinates to align the major axis with the x,
    #the median axis to the y and the minor axis to the z
    rotationthing, _ = R.align_vectors(np.array([[1,0,0],[0,1,0]]), cell_evecs_aligned[:2,:])
    #apply to the already rotated coordinates
    principal_axes_cell_coords = rotationthing.apply(coords)
    #partial principal axes lengths
    Cell_Major_Axis_Min = np.min(principal_axes_cell_coords[:,0])
    Cell_Major_Axis_Max = np.max(principal_axes_cell_coords[:,0])
    Cell_Median_Axis_Min = np.min(principal_axes_cell_coords[:,1])
    Cell_Median_Axis_Max = np.max(principal_axes_cell_coords[:,1])
    Cell_Minor_Axis_Min = np.min(principal_axes_cell_coords[:,2])
    Cell_Minor_Axis_Max = np.max(principal_axes_cell_coords[:,2])
    # total principal axes lengths
    Cell_Major_Axis_Length = Cell_Major_Axis_Max - Cell_Major_Axis_Min
    Cell_Median_Axis_Length = Cell_Median_Axis_Max - Cell_Median_Axis_Min
    Cell_Minor_Axis_Length = Cell_Minor_Axis_Max - Cell_Minor_Axis_Min


    ######### Build dict of angles between principle axes relative to the alignment axis #############
    ax_angle_dict = {}
    ##add trajectory in aligned frame
    ax_angle_dict.update({f'Aligned_Trajectory_Vec_{dim}':trajectory_aligned[d] for d, dim in enumerate(dims)})
    ##add next trajectory in aligned frame
    ax_angle_dict.update({f'Aligned_Next_Trajectory_Vec_{dim}':next_trajectory_aligned[d] for d, dim in enumerate(dims)})

    ### enforce consistent directionality of the principal axes prior to
    ### calculating aligned vector component values
    cell_evecs_aligned = orient_cell_axes(cell_evecs_aligned)

    for a, arr in enumerate(cell_evecs_aligned):
        #get angle between the vector and the planes
        anglevec = [[1,0,0],[0,-1,0],[0,0,1]]
        TotalAngle = angle3D(arr[np.newaxis, :], np.array([anglevec[a]]))[0]
        ax_angle_dict.update({
            f'Cell_{ax_names[a]}_Axis_TotalAngle': TotalAngle, # absolute angle between principal axis and cell's alignment axis
            f'Cell_Aligned_{ax_names[a]}_Axis_Vec_X': arr[0], # x component of axis
            f'Cell_Aligned_{ax_names[a]}_Axis_Vec_Y': arr[1], # y component of axis
            f'Cell_Aligned_{ax_names[a]}_Axis_Vec_Z': arr[2], # z component of axis
            })
    
        

    #Shape stats dict
    Shape_Stats = {'cell': cell_name,
                   'Cell_Volume': Cell_Volume,
                    'Cell_Volume_Front': FrontVolume,
                    'Cell_Volume_Left': LeftVolume,
                    'Cell_Volume_Top': TopVolume,
                    'Volume_Front_Ratio': FrontVolume/Cell_Volume,
                    'Volume_Left_Ratio': LeftVolume/Cell_Volume,
                    'Volume_Top_Ratio': TopVolume/Cell_Volume,
                   'Cell_SurfaceArea': Cell_SurfaceArea,
                   'Cell_Sphericity': Cell_Sphericity,
                    'Cell_Major_Axis_Min': Cell_Major_Axis_Min,
                    'Cell_Major_Axis_Max': Cell_Major_Axis_Max,
                    'Cell_Median_Axis_Min': Cell_Median_Axis_Min,
                    'Cell_Median_Axis_Max': Cell_Median_Axis_Max,
                    'Cell_Minor_Axis_Min': Cell_Minor_Axis_Min,
                    'Cell_Minor_Axis_Max': Cell_Minor_Axis_Max,
                   'Cell_Major_Axis_Length': Cell_Major_Axis_Length,
                   'Cell_Median_Axis_Length': Cell_Median_Axis_Length,
                   'Cell_Minor_Axis_Length': Cell_Minor_Axis_Length,
                   'Cell_Aspect_Ratio': Cell_Major_Axis_Length/Cell_Minor_Axis_Length,
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
                    'Cell_Aligned_X_Stdev': Cell_Aligned_X_Stdev,
                    'Cell_Aligned_Y_Stdev': Cell_Aligned_Y_Stdev,
                    'Cell_Aligned_Z_Stdev': Cell_Aligned_Z_Stdev,
                    }

    #add the principal axes angles
    Shape_Stats.update(ax_angle_dict)
    #the shcoeffs are kept apart from the shape stats, with the cell name to combine them again
    shcoeffs = {'cell': cell_name, **coeffs_mem}

    ### protrusions, reusing the aligned mesh and SH coefficients from above
    protrusion_rows = []
    if measure_protrusions:
        protrusion_rows = measure_protrusions_from_mesh(
            cell_mesh,
            pd.concat([df, pd.Series(coeffs_mem)]),
            SHAPE_INFO_L_ORDER,
            protrusion_params,
            )

    return Shape_Stats, shcoeffs, protrusion_rows



#wrapper for get_shape_info_nonuc for imap
def shape_info_imap(args):
    return get_shape_info(*args)



############################ PROTRUSION DETECTION VIA SPHARM RESIDUAL WATERSHED #############
def _unit_vector(v):
    """Normalize v, returning NaNs if it has no usable direction."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm == 0:
        return np.full(v.shape, np.nan)
    return v / norm


def _classify_protrusion(
        row: dict,  # one protrusion's measurements, as built in measure_protrusions_from_mesh
        params: ProtrusionParams,
        ):
    """
    Classify one protrusion from its measurements (aspect_ratio_median,
    aspect_ratio_minor, minor_axis_length_um, major_axis_{x,y,z},
    direction_{x,y,z}, peak_residual_um, volume_um3, hull_area_ratio).

    1. A high major/median aspect ratio with a short minor axis is a filopodia
       if the major axis points away from the cell centroid (dot product with
       the protrusion's direction) and a lamellipodia if it lies tangential to
       the cell.
    2. Otherwise a high major/minor aspect ratio is a lamellipodia.
    3. Otherwise convex hull decides: pseudopod if the protrusion fills out its
       hull, lamellipodia if there is a large mismatch.
    """
    if (row['aspect_ratio_median'] >= params.aspect_ratio_median_thresh
            and row['minor_axis_length_um'] < params.minor_axis_length_thresh):
        major_axis = np.array([row['major_axis_x'], row['major_axis_y'], row['major_axis_z']], dtype=float)
        direction = np.array([row['direction_x'], row['direction_y'], row['direction_z']], dtype=float)
        radial_alignment = major_axis @ direction
        return 'filopodia' if radial_alignment >= params.radial_alignment_thresh else 'lamellipodia'
    if row['aspect_ratio_minor'] >= params.aspect_ratio_minor_thresh:
        return 'lamellipodia'
    return 'pseudopod' if row['hull_area_ratio'] >= params.hull_area_ratio_thresh else 'lamellipodia'


def measure_protrusions_from_mesh(
        mesh,  # vtkPolyData, already rotated into the alignment frame and centered on the origin
        df: pd.Series,  # row with the cell's name, alignment vectors and SH coefficients (see below)
        l_order: int, # degree of the original shcoeffs
        params: ProtrusionParams = None,  # detection and classification settings, default ProtrusionParams()
        return_labels: bool = False,  # also return the cleaned mesh and per-vertex region labels, for visualization
        ):
    """
    Detect discrete surface protrusions (pseudopods/ruffles) on a cell by
    comparing its surface mesh to a smoothed low-order spherical harmonics
    reconstruction of its own body (the per-vertex difference between actual
    radius and that envelope's radius is the "residual"). Which bumps count
    as genuinely distinct protrusions -- rather than membrane texture riding
    on the flank of a bigger one -- is decided by topographic prominence via
    a persistence/watershed computation over the mesh surface graph, not by
    a single height cutoff: a single global threshold on this same residual
    field shatters one real protrusion into dozens of disconnected pieces.

    Each protrusion is then measured in the alignment frame, whose origin is
    the cell centroid. 'direction' is the unit vector from that origin to the
    protrusion's mean vertex coordinate. The protrusion's own principal axes
    come from a PCA of its vertex coordinates, and each is signed to have a
    non-negative dot product with 'direction'. Axis lengths are the extent of
    the vertices along each axis (as for the cell axes in get_shape_info).

    Parameters
    ----------
    mesh : vtkPolyData
        Cell mesh in the alignment frame, centered on the origin. Need not be
        welded (it is cleaned internally; the input is not modified).
    df : pd.Series
        Row with
        - 'cell', the identifier stored in the 'cell' column of each result;
        - the Euler_Angles_X/Y/Z columns and the original-frame
          Cell_{Major,Median,Minor}_Axis_Vec_{X,Y,Z}, Trajectory_Vec_{X,Y,Z}
          and Next_Trajectory_Vec_{X,Y,Z} columns, which are rotated into the
          alignment frame here (see get_aligned_vectors). Taking the vectors
          from the same frame as the mesh keeps the alignments below correct
          for any alignment method. The cell axes' sign does not matter, they
          are given the directionality of the Cell_Aligned_*_Axis_Vec columns
          (see orient_cell_axes);
        - the x_raw/y_raw/z_raw columns, the cell's centroid in real space;
        - the shcoeffs_L{l}M{m}{C,S} columns, the SH coefficients of mesh as
          returned by get_shcoeffs_mesh (add them to the row with
          pd.concat([df, pd.Series(coeffs_dict)])).
    l_order : int
        Full spherical harmonics expansion order of the shcoeffs in df.
    params : ProtrusionParams, optional
        Detection and classification settings.
    return_labels : bool, optional
        If True, return (rows, mesh, label) instead of rows, where mesh is the
        cleaned mesh the protrusions were found on and label is a per-vertex
        array of region ids (0 = body). A row's region id is
        label[row['seed_vertex']]. Regions dropped by min_area_verts keep
        their id but have no row. Default False.

    Returns
    -------
    list of dict
        One dict per detected protrusion (empty if none pass the area
        filter), tallest first, with:
        - 'cell', 'CellID', 'frame', 'time' (the last three if they are in df),
          'protrusion_rank' (1 = tallest peak residual), 'seed_vertex',
          'n_vertices', 'peak_residual_um', 'area_um2', 'volume_um3'
        - 'direction_x/y/z': unit vector from the cell centroid to the protrusion
          in the alignment frame
        - 'real_direction_x/y/z': the same unit vector in real space, i.e. the
          original frame of the saved mesh and of the Trajectory_Vec and
          Cell_*_Axis_Vec columns (the alignment rotation undone)
        - 'position_x/y/z': the protrusion's mean vertex position relative to
          the cell centroid, in real space orientation (um), i.e.
          'real_direction' with its length kept
        - 'real_position_x/y/z': the protrusion's position in real space (um),
          i.e. 'position' plus the cell's centroid (the x_raw/y_raw/z_raw
          columns of df). 'position' is measured from the mean of the mesh
          vertices, which can differ from that centroid by up to about a micron
        - '{major,median,minor}_axis_length_um', 'aspect_ratio_median'
          (major/median), 'aspect_ratio_minor' (major/minor)
        - '{major,median,minor}_axis_{x,y,z}': protrusion principal axis vectors
          in the alignment frame, 'real_{major,median,minor}_axis_{x,y,z}': the
          same vectors in real space
        - 'trajectory_alignment', 'next_trajectory_alignment': dot product of
          'direction' with the unit trajectory vectors (signed: 1 = protrusion
          toward the trajectory, -1 = away from it)
        - 'cell_{major,median,minor}_axis_alignment': absolute dot product of
          'direction' with each cell principal axis (0..1, since the axes have
          no inherent directionality)
        - 'hull_volume_um3', 'hull_area_um2', 'hull_area_ratio' (area_um2 / hull
          area): the convex hull of the protrusion's vertices
        - 'protrusion_class': 'filopodia', 'lamellipodia', 'pseudopod' or
          'pseudopod_filopodia' (see _classify_protrusion)

    """

    #make sure mesh is clean before watershed
    mesh = shtools_mod.clean_polydata(mesh)

    if params is None:
        params = ProtrusionParams()

    ### SPHARM low-order residual: actual radius minus the smoothed envelope
    coeffs = df.filter(like = 'shcoeff').to_numpy(dtype=float)
    coeffs = coeffs.reshape(2, l_order+1, l_order+1)
    _, grid_low = shtools_mod.get_reconstruction_from_coeffs(coeffs, lrec=params.lrec)

    #grid_low is sampled on pyshtools's DH grid (sampling=2): row 0 = north
    #pole, increasing colatitude to (but not including) pi; column 0 =
    #longitude 0, increasing to (but not including) 2*pi. Wrap the longitude
    #axis so the interpolator below covers the full 0..2*pi range without a seam.
    n_lat, n_lon = grid_low.shape
    grid_colat = np.linspace(0, np.pi, n_lat, endpoint=False)
    grid_lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    grid_low_w = np.concatenate([grid_low, grid_low[:, :1]], axis=1)
    grid_lon_w = np.concatenate([grid_lon, [2 * np.pi]])
    interp_low = spinterp.RegularGridInterpolator(
        (grid_colat, grid_lon_w), grid_low_w, bounds_error=False, fill_value=None
    )

    coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    colat = np.arccos(np.divide(z, rad, out=np.zeros_like(rad), where=(rad != 0)))
    lon = np.pi + np.arctan2(y, x)
    r_low = interp_low(np.stack([colat, lon], axis=1))
    residual = rad - r_low
    n = len(residual)

    ### mesh graph (vertex adjacency + per-vertex area for later metrics)
    polys = numpy_support.vtk_to_numpy(mesh.GetPolys().GetData())
    tris = polys.reshape(-1, 4)[:, 1:4]

    neighbors = [[] for _ in range(n)]
    for a, b, c in tris:
        neighbors[a] += [b, c]
        neighbors[b] += [a, c]
        neighbors[c] += [a, b]

    tri_coords = coords[tris]
    tri_area = 0.5 * np.linalg.norm(
        np.cross(tri_coords[:, 1] - tri_coords[:, 0], tri_coords[:, 2] - tri_coords[:, 0]), axis=1
    )
    vertex_area = np.zeros(n)
    np.add.at(vertex_area, tris[:, 0], tri_area / 3)
    np.add.at(vertex_area, tris[:, 1], tri_area / 3)
    np.add.at(vertex_area, tris[:, 2], tri_area / 3)

    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [0, 2]]], axis=0)
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).flatten()
    degree[degree == 0] = 1

    #mild smoothing used only to seed/order peaks robustly against
    #single-vertex noise -- all reported metrics use the raw residual
    smoothed = residual.copy()
    for _ in range(params.smooth_iters):
        smoothed = adjacency.dot(smoothed) / degree

    ### persistence ("topographic prominence") over the mesh graph
    # Process vertices from highest to lowest smoothed residual. Each time a
    # vertex is reached with no already-processed neighbor, it starts a new
    # basin (a local peak). When a vertex is reached that borders two or
    # more existing basins, the lower one(s) are considered to have "died"
    # at that saddle height -- prominence = that basin's peak height minus
    # the saddle height -- and get merged into the higher basin. Exactly one
    # basin (the global maximum's) never dies.
    order = np.argsort(-smoothed, kind='stable')
    parent = np.arange(n)

    def find(v):
        root = v
        while parent[root] != root:
            root = parent[root]
        while parent[v] != root:
            parent[v], v = root, parent[v]
        return root

    processed = np.zeros(n, dtype=bool)
    root_peak_value = {}
    root_peak_vertex = {}
    prominence_records = []

    for v in order:
        processed[v] = True
        touched = {find(nb) for nb in neighbors[v] if processed[nb]}
        if not touched:
            parent[v] = v
            root_peak_value[v] = smoothed[v]
            root_peak_vertex[v] = v
        elif len(touched) == 1:
            parent[v] = next(iter(touched))
        else:
            ranked = sorted(touched, key=lambda r: -root_peak_value[r])
            survivor = ranked[0]
            for loser in ranked[1:]:
                prominence_records.append({
                    'peak_vertex': root_peak_vertex[loser],
                    'prominence': root_peak_value[loser] - smoothed[v],
                })
                parent[loser] = survivor
                del root_peak_value[loser]
                del root_peak_vertex[loser]
            parent[v] = survivor

    #the one basin still alive at the end (the global maximum) never
    #"died", so it has no finite prominence -- treat it as automatically significant
    global_peak_vertex = next(iter(root_peak_vertex.values()))
    prominence_df = pd.DataFrame(prominence_records)

    median = np.median(residual)
    robust_std = 1.4826 * np.median(np.abs(residual - median))
    prominence_thresh = params.prominence_sigmas * robust_std
    grow_thresh = median + params.grow_sigmas * robust_std

    if len(prominence_df) > 0:
        significant = prominence_df[prominence_df.prominence > prominence_thresh]
        seed_vertices = [global_peak_vertex] + significant.sort_values(
            'prominence', ascending=False).peak_vertex.tolist()
    else:
        seed_vertices = [global_peak_vertex]

    ### seeded flood-fill (multi-source BFS): grow each significant peak
    ### outward until the region drops back to the grow threshold. This must
    ### be a BFS, not a single ordered sweep from high to low value -- a
    ### small ruffle on the shoulder of a real protrusion is itself a local
    ### bump relative to its immediate neighbors (just not significant
    ### enough to be its own seed); in a single top-down sweep it would be
    ### visited before its lower neighbors, find none of them labeled yet,
    ### and be skipped permanently, leaving an unlabeled island in the
    ### middle of an otherwise-correct region.
    label = np.zeros(n, dtype=np.int64)
    queue = deque()
    for i, seed in enumerate(seed_vertices, start=1):
        label[seed] = i
        queue.append(seed)
    while queue:
        v = queue.popleft()
        for nb in neighbors[v]:
            if label[nb] == 0 and smoothed[nb] > grow_thresh:
                label[nb] = label[v]
                queue.append(nb)

    ### vectors to compare each protrusion's direction against, all in the
    ### same (alignment) frame as the mesh
    cell_evecs_aligned, trajectory_aligned, next_trajectory_aligned = get_aligned_vectors(df)
    cell_axes = orient_cell_axes(cell_evecs_aligned)
    cell_centroid = df[['x_raw', 'y_raw', 'z_raw']].values.astype(float)
    #rotation taking the alignment frame back to the original (real space) frame of the saved mesh
    to_real_space = R.from_euler(
        'xyz', df[['Euler_Angles_X','Euler_Angles_Y','Euler_Angles_Z']].values, degrees = True).inv()
    trajectory_unit = _unit_vector(trajectory_aligned)
    next_trajectory_unit = _unit_vector(next_trajectory_aligned)

    ### per-protrusion metrics, dropping any that end up too small
    rows = []
    for i in range(1, len(seed_vertices) + 1):
        verts = np.where(label == i)[0]
        if len(verts) < params.min_area_verts:
            continue
        pcoords = coords[verts]
        direction = _unit_vector(pcoords.mean(axis=0))
        volume = (residual[verts] * vertex_area[verts]).clip(min=0).sum()

        # principal axes of the protrusion's own vertex coordinates, major to
        # minor (xyz components), each pointing away from the cell centroid
        evals, evecs = np.linalg.eigh(np.cov(pcoords.T))
        axes = evecs[:, np.argsort(evals)[::-1]].T
        axes = axes * np.where(axes @ direction < 0, -1.0, 1.0)[:, None]
        # axis lengths as the extent of the vertices along each axis
        projected = (pcoords - pcoords.mean(axis=0)) @ axes.T
        lengths = projected.max(axis=0) - projected.min(axis=0)
        aspect_ratio_median = lengths[0] / lengths[1] if lengths[1] > 0 else np.nan
        aspect_ratio_minor = lengths[0] / lengths[2] if lengths[2] > 0 else np.nan

        # convex hull of the protrusion; degenerate (e.g. flat) patches have none
        try:
            hull = ConvexHull(pcoords)
            hull_volume, hull_area = hull.volume, hull.area
        except QhullError:
            hull_volume = hull_area = np.nan
        hull_area_ratio = vertex_area[verts].sum() / hull_area if hull_area > 0 else np.nan

        row = {
            'cell': df.cell,
            **{col: df[col] for col in ('CellID', 'frame', 'time') if col in df.index},
            'seed_vertex': seed_vertices[i - 1],
            'n_vertices': len(verts),
            'peak_residual_um': residual[verts].max(),
            'area_um2': vertex_area[verts].sum(),
            'volume_um3': volume,
            'direction_x': direction[0],
            'direction_y': direction[1],
            'direction_z': direction[2],
            'major_axis_length_um': lengths[0],
            'median_axis_length_um': lengths[1],
            'minor_axis_length_um': lengths[2],
            'aspect_ratio_median': aspect_ratio_median,
            'aspect_ratio_minor': aspect_ratio_minor,
        }
        real_axes = to_real_space.apply(axes)
        for ax_name, ax, real_ax in zip(ax_names, axes, real_axes):
            row.update({f'{ax_name.lower()}_axis_{d}': ax[k] for k, d in enumerate('xyz')})
            row.update({f'real_{ax_name.lower()}_axis_{d}': real_ax[k] for k, d in enumerate('xyz')})
        real_direction = to_real_space.apply(direction)
        row.update({f'real_direction_{d}': real_direction[k] for k, d in enumerate('xyz')})
        # unnormalized: mean vertex position of the protrusion relative to the cell centroid
        position = to_real_space.apply(pcoords.mean(axis=0))
        row.update({f'position_{d}': position[k] for k, d in enumerate('xyz')})
        row.update({f'real_position_{d}': cell_centroid[k] + position[k] for k, d in enumerate('xyz')})
        row.update({
            'trajectory_alignment': direction @ trajectory_unit,
            'next_trajectory_alignment': direction @ next_trajectory_unit,
        })
        # cell axes have no inherent directionality, so only the magnitude of
        # the alignment is meaningful
        row.update({
            f'cell_{ax_name.lower()}_axis_alignment': abs(direction @ cell_ax)
            for ax_name, cell_ax in zip(ax_names, cell_axes)
        })
        row.update({
            'hull_volume_um3': hull_volume,
            'hull_area_um2': hull_area,
            'hull_area_ratio': hull_area_ratio,
        })
        row['protrusion_class'] = _classify_protrusion(row, params)
        rows.append(row)

    rows.sort(key=lambda r: -r['peak_residual_um'])
    for rank, r in enumerate(rows, start=1):
        r['protrusion_rank'] = rank

    if return_labels:
        return rows, mesh, label
    return rows


def get_protrusion_info(
        df: pd.Series,  # row with 'cell', Euler angles, principal axes, trajectory vectors and shcoeffs
        meshdir: Path,  # path to the mesh directory
        l_order: int, # degree of the original shcoeffs
        params: ProtrusionParams = None,  # detection and classification settings, default ProtrusionParams()
        ):
    """
    Measure the protrusions of a single cell on its own, i.e. without the rest
    of get_shape_info: read its mesh, rotate it into the alignment frame, fit
    SH coefficients and pass them to measure_protrusions_from_mesh (see there
    for the algorithm, the settings and the returned columns).

    Parameters
    ----------
    df : pd.Series
        Row with 'cell', the Euler_Angles_X/Y/Z columns used to rotate the
        mesh into the alignment frame, and the Cell_{Major,Median,Minor}_Axis_Vec_{X,Y,Z},
        Trajectory_Vec_{X,Y,Z} and Next_Trajectory_Vec_{X,Y,Z} columns (same
        convention as get_shape_info). Direction outputs are expressed in
        this frame.
    meshdir : Path
        Directory containing '{cell}_cell_mesh.vtp' files.
    params : ProtrusionParams, optional
        Detection and classification settings.

    Returns
    -------
    list of dict
        One dict per detected protrusion, see measure_protrusions_from_mesh.
    """

    ### read and align the mesh exactly as get_shape_info does
    cell_mesh = get_aligned_mesh(df, meshdir)

    return measure_protrusions_from_mesh(
        cell_mesh,
        df,
        l_order,
        params,
        )


#wrapper for get_protrusion_info for imap
def protrusion_info_imap(args):
    return get_protrusion_info(*args)