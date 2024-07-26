# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:43:34 2024

@author: Aaron
"""


from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from aicsimageio.readers.tiff_reader import TiffReader


im = TiffReader('E:/Aaron/Chemotaxis_iSIM/Processed_Data/PILRs/20230511_488EGFP-CAAX_640JF646actin-halotag1_cell_1_frame_46_PILR.tiff')

if ('im' in locals()) and (im.data.max()==0):
    print('yes')
    
    
i = '20230524_488EGFP-CAAX_640SPY650-DNA3_cell_10_frame_6_segmented.tiff'
    
base = 'E:/Aaron/Chemotaxis_iSIM/'
savedir = base+ 'Processed_Data/'


#make dirs if it doesn't exist
datadir = base+'Data_and_Figs/'
if not os.path.exists(datadir):
    os.makedirs(datadir)
meshf = savedir+'Meshes/'  
if not os.path.exists(meshf):
    os.makedirs(meshf)
pilrf = savedir+'PILRs/'
if not os.path.exists(pilrf):
    os.makedirs(pilrf)
    
xyres = 0.1613 #um / pixel 
zstep = 0.5 # um
align_method = 'trajectory'
norm_rot = 'provided'
l_order = 10
nisos = [1,63]
sigma = 0
errorlist = []

if norm_rot == 'provided':
    widthpeaks = pd.read_csv(datadir + 'Closest_Width_Peaks.csv', index_col = 0)

norm_rot = float(widthpeaks[widthpeaks.cell == i.split('_segment')[0]]['Closest_minimums'].values[0])
normal_rotation_method = norm_rot

impath = savedir + i
str_name = 'nucleus'

def shcoeffs_and_PILR_nonuc(
        impath: str,
        savedir: str,
        xyres: float,
        zstep: float,
        str_name: str,
        exceptions_list: list,
        normal_rotation_method: str,
        l_order: int,
        nisos: int,
        sigma: float = 0,
        align_method: str = 'None',
        ):

    """
        Parameters
        ----------
        impath : str
            Input image path. Expected to have shape CZYX. Channel order must be Membrane, Nucleus, Structure
        savedir : str
            Directory that will contain the Mesh and PILR folders
        xyres : float
            microns/pixel resolution of the image
        zstep : float
            Z step of the image
        normal_rotation_method : str
            "widest" is longest axis parallel to trajectory
        exceptions_list : List
            List to append names of images that cannot be represented well by SH
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
    
    #get cell name from impath
    cell_name = impath.split('/')[-1].split('_segmented')[0]
    #read image
    im = TiffReader(impath)
    
    #read euler angles for alignment
    infopath = '/'.join(impath.split('/')[:-1]) + '/' + cell_name + '_cell_info.csv'
    #if align_method is a numpy array, use that as the vector to align to
    if type(align_method) == np.ndarray:
        vec = align_method.copy()
        #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        xaxis = np.array([[1,0,0], [0,1,0], [0,0,1]]).astype('float64')
        upnorm = np.cross(vec,[1,0,0])
        # if upnorm[2]<0:
        #     upnorm = upnorm.copy() * -1
        sidenorm = np.cross(vec,upnorm)
        current_vec = np.stack((vec, sidenorm, upnorm), axis = 0)
        # current_vec = np.concatenate((current_vec,[5*vec]), axis = 0)
        rotationthing = R.align_vectors(xaxis, current_vec)
        #below is actual rotation matrix if needed
        #rot_mat = rotationthing[0].as_matrix()
        rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)
        euler_angles = np.array([rotthing_euler[0], rotthing_euler[1], rotthing_euler[2]])
    elif align_method == 'trajectory':
        info = pd.read_csv(infopath, index_col=0)
        vec = np.array([info.Trajectory_X[0], info.Trajectory_Y[0], info.Trajectory_Z[0]])
        #align current vector with x axis and get euler angles of resulting rotation matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        xaxis = np.array([[1,0,0], [0,1,0], [0,0,1]]).astype('float64')
        upnorm = np.cross(vec,[1,0,0])
        # if upnorm[2]<0:
        #     upnorm = upnorm.copy() * -1
        sidenorm = np.cross(vec,upnorm)
        current_vec = np.stack((vec, sidenorm, upnorm), axis = 0)
        # current_vec = np.concatenate((current_vec,[5*vec]), axis = 0)
        rotationthing = R.align_vectors(xaxis, current_vec)
        #below is actual rotation matrix if needed
        #rot_mat = rotationthing[0].as_matrix()
        rotthing_euler = rotationthing[0].as_euler('xyz', degrees = True)
        euler_angles = np.array([rotthing_euler[0], rotthing_euler[1], rotthing_euler[2]])
        
    if len(im.shape)>3:
        ci = im.data[0,:,:,:]
        si = im.data[1,:,:,:]
    else:
        ci = im.data
        
    #get cell mesh and coeffs
    (coeffs_mem, grid_rec, widestangle, exceptions_list), (image_, cell_mesh, grid, centroid_mem) = get_shcoeffs_mod(        
        image = ci,
        img_name= cell_name,
        lmax = l_order,
        xyres = xyres/xyres, #use pixels and not microns because I'll need pixel dimensions for PILRs
        zstep = zstep/xyres,
        Euler_Angles = euler_angles,
        exceptions_list = exceptions_list,
        sigma = sigma,
        normal_rotation_method = normal_rotation_method,
        )
    
    
    #get reconstruction errors both ways
    cell_recon, grid_recon = shtools_mod.get_reconstruction_from_coeffs(np.array(list(coeffs_mem.values())).reshape(2,l_order+1,l_order+1))
    #get average nearest distance from original mesh to reconstruction
    tree = KDTree(numpy_support.vtk_to_numpy(cell_mesh.GetPoints().GetData()))
    d, idx = tree.query(numpy_support.vtk_to_numpy(cell_recon.GetPoints().GetData()))
    OriginaltoReconError = np.mean(d)
    #get average nearest distance from reconstruction to original mesh
    tree = KDTree(numpy_support.vtk_to_numpy(cell_recon.GetPoints().GetData()))
    d, idx = tree.query(numpy_support.vtk_to_numpy(cell_mesh.GetPoints().GetData()))
    RecontoOriginalError = np.mean(d)
    
    
    if ('si' in locals()) and (si.max()>0):
        #create inner sphere
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(0.0, 0.0, 0.0)
        sphereSource.SetRadius(nisos[0]/2)
        # Make the surface smooth.
        sphereSource.SetPhiResolution(100)
        sphereSource.SetThetaResolution(100)
        sphereSource.Update()
        spherepoly = sphereSource.GetOutput()
        
        
        (sphere_coeffs, grid_rec), (grid_down) = get_shcoeffs_mesh(
                spherepoly,
                lmax= l_order)
    
        
        #provide escape for cells with no signal in the "structure channel"
        if np.max(si) == 0:
            #get voxelized intracellular structure image
            img, origin = cytoparam_mod.voxelize_meshes([cell_mesh])
        
            #get rotated segmentented str signal alone
            strimg = img.copy()
            strimg[strimg>0]=255
        
        else:
            #get structure mesh
            str_mesh, _, cent = shtools_mod.get_mesh_from_image(
                image = si,
                translate_to_origin=False,
                lcc = False,
                center = np.array(centroid_mem)[0]
                )
            #euler rotation and scaling
            str_mesh = rotate_and_scale_mesh(str_mesh,
                                              euler_angles,
                                              np.array([xyres, xyres, zstep])/xyres)
            #widest angle rotation
            str_mesh = rotate_and_scale_mesh(str_mesh,
                                              rotations = np.array([widestangle,0,0]))
            #adjust the structure center to the final position of the cell after rotation
            coords = numpy_support.vtk_to_numpy(str_mesh.GetPoints().GetData())
            coords -= np.array(centroid_mem)[1]
            str_mesh = update_mesh_points(str_mesh, coords[:, 0], coords[:, 1], coords[:, 2])
            
            #get voxelized intracellular structure image
            img, origin = cytoparam_mod.voxelize_meshes([cell_mesh,str_mesh])
            
            #get rotated segmentented str signal alone
            strimg = img.copy()
            strimg[strimg<2]=0
            strimg[strimg>0]=255
                    
            #scale structure mesh
            #set transform and apply
            meshf = savedir+'Meshes/'
            transformation = vtk.vtkTransform()
            transformation.Scale(xyres, xyres, xyres)
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(transformation)
            transformFilter.SetInputData(str_mesh)
            transformFilter.Update()
            str_mesh = transformFilter.GetOutput()
            #save str mesh
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(meshf + cell_name + '_str_mesh.vtp')
            writer.SetInputData(str_mesh)
            writer.Write()
            

        #########parameterize cell
        aicstif = cytoparam_mod.cellular_mapping(
            coeffs_mem = coeffs_mem,
            centroid_mem = abs(origin)[0],
            coeffs_nuc = sphere_coeffs,
            centroid_nuc = abs(origin)[0],
            nisos = nisos,
            images_to_probe = [[str_name,strimg]],
            )
              
        #Save PILR
        pilrf = savedir+'PILRs/'
        if os.path.exists(pilrf+cell_name+'_PILR.tiff'):
            os.remove(pilrf+cell_name+'_PILR.tiff')
        OmeTiffWriter.save(aicstif.get_image_data('CZYX', S=0, T=0), pilrf+cell_name+'_PILR.tiff', dim_order='CZYX', channel_names=aicstif.channel_names)
        
        
    #now that PILR has been made,
    #scale cell and nuc meshes so that I can take some shape stats
    #set transform and apply to cell
    transformation = vtk.vtkTransform()
    transformation.Scale(xyres, xyres, xyres)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transformation)
    transformFilter.SetInputData(cell_mesh)
    transformFilter.Update()
    cell_mesh = transformFilter.GetOutput()
    
    
    
    
    # remove file if it already exists
    meshf = savedir+'Meshes/'
    #save cell mesh
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(meshf + cell_name + '_cell_mesh.vtp')
    writer.SetInputData(cell_mesh)
    writer.Write()
    
    
    #Get physical properties of cell
    CellMassProperties = vtk.vtkMassProperties()
    CellMassProperties.SetInputData(cell_mesh)
    
    
    
    #get cell major, minor, and mini axes using the segmented image
    cell_coords = numpy_support.vtk_to_numpy(cell_mesh.GetPoints().GetData())
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
    #rotate the cell coordinates to align the major axis with the x, the minor axis to the y and the "mini" axis to the z
    rotationthing = R.align_vectors(np.array([[1,0,0],[0,1,0]]), cell_evecs.T[:2,:])
    cell_coords = rotationthing[0].apply(cell_coords)
    
    
    #measure the volume of the cell in the x domain
    FrontVolume = measure_volume_half(cell_mesh, 'x')
    RightVolume = measure_volume_half(cell_mesh, 'y')
    TopVolume = measure_volume_half(cell_mesh, 'z')
    
    
    #Shape stats dict
    Shape_Stats = {'cell': cell_name,
                   'Euler_angles_X': euler_angles[0],
                   'Euler_angles_Y':euler_angles[1],
                   'Euler_angles_Z':euler_angles[2],
                   'Width_Rotation_Angle': widestangle,
                  'Cell_Centroid_X': centroid_mem[0],
                  'Cell_Centroid_Y': centroid_mem[1],
                  'Cell_Centroid_Z': centroid_mem[2],
                   'Cell_Volume': CellMassProperties.GetVolume(),
                    'Cell_Volume_Front': FrontVolume,
                    'Cell_Volume_Right': RightVolume,
                    'Cell_Volume_Top': TopVolume,
                   'Cell_SurfaceArea': CellMassProperties.GetSurfaceArea(),
                   'Cell_MajorAxis': np.max(cell_coords[:,0])-np.min(cell_coords[:,0]),
                   'Cell_MajorAxis_Vec_X': cell_evecs[0,0],
                   'Cell_MajorAxis_Vec_Y': cell_evecs[1,0],
                   'Cell_MajorAxis_Vec_Z': cell_evecs[2,0],
                   'Cell_MinorAxis': np.max(cell_coords[:,1])-np.min(cell_coords[:,1]),
                   'Cell_MinorAxis_Vec_X': cell_evecs[0,1],
                   'Cell_MinorAxis_Vec_Y': cell_evecs[1,1],
                   'Cell_MinorAxis_Vec_Z': cell_evecs[2,1],
                   'Cell_MiniAxis': np.max(cell_coords[:,2])-np.min(cell_coords[:,2]),
                   'Cell_MiniAxis_Vec_X': cell_evecs[0,2],
                   'Cell_MiniAxis_Vec_Y': cell_evecs[1,2],
                   'Cell_MiniAxis_Vec_Z': cell_evecs[2,2],
                   'OriginaltoReconError': OriginaltoReconError,
                   'RecontoOriginalError': RecontoOriginalError
                    }


    return Shape_Stats, coeffs_mem, exceptions_list






def voxelize_mesh(
    imagedata: vtk.vtkImageData, shape: Tuple, mesh: vtk.vtkPolyData, origin: List
):

    """
    Voxelize a triangle mesh into an image.

    Parameters
    --------------------
    imagedata: vtkImageData
        Imagedata that will be uses as support for voxelization.
    shape: tuple
        Shape that imagedata scalars will take after
        voxelization.
    mesh: vtkPolyData
        Mesh to be voxelized
    origin: List
        xyz specifying the lower left corner of the mesh.

    Returns
    -------
    img: np.array
        Binary array.
    """

    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(mesh)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputWholeExtent(imagedata.GetExtent())
    pol2stenc.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(imagedata)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()

    # Convert scalars from vtkImageData back to numpy
    scalars = imgstenc.GetOutput().GetPointData().GetScalars()
    img = vtk_to_numpy(scalars).reshape(shape)

    return img




def voxelize_meshes(meshes: List):

    """
    List of meshes to be voxelized into an image. Usually
    the input corresponds to the cell membrane and nuclear
    shell meshes.

    Parameters
    --------------------
    meshes: List
        List of vtkPolydatas representing the meshes to
        be voxelized into an image.
    Returns
    -------
    img: np.array
        3D image where voxels with value i represent are
        those found in the interior of the i-th mesh in
        the input list. If a voxel is interior to one or
        more meshes form the input list, it will take the
        value of the right most mesh in the list.
    origin:
        Origin of the meshes in the voxelized image.
    """

    # 1st mesh is used as reference (cell) and it should be
    # the larger than the 2nd one (nucleus).
    mesh = meshes[0]

    # Find mesh coordinates
    coords = vtk_to_numpy(mesh.GetPoints().GetData())

    # Find bounds of the mesh
    rmin = (coords.min(axis=0) - 0.5).astype(np.int)
    rmax = (coords.max(axis=0) + 0.5).astype(np.int)

    # Width, height and depth
    w = int(2 + (rmax[0] - rmin[0]))
    h = int(2 + (rmax[1] - rmin[1]))
    d = int(2 + (rmax[2] - rmin[2]))

    # Create image data
    imagedata = vtk.vtkImageData()
    imagedata.SetDimensions([w, h, d])
    imagedata.SetExtent(0, w - 1, 0, h - 1, 0, d - 1)
    imagedata.SetOrigin(rmin)
    imagedata.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    # Set all values to 1
    imagedata.GetPointData().GetScalars().FillComponent(0, 1)

    # Create an empty 3D numpy array to sum up
    # voxelization of all meshes
    img = np.zeros((d, h, w), dtype=np.uint8)

    # Voxelize one mesh at the time
    for mid, mesh in enumerate(meshes):
        seg = voxelize_mesh(
            imagedata=imagedata, shape=(d, h, w), mesh=mesh, origin=rmin
        )
        img[seg > 0] = mid + 1

    # Origin of the reference system in the image
    origin = rmin.reshape(1, 3)

    return img, origin




def get_mesh_from_image_and_rotate(
    image: np.array,
    xyres: float,
    zstep: float,
    Euler_Angles: np.array,
    sigma: float = 0,
    provided_normal_rotation_angle: float = 0,
    lcc: bool = True,
    translate_to_origin: bool = True,
    center: np.array = None,
):

    """Converts a numpy array into a vtkImageData and then into a 3d mesh
    using vtkContourFilter. The input is assumed to be binary and the
    isosurface value is set to 0.5.

    Optionally the input can be pre-processed by i) extracting the largest
    connected component and ii) applying a gaussian smooth to it. In case
    smooth is used, the image is binarize using thershold 1/e.

    A size threshold is applying to garantee that enough points will be
    used to compute the SH parametrization.

    Also, points as the edge of the image are set to zero (background)
    to make sure the mesh forms a manifold.

    Parameters
    ----------
    image : np.array
        Input array where the mesh will be computed on
    Returns
    -------
    mesh : vtkPolyData
        3d mesh in VTK format
    img_output : np.array
        Input image after pre-processing
    centroid : np.array
        x, y, z coordinates of the mesh centroid

    Other parameters
    ----------------
    lcc : bool, optional
        Wheather or not to compute the mesh only on the largest
        connected component found in the input connected component,
        default is True.
    sigma : float, optional
        The degree of smooth to be applied to the input image, default
        is 0 (no smooth).
    translate_to_origin : bool, optional
        Wheather or not translate the mesh to the origin (0,0,0),
        default is True.
    """

    img = image.copy()

    # VTK requires YXZ
    img = np.swapaxes(img, 0, 2)

    # Extracting the largest connected component
    if lcc:

        img = skmorpho.label(img.astype(np.uint8))

        counts = np.bincount(img.flatten())

        lcc = 1 + np.argmax(counts[1:])

        img[img != lcc] = 0
        img[img == lcc] = 1

    # Smooth binarize the input image and binarize
    if sigma:

        img = skfilters.gaussian(img.astype(np.float32), sigma=(sigma, sigma, sigma))

        img[img < 1.0 / np.exp(1.0)] = 0
        img[img > 0] = 1

        if img.sum() == 0:
            raise ValueError(
                "No foreground voxels found after pre-processing. Try using sigma=0."
            )

    # Set image border to 0 so that the mesh forms a manifold
    img[[0, -1], :, :] = 0
    img[:, [0, -1], :] = 0
    img[:, :, [0, -1]] = 0
    img = img.astype(np.float32)

    if img.sum() == 0:
        raise ValueError(
            "No foreground voxels found after pre-processing."
            "Is the object of interest centered?"
        )

    # Create vtkImageData
    imgdata = vtk.vtkImageData()
    imgdata.SetDimensions(img.shape)

    img = img.transpose(2, 1, 0)
    img_output = img.copy()
    img = img.flatten()
    arr = numpy_support.numpy_to_vtk(img, array_type=vtk.VTK_FLOAT)
    arr.SetName("Scalar")
    imgdata.GetPointData().SetScalars(arr)

    # Create 3d mesh
    cf = vtk.vtkContourFilter()
    cf.SetInputData(imgdata)
    cf.SetValue(0, 0.5)
    cf.Update()

    mesh = cf.GetOutput()

 
    
    #rotate and scale mesh
    #worked from https://kitware.github.io/vtk-examples/site/Python/PolyData/AlignTwoPolyDatas/
    #rotate around z axis
    transformation = vtk.vtkTransform()
    #rotate the shape
    transformation.RotateWXYZ(Euler_Angles[2], 0, 0, 1)
    transformation.RotateWXYZ(Euler_Angles[0], 1, 0, 0)
    #set scale to actual image scale
    transformation.Scale(xyres, xyres, zstep)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transformation)
    transformFilter.SetInputData(mesh)
    transformFilter.Update()
    mesh = transformFilter.GetOutput()
    


    #rotate the mesh around x by a specific angle
    if provided_normal_rotation_angle>0:
        #rotate around x by the widest y axis
        transformation = vtk.vtkTransform()
        #rotate the shape
        transformation.RotateWXYZ(provided_normal_rotation_angle, 1, 0, 0)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transformation)
        transformFilter.SetInputData(mesh)
        transformFilter.Update()
        mesh = transformFilter.GetOutput()


    if translate_to_origin:
        # Calculate the mesh centroid
        coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
        centroid = coords.mean(axis=0, keepdims=True)
    
        # Translate to origin
        coords -= centroid
        mesh = shtools_mod.update_mesh_points(mesh, coords[:, 0], coords[:, 1], coords[:, 2])
    else:
        coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
        centroid = center.copy()
        coords -= centroid 
        mesh = shtools_mod.update_mesh_points(mesh, coords[:, 0], coords[:, 1], coords[:, 2])


    return mesh, img_output, tuple(centroid.squeeze())



