import vtk
import pyshtools
import numpy as np
from typing import Tuple
from vtk.util import numpy_support
from skimage import transform as sktrans
from skimage import filters as skfilters
from skimage import morphology as skmorpho
from sklearn import decomposition as skdecomp
from scipy import interpolate as sciinterp
from scipy import stats as scistats
from vtkmodules.vtkFiltersCore import (
    vtkCleanPolyData,
    vtkTriangleFilter
)

EPS = 1e-12


# clean up a messy mesh 
def clean_polydata(mesh: vtk.vtkPolyData,):
    tri1 = vtkTriangleFilter()
    tri1.SetInputData(mesh)
    clean1 = vtkCleanPolyData()
    clean1.SetInputConnection(tri1.GetOutputPort())
    clean1.Update()
    output = clean1.GetOutput()
    return output
    


def rotate_and_scale_mesh(
        mesh,
        rotations: np.array = np.array([0,0,0]), #rotations in [x,y,z]
        scale: np.array = np.array([1,1,1]), #scaling factors for dimensions in [x,y,z]
        ):
    #rotate and scale mesh
    #worked from https://kitware.github.io/vtk-examples/site/Python/PolyData/AlignTwoPolyDatas/
    transformation = vtk.vtkTransform()
    #input rotations
    transformation.RotateWXYZ(rotations[2], 0, 0, 1)
    transformation.RotateWXYZ(rotations[1], 0, 1, 0)
    transformation.RotateWXYZ(rotations[0], 1, 0, 0)
    #input scaling
    transformation.Scale(scale[0], scale[1], scale[2])
    #apply transformations
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transformation)
    transformFilter.SetInputData(mesh)
    transformFilter.Update()
    
    return transformFilter.GetOutput()
    

def update_mesh_points(
    mesh: vtk.vtkPolyData, x_new: np.array, y_new: np.array, z_new: np.array
):

    """Updates the xyz coordinates of points in the input mesh
    with new coordinates provided.

    Parameters
    ----------
    mesh : vtkPolyData
        Mesh in VTK format to be updated.
    x_new, y_new and z_new : np.array
        Array containing the new coordinates.
    Returns
    -------
    mesh_updated : vtkPolyData
        Mesh with updated coordinates.

    Notes
    -----
    This function also re-calculate the new normal vectors
    for the updated mesh.
    """

    mesh.GetPoints().SetData(numpy_support.numpy_to_vtk(np.c_[x_new, y_new, z_new]))
    mesh.Modified()

    # Fix normal vectors orientation

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.Update()

    mesh_updated = normals.GetOutput()

    return mesh_updated


def get_mesh_from_image(
    image: np.array,
    sigma: float = 0,
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

    if translate_to_origin:
        # Calculate the mesh centroid
        coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
        centroid = coords.mean(axis=0, keepdims=True)
    
        #subtract centroid from coordinates
        coords -= centroid
        # Translate to origin
        mesh = update_mesh_points(mesh, coords[:, 0], coords[:, 1], coords[:, 2])

    else:
        #this will translate to "center", a optional specificed point
        coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
        centroid = center.copy()
        coords -= centroid
        mesh = update_mesh_points(mesh, coords[:, 0], coords[:, 1], coords[:, 2])

    
    return mesh, img_output, tuple(centroid.squeeze())




def rotate_image_2d(image: np.array, angle: float, interpolation_order: int = 0):

    """Rotate multichannel image in 2D by a given angle. The
    expected shape of image is (C,Z,Y,X). The rotation will
    be done clock-wise around the center of the image.

    Parameters
    ----------
    angle : float
        Angle in degrees
    interpolation_order : int
        Order of interpolation used during the image rotation
    Returns
    -------
    img_rot : np.array
        Rotated image
    """

    if image.ndim != 4:
        raise ValueError(
            f"Invalid shape {image.shape} of input image."
            "Expected 4 dimensional images as input."
        )

    if not isinstance(interpolation_order, int):
        raise ValueError("Only integer values are accepted for interpolation order.")

    # Make z to be the last axis. Required for skimage rotation
    image = np.swapaxes(image, 1, 3)

    img_aligned = []
    for stack in image:
        stack_aligned = sktrans.rotate(
            image=stack,
            angle=-angle,
            resize=True,
            order=interpolation_order,
            preserve_range=True,
        )
        img_aligned.append(stack_aligned)
    img_aligned = np.array(img_aligned)

    # Swap axes back
    img_aligned = np.swapaxes(img_aligned, 1, 3)

    img_aligned = img_aligned.astype(image.dtype)

    return img_aligned


def align_image_2d(
    image: np.array, alignment_channel: int = None, make_unique: bool = False
):

    """Align a multichannel 3D image based on the channel
    specified by alignment_channel. The expected shape of
    image is (C,Z,Y,X) or (Z,Y,X).

    Parameters
    ----------
    image : np.array
        Input array of shape (C,Z,Y,X) or (Z,Y,X).
    alignment_channel : int
        Number of channel to be used as reference for alignemnt. The
        alignment will be propagated to all other channels.
    make_unique : bool
        Set true to make sure the alignment rotation is unique.
    Returns
    -------
    img_aligned : np.array
        Aligned image
    angle : float
        Angle used for align the shape.
    """

    if image.ndim not in [3, 4]:
        raise ValueError(f"Invalid shape {image.shape} of input image.")

    if image.ndim == 4:
        if alignment_channel is None:
            raise ValueError(
                "An alignment channel must be provided with multichannel images."
            )
        if not isinstance(alignment_channel, int):
            raise ValueError("Number of alignment channel must be an integer")

    if image.ndim == 3:
        alignment_channel = 0
        image = image.reshape(1, *image.shape)

    z, y, x = np.where(image[alignment_channel])

    xy = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])

    pca = skdecomp.PCA(n_components=2)

    pca = pca.fit(xy)

    eigenvecs = pca.components_

    if make_unique:

        # Calculate angle with arctan2
        angle = 180.0 * np.arctan2(eigenvecs[0][1], eigenvecs[0][0]) / np.pi

        # Rotate x coordinates
        x_rot = (x - x.mean()) * np.cos(np.pi * angle / 180) + (y - y.mean()) * np.sin(
            np.pi * angle / 180
        )

        # Check the skewness of the rotated x coordinate
        xsk = scistats.skew(x_rot)
        if xsk < 0.0:
            angle += 180

        # Map all angles to anti-clockwise
        angle = angle % 360

    else:

        # Calculate smallest angle
        angle = 0.0
        if np.abs(eigenvecs[0][0]) > EPS:
            angle = 180.0 * np.arctan(eigenvecs[0][1] / eigenvecs[0][0]) / np.pi

    # Apply skimage rotation clock-wise
    img_aligned = rotate_image_2d(image=image, angle=angle)

    return img_aligned, angle


def apply_image_alignment_2d(image: np.array, angle: float):

    """Apply an existing set of alignment parameters to a
    multichannel 3D image. The expected shape of
    image is (C,Z,Y,X) or (Z,Y,X).

    Parameters
    ----------
    image : np.array
        Input array of shape (C,Z,Y,X) or (Z,Y,X).
    angle : float
        2D rotation angle in degrees
    Returns
    -------
    img_aligned : np.array
        Aligned image
    """

    if image.ndim not in [3, 4]:
        raise ValueError(f"Invalid shape {image.shape} of input image.")

    if image.ndim == 3:
        image = image.reshape(1, *image.shape)

    img_aligned = rotate_image_2d(image=image, angle=angle)

    return img_aligned




def get_even_reconstruction_from_grid(
    grid: np.array, npoints: int = 512, centroid: Tuple = (0, 0, 0)
):

    """Converts a parametric 2D grid of type (lon,lat,rad) into
    a 3d mesh. lon in [0,2pi], lat in [0,pi]. The method uses
    a spherical mesh with an even distribution of points. The
    even distribution is obtained via the Fibonacci grid rule.

    Parameters
    ----------
    grid : np.array
        Input grid where the element grid[i,j] represents the
        radial coordinate at longitude i*2pi/grid.shape[0] and
        latitude j*pi/grid.shape[1].

    Returns
    -------
    mesh : vtkPolyData
        Mesh that represents the input parametric grid.

    Other parameters
    ----------------
    npoints: int
        Number of points in the initial spherical mesh
    centroid : tuple of floats, optional
        x, y and z coordinates of the centroid where the mesh
        will be translated to, default is (0,0,0).
    """

    res_lat = grid.shape[0]
    res_lon = grid.shape[1]

    # Creates an interpolator
    lon = np.linspace(start=0, stop=2 * np.pi, num=res_lon, endpoint=True)
    lat = np.linspace(start=0, stop=1 * np.pi, num=res_lat, endpoint=True)

    fgrid = sciinterp.RectBivariateSpline(lon, lat, grid.T)

    # Create x,y,z coordinates based on the Fibonacci Lattice
    # http://extremelearning.com.au/evenly-distributing-points-on-a-sphere/
    golden_ratio = 0.5 * (1 + np.sqrt(5))
    idxs = np.arange(0, npoints, dtype=np.float32)
    fib_theta = np.arccos(2 * ((idxs + 0.5) / npoints) - 1)
    fib_phi = (2 * np.pi * (idxs / golden_ratio)) % (2 * np.pi) - np.pi

    fib_lat = fib_theta
    fib_lon = fib_phi + np.pi

    fib_grid = fgrid.ev(fib_lon, fib_lat)

    # Assign to sphere
    fib_x = centroid[0] + fib_grid * np.sin(fib_theta) * np.cos(fib_phi)
    fib_y = centroid[1] + fib_grid * np.sin(fib_theta) * np.sin(fib_phi)
    fib_z = centroid[2] + fib_grid * np.cos(fib_theta)

    # Add points (x,y,z) to a polydata
    points = vtk.vtkPoints()
    for (x, y, z) in zip(fib_x, fib_y, fib_z):
        points.InsertNextPoint(x, y, z)

    rec = vtk.vtkPolyData()
    rec.SetPoints(points)

    # Calculates the connections between points via Delaunay triangulation
    delaunay = vtk.vtkDelaunay3D()
    delaunay.SetInputData(rec)
    delaunay.Update()

    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(delaunay.GetOutput())
    surface_filter.Update()

    # Smooth the mesh to get a more even distribution of points
    NITER_SMOOTH = 128
    smooth = vtk.vtkSmoothPolyDataFilter()
    smooth.SetInputData(surface_filter.GetOutput())
    smooth.SetNumberOfIterations(NITER_SMOOTH)
    smooth.FeatureEdgeSmoothingOff()
    smooth.BoundarySmoothingOn()
    smooth.Update()

    rec.DeepCopy(smooth.GetOutput())

    # Compute normal vectors
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(rec)
    normals.Update()

    mesh = normals.GetOutput()

    return mesh


def get_even_reconstruction_from_coeffs(
    coeffs: np.array, lrec: int = 0, npoints: int = 512
):

    """Converts a set of spherical harmonic coefficients into
    a 3d mesh using the Fibonacci grid for generating a mesh
    with a more even distribution of points

    Parameters
    ----------
    coeffs : np.array
        Input array of spherical harmonic coefficients. These
        array has dimensions 2xLxM, where the first dimension
        is 0 for cosine-associated coefficients and 1 for
        sine-associated coefficients. Second and thrid dimensions
        represent the expansion parameters (l,m).

    Returns
    -------
    mesh : vtkPolyData
        Mesh that represents the input parametric grid.

    Other parameters
    ----------------
    lrec : int, optional
        Only coefficients l<lrec will be used for creating the
        mesh, default is 0 meaning all coefficients available
        in the matrix coefficients will be used.
    npoints : int optional
        Number of points in the initial spherical mesh

    Notes
    -----
        The mesh resolution is set by the size of the coefficients
        matrix and therefore not affected by lrec.
    """

    coeffs_ = coeffs.copy()

    if (lrec > 0) and (lrec < coeffs_.shape[1]):
        coeffs_[:, lrec:, :] = 0

    grid = pyshtools.expand.MakeGridDH(coeffs_, sampling=2)

    mesh = get_even_reconstruction_from_grid(grid, npoints)

    return mesh, grid


def get_reconstruction_from_grid(grid: np.array, centroid: Tuple = (0, 0, 0)):

    """Converts a parametric 2D grid of type (lon,lat,rad) into
    a 3d mesh. lon in [0,2pi], lat in [0,pi].

    Parameters
    ----------
    grid : np.array
        Input grid where the element grid[i,j] represents the
        radial coordinate at longitude i*2pi/grid.shape[0] and
        latitude j*pi/grid.shape[1].

    Returns
    -------
    mesh : vtkPolyData
        Mesh that represents the input parametric grid.

    Other parameters
    ----------------
    centroid : tuple of floats, optional
        x, y and z coordinates of the centroid where the mesh
        will be translated to, default is (0,0,0).
    """

    res_lat = grid.shape[0]
    res_lon = grid.shape[1]

    # Creates an initial spherical mesh with right dimensions.
    rec = vtk.vtkSphereSource()
    rec.SetPhiResolution(res_lat + 2)
    rec.SetThetaResolution(res_lon)
    rec.Update()
    rec = rec.GetOutput()

    grid_ = grid.T.flatten()

    # Update the points coordinates of the spherical mesh according to the inout grid
    for j, lon in enumerate(np.linspace(0, 2 * np.pi, num=res_lon, endpoint=False)):
        for i, lat in enumerate(
            np.linspace(np.pi / (res_lat + 1), np.pi, num=res_lat, endpoint=False)
        ):
            theta = lat
            phi = lon - np.pi
            k = j * res_lat + i
            x = centroid[0] + grid_[k] * np.sin(theta) * np.cos(phi)
            y = centroid[1] + grid_[k] * np.sin(theta) * np.sin(phi)
            z = centroid[2] + grid_[k] * np.cos(theta)
            rec.GetPoints().SetPoint(k + 2, x, y, z)
    # Update coordinates of north and south pole points
    north = grid_[::res_lat].mean()
    south = grid_[(res_lat - 1) :: res_lat].mean()
    rec.GetPoints().SetPoint(0, centroid[0] + 0, centroid[1] + 0, centroid[2] + north)
    rec.GetPoints().SetPoint(1, centroid[0] + 0, centroid[1] + 0, centroid[2] - south)

    # Compute normal vectors
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(rec)
    # Set splitting off to avoid output mesh from having different number of
    # points compared to input
    normals.SplittingOff()
    normals.Update()

    mesh = normals.GetOutput()

    return mesh


def get_reconstruction_from_coeffs(coeffs: np.array, lrec: int = 0):

    """Converts a set of spherical harmonic coefficients into
    a 3d mesh.

    Parameters
    ----------
    coeffs : np.array
        Input array of spherical harmonic coefficients. These
        array has dimensions 2xLxM, where the first dimension
        is 0 for cosine-associated coefficients and 1 for
        sine-associated coefficients. Second and thrid dimensions
        represent the expansion parameters (l,m).

    Returns
    -------
    mesh : vtkPolyData
        Mesh that represents the input parametric grid.

    Other parameters
    ----------------
    lrec : int, optional
        Degree of the reconstruction. If lrec<l, then only
        coefficients l<lrec will be used for creating the mesh.
        If lrec>l, then the mesh will be oversampled.
        Default is 0 meaning all coefficients
        available in the matrix coefficients will be used.

    Notes
    -----
        The mesh resolution is set by the size of the coefficients
        matrix and therefore not affected by lrec.
    """

    # Degree of the expansion
    lmax = coeffs.shape[1]

    if lrec == 0:
        lrec = lmax

    # Create array (oversampled if lrec>lrec)
    coeffs_ = np.zeros((2, lrec, lrec), dtype=np.float32)

    # Adjust lrec to the expansion degree
    if lrec > lmax:
        lrec = lmax

    # Copy coefficients
    coeffs_[:, :lrec, :lrec] = coeffs[:, :lrec, :lrec]

    # Expand into a grid
    grid = pyshtools.expand.MakeGridDH(coeffs_, sampling=2)

    # Get mesh
    mesh = get_reconstruction_from_grid(grid)

    return mesh, grid


def get_reconstruction_error(grid_input: np.array, grid_rec: np.array):

    """Compute mean square error between two parametric grids. When applied
    to the input parametric grid and its corresponsing reconstructed
    version, it gives an idea of the quality of the parametrization with
    low values indicate good parametrization.

    Parameters
    ----------
    grid_input : np.array
        Parametric grid
    grid_rec : np.array
        Reconstructed parametric grid

    Returns
    -------
    mse : float
        Mean square error
    """

    mse = np.power(grid_input - grid_rec, 2).mean()

    return mse


def save_polydata(mesh: vtk.vtkPolyData, filename: str):

    """Saves a mesh as a vtkPolyData file.

    Parameters
    ----------
    mesh : vtkPolyData
        Input mesh
    filename : str
        File path where the mesh will be saved
    output_type : vtk or ply
        Format of output polydata file
    """

    # Output file format
    output_type = filename.split(".")[-1]

    if output_type not in ["vtk", "ply", "vtp"]:
        raise ValueError(
            f"Output format {output_type} not supported. Please use vtk or ply."
        )

    if output_type == "vtk":
        writer = vtk.vtkPolyDataWriter()
    elif output_type == 'vtp':
        writer = vtk.vtkXMLPolyDataWriter()
    else:
        writer = vtk.vtkPLYWriter()
    writer.SetInputData(mesh)
    writer.SetFileName(filename)
    writer.Write()


    
