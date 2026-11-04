"""
Modified from Allen Cell aicsshparam  
"""
from pathlib import Path
import vtk
import pyshtools
import numpy as np
from typing import Tuple
from vtk.util import numpy_support
from skimage import transform as sktrans
from skimage import filters as skfilters
from skimage import morphology as skmorpho
from skimage import measure as skmeasure
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
    


def rotate_mesh(
        mesh,
        rotations: np.array = np.array([0,0,0]), #rotations in [x,y,z]
        center = None, #point to rotate about: None (the origin), 'centroid' (the mean of the vertices) or an [x,y,z] point
        ):
    #rotate mesh. The rotations are about the origin unless a center is given, so that a mesh in its
    #real-space position turns in place instead of swinging around the origin.
    #center is in the units of the mesh. Scaling to physical units is done when the mesh is made, with the
    #spacing argument of get_mesh_from_image.
    #worked from https://kitware.github.io/vtk-examples/site/Python/PolyData/AlignTwoPolyDatas/
    transformation = vtk.vtkTransform()
    #rotate if you need to rotate
    if any(rotations!=0):
        if isinstance(center, str):
            if center != 'centroid':
                raise ValueError(f"center must be None, 'centroid' or a point, not '{center}'")
            center = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float64).mean(axis=0)
        center = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64)
        #vtkTransform is pre-multiplied: the operations act on the points in the reverse order of the calls
        transformation.Translate(*center)
        #input rotations
        transformation.RotateWXYZ(rotations[2], 0, 0, 1)
        transformation.RotateWXYZ(rotations[1], 0, 1, 0)
        transformation.RotateWXYZ(rotations[0], 1, 0, 0)
        transformation.Translate(*(-center))
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
    normals.SplittingOff()  #splitting duplicates the points along sharp edges, which opens the mesh
    normals.Update()

    mesh_updated = normals.GetOutput()

    return mesh_updated


def translate_mesh_to_origin(mesh: vtk.vtkPolyData):
    """Moves a mesh (e.g. one saved in its real-space position) so that the mean of its
    vertices is at (0,0,0). Returns the translated mesh and the vertex mean that was subtracted
    (the mesh's real-space position, in the units of the mesh)."""
    coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    # centering in double precision keeps the vertices of a mesh that sits far from the origin exact
    shift = coords.astype(np.float64).mean(axis=0)
    coords = (coords.astype(np.float64) - shift).astype(coords.dtype)
    return update_mesh_points(mesh, coords[:, 0], coords[:, 1], coords[:, 2]), shift


def mask_to_polydata(
        mask,
        spacing = [1,1,1],
        smooth_iters=0
        ):
    """Marching cubes on a padded mask -> closed, smoothed vtkPolyData
    with (x, y, z) point coordinates.

    Parameters
    ----------
    mask : np.array
        Input array where the mesh will be computed on zyx
    Returns
    -------
    3d mesh in VTK format

    Other parameters
    ----------------
    spacing : list, optional
        zyx spatial resolution of the mask
    smooth_iters : int, optional
        Number of iterations to smooth the surface with a windowed sinc filter.
    """
    padded = np.pad(mask, 1).astype(np.float32)
    verts, faces, _, _ = skmeasure.marching_cubes(
        padded, level=0.5, spacing=spacing, step_size=1)
    verts -= np.array(spacing)      # undo padding offset
    verts = verts[:, ::-1].copy()   # (z, y, x) -> (x, y, z)

    pts = vtk.vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(verts, deep=True))
    cells = vtk.vtkCellArray()
    f = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64),
                   faces.astype(np.int64)]).ravel()
    cells.SetCells(len(faces), numpy_support.numpy_to_vtkIdTypeArray(f, deep=True))

    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)

    if smooth_iters > 0:
        sm = vtk.vtkWindowedSincPolyDataFilter()
        sm.SetInputData(pd)
        sm.SetNumberOfIterations(smooth_iters)
        sm.SetPassBand(0.1)
        sm.BoundarySmoothingOff()
        sm.NonManifoldSmoothingOn()
        sm.NormalizeCoordinatesOn()
        sm.Update()
        pd = sm.GetOutput()

    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputData(pd)
    nrm.SplittingOff()  #splitting duplicates the points along sharp (voxel stair-step) edges, which opens the mesh
    nrm.ConsistencyOn()
    nrm.AutoOrientNormalsOn()
    nrm.Update()
    return nrm.GetOutput()


def largest_connected_component(mask: np.array):
    """Binary (0/1, uint8) image of the largest connected component of the foreground of mask (any dimension).
    An empty mask is returned as all zeros."""
    labeled = skmorpho.label(mask.astype(np.uint8))
    counts = np.bincount(labeled.flatten())
    if len(counts) < 2:
        return np.zeros(mask.shape, dtype=np.uint8)
    return (labeled == 1 + np.argmax(counts[1:])).astype(np.uint8)


def get_mesh_from_image(
    image: np.array,
    lcc: bool = True,
    spacing: list = [1,1,1],
    smooth_iters: int = 50,
    translate_to_origin: bool = True,
    offset: np.array = None,
):

    """Converts a binary 3d image into a closed 3d mesh with marching cubes
    (see mask_to_polydata). The isosurface value is 0.5 and the image is
    padded with background so the mesh always forms a manifold.

    Parameters
    ----------
    image : np.array
        Input array (Z, Y, X) where the mesh will be computed on.
    Returns
    -------
    mesh : vtkPolyData
        3d mesh in VTK format, with (x, y, z) point coordinates.

    Other parameters
    ----------------
    lcc : bool, optional
        Wheather or not to compute the mesh only on the largest
        connected component found in the input,
        default is True.
    spacing : list, optional
        (z, y, x) size of a voxel; the mesh is in these units, default is
        voxels. Note that this is z first, unlike the x, y, z order of
        offset, which is in the units of spacing (so voxels for the
        default, microns for a spacing in microns).
    smooth_iters : int, optional
        Iterations of the windowed sinc smoothing of the mesh surface,
        default is 50, 0 leaves the marching cubes surface as it is.
    translate_to_origin : bool, optional
        Wheather or not translate the mesh to the origin (0,0,0),
        default is True.
    offset : np.array, optional
        Only used if translate_to_origin is False: x, y, z translation added to
        the mesh, in the units of spacing. For the crop offset (the position of
        the image's first voxel within a larger image) this is the voxel index
        times the voxel size, so that the vertices are in the coordinates of
        the larger image; a negative offset subtracts a point. Without offset
        the mesh is left where it is in the image.
    """

    img = image.copy()

    # Extracting the largest connected component
    if lcc:
        img = largest_connected_component(img)

    if img.sum() == 0:
        raise ValueError(
            "No foreground voxels found after pre-processing."
            "Is the object of interest centered?"
        )

    ### convert image to mesh
    mesh = mask_to_polydata(img, spacing, smooth_iters)

    if translate_to_origin:
        # Translate to origin
        mesh, _ = translate_mesh_to_origin(mesh)

    else:
        #leave the mesh where it is in the image, translated by the optional offset
        if offset is not None:
            coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
            coords += np.asarray(offset, dtype=np.float64)
            mesh = update_mesh_points(mesh, coords[:, 0], coords[:, 1], coords[:, 2])

    return mesh






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
    normals.SplittingOff()
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

    grid_ = grid.T.flatten()

    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()

    # Add interior points
    coords = []
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
            coords.append((x, y, z))
            points.InsertNextPoint(x, y, z)

    # Add poles (indices: res_lat*res_lon = north, +1 = south)
    north_r = grid_[::res_lat].mean()
    south_r = grid_[(res_lat - 1)::res_lat].mean()
    north_idx = points.InsertNextPoint(centroid[0], centroid[1], centroid[2] + north_r)
    south_idx = points.InsertNextPoint(centroid[0], centroid[1], centroid[2] - south_r)

    def idx(i, j):
        return j * res_lat + i

    # Interior quads — explicit outward winding order
    for j in range(res_lon):
        j_next = (j + 1) % res_lon
        for i in range(res_lat - 1):
            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, idx(i,     j))
            quad.GetPointIds().SetId(1, idx(i,     j_next))
            quad.GetPointIds().SetId(2, idx(i + 1, j_next))
            quad.GetPointIds().SetId(3, idx(i + 1, j))
            polys.InsertNextCell(quad)

    # North pole triangles
    for j in range(res_lon):
        j_next = (j + 1) % res_lon
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, north_idx)
        tri.GetPointIds().SetId(1, idx(0, j_next))
        tri.GetPointIds().SetId(2, idx(0, j))
        polys.InsertNextCell(tri)

    # South pole triangles
    for j in range(res_lon):
        j_next = (j + 1) % res_lon
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, south_idx)
        tri.GetPointIds().SetId(1, idx(res_lat - 1, j))
        tri.GetPointIds().SetId(2, idx(res_lat - 1, j_next))
        polys.InsertNextCell(tri)

    rec = vtk.vtkPolyData()
    rec.SetPoints(points)
    rec.SetPolys(polys)

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(rec)
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.Update()

    return normals.GetOutput()


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


def save_polydata(mesh: vtk.vtkPolyData, filename: Path):

    """Saves a mesh as a vtkPolyData file.

    Parameters
    ----------
    mesh : vtkPolyData
        Input mesh
    filename : Path
        File path where the mesh will be saved
    output_type : vtk or ply
        Format of output polydata file
    """

    # Output file format
    output_type = filename.suffix.lower()[1:]

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


####### grabbed from https://examples.vtk.org/site/Python/PolyData/BooleanOperationPolyDataFilter/    
def read_polydata(
        file_name:Path,
        ):
    if not file_name.exists():
        raise FileNotFoundError(f"{file_name} not found")
    extension = file_name.suffix.lower()
    if extension == '.ply':
        reader = vtk.vtkPLYReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == '.obj':
        reader = vtk.vtkOBJReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == '.stl':
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == '.vtk':
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == '.g':
        reader = vtk.vtkBYUReader()
        reader.SetGeometryFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    else:
        # Return a None if the extension is unknown.
        poly_data = None
    return poly_data