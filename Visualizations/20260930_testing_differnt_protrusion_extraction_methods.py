


import numpy as np
import tifffile
from skimage.filters import gaussian
from skimage import morphology
from skimage.measure import marching_cubes
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from neutrophil_shape.config.loader import load_config
from scipy.ndimage import distance_transform_edt, maximum_filter
from neutrophil_shape.aicssegmentation.core.vessel import filament_3d_wrapper




config = load_config(microscope_type='lls')
xyres = config.im_params.xyres

imdir = Path('E:/Aaron/random_lls/processed_images')

allseg = list(imdir.glob('*_segmented.ome.tiff'))


######### OPEN AN IMAGE
cellnum = 50
im = tifffile.imread(imdir / allseg[cellnum])[0]





######## SMOOTHEN AN IMAGE
radius = 1
objective_sigma = (radius/xyres)/np.sqrt(3)
# sigma = 
smooth = gaussian(im, sigma = objective_sigma)

slic = im.shape[-3]//2
fig, axes = plt.subplots(1, 2)
axes[0].imshow(im[slic])
axes[1].imshow(smooth[slic])



############ SKELETONIZE AN IMAGE
skele = morphology.skeletonize(im)

sz, sy, sx = np.argwhere(skele).T

## im mesh
verts, faces, _, _ = marching_cubes(smooth, level=0.5, )
verts_xyz = verts[:, ::-1]  # (z, y, x) -> (x, y, z) for plotting
tris = verts_xyz[faces]

fig = plt.figure(figsize=(13, 6.5))
ax_mesh = fig.add_subplot(1, 2, 1, projection="3d")
ax_skel = fig.add_subplot(1, 2, 2, projection="3d")
axes = [ax_mesh, ax_skel]

mesh = Poly3DCollection(tris, facecolor="steelblue", edgecolor="none")
ax_mesh.add_collection3d(mesh)
ax_mesh.set_title("Segmentation mesh")
 
ax_skel.scatter(sx, sy, sz, s=12, c=sz, cmap="viridis")
ax_skel.set_title("Skeleton")
 
# Identical limits and aspect so both objects appear at the same scale
lo = verts_xyz.min(axis=0)
hi = verts_xyz.max(axis=0)
for ax in axes:
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
 
# --- Keep both views synchronized while dragging ---
def sync_views(event):
    src = event.inaxes
    if src not in axes or event.button is None:
        return
    for other in axes:
        if other is not src:
            other.view_init(elev=src.elev, azim=src.azim, roll=src.roll)
    fig.canvas.draw_idle()
 
fig.canvas.mpl_connect("motion_notify_event", sync_views)
 
plt.tight_layout()
plt.show()
 





############ COMPARE SKELETON AND MEDIAL AXIS
skele = morphology.skeletonize(im)
sz, sy, sx = np.argwhere(skele).T
med = morphology.medial_axis(im)







############## LOOK FOR LOCA MAXIMA IN A SDT 

sdt = distance_transform_edt(im)
fpt = morphology.ball(10)
lm_skele = maximum_filter(sdt, footprint = fpt)

slic = im.shape[-3]//2
fig, axes = plt.subplots(1, 3)
axes[0].imshow(im[slic])
axes[1].imshow(sdt[slic])
axes[2].imshow(lm_skele[slic])





########### USE HESSIAN FILTER TO FIND RIDGES IN THE SDT
im_mask = im>0
sdt = distance_transform_edt(im_mask)
H = hessian_matrix(sdt, sigma=5, use_gaussian_derivatives=True)
ev = hessian_matrix_eigvals(H)              # sorted: ev[0] >= ev[1] >= ev[2]

thr = 0.4 * -ev[1][im_mask].min()                # tune this
candidate = im & (ev[1] < -thr)             # tube-like ridge voxels
med = morphology.skeletonize(candidate)             # thin the ridge band to 1 voxel

#### first visualize the hessian filter
slic = im.shape[-3]//2
fig, axes = plt.subplots(1, 3)
axes[0].imshow(im[slic])
axes[1].imshow(ev[1][slic])
axes[2].imshow(np.max(med, axis = 0))


######## Then look at candidate skeleton
mz,my,mx = np.argwhere(med).T


## im mesh
verts, faces, _, _ = marching_cubes(smooth, level=0.5, )
verts_xyz = verts[:, ::-1]  # (z, y, x) -> (x, y, z) for plotting
tris = verts_xyz[faces]

fig = plt.figure(figsize=(13, 6.5))
ax_mesh = fig.add_subplot(1, 2, 1, projection="3d")
ax_med = fig.add_subplot(1, 2, 2, projection="3d")
axes = [ax_mesh, ax_med]
 

mesh = Poly3DCollection(tris, facecolor="steelblue", edgecolor="none")
ax_mesh.add_collection3d(mesh)
ax_mesh.set_title("Segmentation mesh")

ax_med.scatter(mx, my, mz, s=12, c=mz, cmap="viridis")
ax_med.set_title("Medial Axis")
 



# Identical limits and aspect so both objects appear at the same scale
lo = verts_xyz.min(axis=0)
hi = verts_xyz.max(axis=0)
for ax in axes:
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")


# --- Keep both views synchronized while dragging ---
def sync_views(event):
    src = event.inaxes
    if src not in axes or event.button is None:
        return
    for other in axes:
        if other is not src:
            other.view_init(elev=src.elev, azim=src.azim, roll=src.roll)
    fig.canvas.draw_idle()
 
fig.canvas.mpl_connect("motion_notify_event", sync_views)
 
plt.tight_layout()
plt.show()







############## USE 3D VESSEL FILTER ON SDT AND TRESHOLD
im_mask = im>0
sdt = distance_transform_edt(im_mask)


f3_param = [[3, 0.8]]
fil = filament_3d_wrapper(sdt, f3_param)

#### first visualize the hessian filter
slic = im.shape[-3]//2
fig, axes = plt.subplots(1, 2)
axes[0].imshow(im[slic])
axes[1].imshow(fil[slic])


######### PLOT RESULTING SKELETON AS 3D
skele = morphology.skeletonize(fil)

sz, sy, sx = np.argwhere(skele).T

## im mesh
verts, faces, _, _ = marching_cubes(smooth, level=0.5, )
verts_xyz = verts[:, ::-1]  # (z, y, x) -> (x, y, z) for plotting
tris = verts_xyz[faces]

fig = plt.figure(figsize=(13, 6.5))
ax_mesh = fig.add_subplot(1, 2, 1, projection="3d")
ax_skel = fig.add_subplot(1, 2, 2, projection="3d")
axes = [ax_mesh, ax_skel]

mesh = Poly3DCollection(tris, facecolor="steelblue", edgecolor="none")
ax_mesh.add_collection3d(mesh)
ax_mesh.set_title("Segmentation mesh")
 
ax_skel.scatter(sx, sy, sz, s=12, c=sz, cmap="viridis")
ax_skel.set_title("Skeleton")
 
# Identical limits and aspect so both objects appear at the same scale
lo = verts_xyz.min(axis=0)
hi = verts_xyz.max(axis=0)
for ax in axes:
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
 
# --- Keep both views synchronized while dragging ---
def sync_views(event):
    src = event.inaxes
    if src not in axes or event.button is None:
        return
    for other in axes:
        if other is not src:
            other.view_init(elev=src.elev, azim=src.azim, roll=src.roll)
    fig.canvas.draw_idle()
 
fig.canvas.mpl_connect("motion_notify_event", sync_views)
 
plt.tight_layout()
plt.show()
 









"""
kimimaro builds a cost field from the distance map, where voxels far from
the boundary are cheap, and extracts shortest paths through it. On a plateau
all voxels have equal cost, so the path goes straight through the middle
instead of wandering. It also returns radius per skeleton vertex. I haven't run this snippet:
"""

import kimimaro
skels = kimimaro.skeletonize(
    mask.astype(np.uint32),
    teasar_params={"scale": 1.5, "const": 0},   # scale sets the cost emphasis, const the spur-pruning distance (see kimimaro docs for units)
    anisotropy=(1, 1, 1),                        # your voxel spacing (z, y, x)
)
skel = skels[1]              


