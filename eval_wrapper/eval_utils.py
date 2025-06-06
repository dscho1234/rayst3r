import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm
import torch
import open3d as o3d

def colorize_points_with_turbo_all_dims(points, method='norm',cmap='turbo'):
    """
    Assigns colors to 3D points using the 'turbo' colormap based on a scalar computed from all 3 dimensions.

    Args:
        points (np.ndarray): (N, 3) array of 3D points.
        method (str): Method for reducing 3D point to scalar. Options: 'norm', 'pca'.

    Returns:
        np.ndarray: (N, 3) RGB colors in [0, 1].
    """
    assert points.shape[1] == 3, "Input must be of shape (N, 3)"

    if method == 'norm':
        scalar = np.linalg.norm(points, axis=1)
    elif method == 'pca':
        # Project onto first principal component
        mean = points.mean(axis=0)
        centered = points - mean
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        scalar = centered @ vh[0]  # Project onto first principal axis
    else:
        raise ValueError(f"Unknown method '{method}'")

    # Normalize scalar to [0, 1]
    scalar_min, scalar_max = scalar.min(), scalar.max()
    normalized = (scalar - scalar_min) / (scalar_max - scalar_min + 1e-8)

    # Apply turbo colormap
    cmap = plt.colormaps.get_cmap(cmap)
    colors = cmap(normalized)[:, :3]  # Drop alpha

    return colors

def npy2ply(points,colors=None,normals=None):
  cloud = o3d.geometry.PointCloud()
  cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))

  # compute the normals
  if colors is not None:
    if colors.max()>1:
      colors = colors/255.0
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
  else:
    colors = colorize_points_with_turbo_all_dims(points)
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
  if normals is not None:
    cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
  return cloud

def transform_pointmap(pointmap_cam,c2w):
    # pointmap: shape H x W x 3
    # cw2: shape 4 x 4
    # we want to transform the pointmap to the world frame
    pointmap_cam_h = torch.cat([pointmap_cam,torch.ones(pointmap_cam.shape[:-1]+(1,)).to(pointmap_cam.device)],dim=-1)
    pointmap_world_h = pointmap_cam_h @ c2w.T
    pointmap_world = pointmap_world_h[...,:3]/pointmap_world_h[...,3:4]
    return pointmap_world

def filter_all_masks(pred_dict, batch, max_outlier_views=1):
    pred_masks = (torch.sigmoid(pred_dict['classifier'][0]).float() < 0.5).bool()  # [V, H, W]
    n_views, H, W = pred_masks.shape
    device = pred_masks.device

    K = batch['input_cams']['Ks'][0][0]  # [3, 3]
    c2ws = batch['new_cams']['c2ws'][0]  # [V, 4, 4]
    w2cs = torch.linalg.inv(c2ws)        # [V, 4, 4]

    pointmaps = pred_dict['pointmaps'][0]  # [V, H, W, 3]
    pointmaps_h = torch.cat([pointmaps, torch.ones_like(pointmaps[..., :1])], dim=-1)  # [V, H, W, 4]

    visibility_count = torch.zeros((n_views, H, W), dtype=torch.int32, device=device)

    for j in range(n_views):
        # Project pointmap j to all other views i ≠ j
        pmap_h = pointmaps_h[j]  # [H, W, 4], world-space points from view j
        pmap_h = pmap_h.view(1, H, W, 4).expand(n_views, -1, -1, -1)  # [V, H, W, 4]

        # Compute T_{i←j} = w2cs[i] @ c2ws[j]
        T = w2cs @ c2ws[j]  # [V, 4, 4]
        T = T.view(n_views, 1, 1, 4, 4)  # [V, 1, 1, 4, 4]

        # Transform to i-th camera frame
        pts_cam = torch.matmul(T, pmap_h.unsqueeze(-1)).squeeze(-1)[..., :3]  # [V, H, W, 3]

        # Project to image
        img_coords = torch.matmul(pts_cam, K.T)  # [V, H, W, 3]
        img_coords = img_coords[..., :2] / img_coords[..., 2:3].clamp(min=1e-6)
        img_coords = img_coords.round().long()  # [V, H, W, 2]

        x = img_coords[..., 0].clamp(0, W - 1)
        y = img_coords[..., 1].clamp(0, H - 1)
        valid = (img_coords[..., 0] >= 0) & (img_coords[..., 0] < W) & \
                (img_coords[..., 1] >= 0) & (img_coords[..., 1] < H)

        # Get depth of the reprojected point from j into i
        reprojected_depth = pts_cam[..., 2]  # [V, H, W]

        # Get depth of each view's original pointmap
        target_depth = pointmaps[:, :, :, 2]  # [V, H, W]

        # Lookup the depth value in view i at the projected location (x, y)
        depth_at_pixel = target_depth[torch.arange(n_views).view(-1, 1, 1), y, x]  # [V, H, W]

        # Check that the point is in front (closest along ray)
        is_closest = reprojected_depth < depth_at_pixel  # [V, H, W]

        # Lookup mask values at projected location
        projected_mask = pred_masks[torch.arange(n_views).view(-1, 1, 1), y, x] & valid  # [V, H, W]

        # Only consider as visible if it’s within mask and closest point
        visible = projected_mask & is_closest  # [V, H, W]

        # Count how many views see each pixel from j
        visibility_count[j] = visible.sum(dim=0)

    visibility_mask = (visibility_count <= max_outlier_views).bool()
    batch['new_cams']['valid_masks'] = visibility_mask & batch['new_cams']['valid_masks']
    return batch