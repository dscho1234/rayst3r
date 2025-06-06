import numpy as np
import torch
import copy 
from utils.utils import invalid_to_nans, invalid_to_zeros

def compute_pointmap(depth, cam2w, intrinsics):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    h, w = depth.shape
    
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')

    x_cam = (i - cx) * depth / fx
    y_cam = (j - cy) * depth / fy

    points_cam = np.stack([x_cam, y_cam, depth], axis=-1) 
    points_world = np.dot(cam2w[:3, :3], points_cam.reshape(-1, 3).T).T + cam2w[:3, 3]
    points_world = points_world.reshape(h, w, 3)

    return points_world

def invert_poses(raw_poses):
    poses = copy.deepcopy(raw_poses)
    original_shape = poses.shape
    poses = poses.reshape(-1, 4, 4)
    R = copy.deepcopy(poses[:, :3, :3])
    t = copy.deepcopy(poses[:, :3, 3])
    poses[:, :3, :3] = R.transpose(1, 2)
    poses[:, :3, 3] = torch.bmm(-R.transpose(1, 2), t.unsqueeze(-1)).squeeze(-1)
    poses = poses.reshape(*original_shape)
    return poses

def center_pointmaps_set(dict,w2cs):
    swap_dim = False
    if dict["pointmaps"].shape[1] == 3:
        swap_dim = True
        dict["pointmaps"] = dict["pointmaps"].transpose(1,-1)
    
    original_shape = dict["pointmaps"].shape
    device = dict["pointmaps"].device
    B = original_shape[0]

    # recompute pointmaps in camera frame
    pointmaps = dict["pointmaps"]
    pointmaps_h = torch.cat([pointmaps,torch.ones(pointmaps.shape[:-1]+(1,)).to(device)],dim=-1)
    pointmaps_h = pointmaps_h.reshape(B,-1,4)
    pointmaps_recentered_h = torch.bmm(w2cs,pointmaps_h.transpose(1,2)).transpose(1,2)
    pointmaps_recentered = pointmaps_recentered_h[...,:3]/pointmaps_recentered_h[...,3:4]
    pointmaps_recentered = pointmaps_recentered.reshape(*original_shape)

    # recompute c2ws
    if "c2ws" in dict:
        c2ws_recentered = torch.bmm(w2cs,dict["c2ws"].reshape(-1,4,4))
        c2ws_recentered = c2ws_recentered.reshape(dict["c2ws"].shape)
        dict["c2ws"] = c2ws_recentered

    # assign to dict
    dict["pointmaps"] = pointmaps_recentered
    if swap_dim:
        dict["pointmaps"] = dict["pointmaps"].transpose(1,-1)
    return dict

def center_pointmaps(batch):
    original_poses = batch["new_cams"]["c2ws"] # assuming first camera is the one we want to predict
    w2cs = invert_poses(batch["new_cams"]["c2ws"])
    
    batch["new_cams"] = center_pointmaps_set(batch["new_cams"],w2cs) 
    batch["input_cams"] = center_pointmaps_set(batch["input_cams"],w2cs) 
    batch["original_poses"] = original_poses
    return batch


def uncenter_pointmaps(pred,gt,batch):
    original_poses = batch["original_poses"]

    batch["new_cams"] = center_pointmaps_set(batch["new_cams"],original_poses)
    batch["input_cams"] = center_pointmaps_set(batch["input_cams"],original_poses)

    #gt = center_pointmaps_set(gt,original_poses)
    #pred = center_pointmaps_set(pred,original_poses)
    return pred, gt, batch

def compute_rays(batch):
    h, w = batch["new_cams"]["pointmaps"].shape[-3:-1]
    B = batch["new_cams"]["pointmaps"].shape[0]
    device = batch["new_cams"]["pointmaps"].device
    Ks = batch["new_cams"]["Ks"]
    i_s, j_s = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    i_s, j_s = torch.tensor(i_s).repeat(B,1,1).to(device), torch.tensor(j_s).repeat(B,1,1).to(device)

    f_x = Ks[:,0,0].reshape(-1,1,1)
    f_y = Ks[:,1,1].reshape(-1,1,1)
    c_x = Ks[:,0,2].reshape(-1,1,1)
    c_y = Ks[:,1,2].reshape(-1,1,1)

    # compute rays with z=1
    x_cam = (i_s - c_x)  / f_x
    y_cam = (j_s - c_y)  / f_y
    rays = torch.cat([x_cam.unsqueeze(-1),y_cam.unsqueeze(-1)],dim=-1)
    return rays

def normalize_pointcloud(pts1, pts2=None, norm_mode='avg_dis', valid1=None, valid2=None, valid3=None, ret_factor=False,pts3=None):
    assert pts1.ndim >= 3 and pts1.shape[-1] == 3
    assert pts2 is None or (pts2.ndim >= 3 and pts2.shape[-1] == 3)
    norm_mode, dis_mode = norm_mode.split('_')
    
    if norm_mode == 'avg':
        # gather all points together (joint normalization)
        nan_pts1, nnz1 = invalid_to_zeros(pts1, valid1, ndim=3)
        nan_pts2, nnz2 = invalid_to_zeros(pts2, valid2, ndim=3) if pts2 is not None else (None, 0)
        all_pts = torch.cat((nan_pts1, nan_pts2), dim=1) if pts2 is not None else nan_pts1
        if pts3 is not None:
            nan_pts3, nnz3 = invalid_to_zeros(pts3, valid3, ndim=3)
            all_pts = torch.cat((all_pts, nan_pts3), dim=1)
            nnz1 += nnz3
        # compute distance to origin
        all_dis = all_pts.norm(dim=-1)
        if dis_mode == 'dis': 
            pass # do nothing
        elif dis_mode == 'log1p':
            all_dis = torch.log1p(all_dis)
        elif dis_mode == 'warp-log1p':
            # actually warp input points before normalizing them
            log_dis = torch.log1p(all_dis)
            warp_factor = log_dis / all_dis.clip(min=1e-8)
            H1, W1 = pts1.shape[1:-1]
            pts1 = pts1 * warp_factor[:,:W1*H1].view(-1,H1,W1,1)
            if pts2 is not None:
                H2, W2 = pts2.shape[1:-1]
                pts2 = pts2 * warp_factor[:,W1*H1:].view(-1,H2,W2,1)
            all_dis = log_dis # this is their true distance afterwards
        else:
            raise ValueError(f'bad {dis_mode=}')

        norm_factor = all_dis.sum(dim=1) / (nnz1 + nnz2 + 1e-8)
    else:
        # gather all points together (joint normalization)
        nan_pts1 = invalid_to_nans(pts1, valid1, ndim=3)
        nan_pts2 = invalid_to_nans(pts2, valid2, ndim=3) if pts2 is not None else None
        all_pts = torch.cat((nan_pts1, nan_pts2), dim=1) if pts2 is not None else nan_pts1

        # compute distance to origin
        all_dis = all_pts.norm(dim=-1)

        if norm_mode == 'avg':
            norm_factor = all_dis.nanmean(dim=1)
        elif norm_mode == 'median':
            norm_factor = all_dis.nanmedian(dim=1).values.detach()
        elif norm_mode == 'sqrt':
            norm_factor = all_dis.sqrt().nanmean(dim=1)**2
        else:
            raise ValueError(f'bad {norm_mode=}')

    norm_factor = norm_factor.clip(min=1e-8)
    while norm_factor.ndim < pts1.ndim:
        norm_factor.unsqueeze_(-1)

    res = (pts1 / norm_factor,)
    if pts2 is not None:
        res = res + (pts2 / norm_factor,)
    if pts3 is not None:
        res = res + (pts3 / norm_factor,)
    if ret_factor:
        res = res + (norm_factor,)
    return res

def compute_pointmap_torch(depth, cam2w, intrinsics,device='cuda'):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    h, w = depth.shape
    
    #i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    i, j = torch.meshgrid(torch.arange(w).to(device), torch.arange(h).to(device), indexing='xy')
    x_cam = (i - cx) * depth / fx
    y_cam = (j - cy) * depth / fy

    points_cam = torch.stack([x_cam, y_cam, depth], dim=-1) 
    points_world = (cam2w[:3, :3] @ points_cam.reshape(-1, 3).T).T + cam2w[:3, 3]
    points_world = points_world.reshape(h, w, 3)

    return points_world

def depth2pts(depths, Ks):
    """
    Convert depth map to 3D points
    """
    device = depths.device
    B = depths.shape[0]
    pts = []
    for b in range(B):
        depth_b = depths[b]
        K = Ks[b]
        pts.append(compute_pointmap_torch(depth_b,torch.eye(4).to(device), K,device))
    pts = torch.stack(pts, dim=0)
    return pts