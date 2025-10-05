from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
import open3d as o3d
import time
import copy
import zarr
import argparse


from eval_wrapper.sample_poses import pointmap_to_poses
from utils.fusion import fuse_batch
from models.rayquery import *
from models.losses import *
import utils.misc as misc
import torch.distributed as dist
from utils.collate import collate
from engine import eval_model
from utils.viz import just_load_viz
from utils.geometry import compute_pointmap_torch
from eval_wrapper.eval_utils import npy2ply, filter_all_masks
from huggingface_hub import hf_hub_download
from unidepth.utils import colorize

from utils.utils import parallel_reading, register_codecs

class EvalWrapper(torch.nn.Module):
    def __init__(self,checkpoint_path,distributed=False,device="cuda",dtype=torch.float32,**kwargs):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_string = checkpoint['args'].model
        
        self.model = eval(model_string).to(device)
        if distributed:
            rank, world_size, local_rank = misc.setup_distributed()
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank],find_unused_parameters=True)
        
        self.dtype = dtype
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    def forward(self,x,dino_model=None):
        pred, gt, loss, scale = eval_model(self.model,x,mode='viz',dino_model=dino_model,return_scale=True, convert_depth_to_metric=False)
        return pred, gt, loss, scale

class PostProcessWrapper(torch.nn.Module):
    def __init__(self,pred_mask_threshold = 0.5, mode='novel_views',
    debug=False,conf_dist_mode='isotonic',set_conf=None,percentile=20,
    no_input_mask=False,no_pred_mask=False):
        super().__init__()
        self.pred_mask_threshold = pred_mask_threshold
        self.mode = mode
        self.debug = debug
        self.conf_dist_mode = conf_dist_mode
        self.set_conf = set_conf
        self.percentile = percentile
        self.no_input_mask = no_input_mask
        self.no_pred_mask = no_pred_mask

    def transform_pointmap(self,pointmap_cam,c2w):
        # pointmap: shape H x W x 3
        # cw2: shape 4 x 4
        # we want to transform the pointmap to the world frame
        pointmap_cam_h = torch.cat([pointmap_cam,torch.ones(pointmap_cam.shape[:-1]+(1,)).to(pointmap_cam.device)],dim=-1)
        pointmap_world_h = pointmap_cam_h @ c2w.T
        pointmap_world = pointmap_world_h[...,:3]/pointmap_world_h[...,3:4]
        return pointmap_world

    def reject_conf_points(self,conf_pts):
        if self.set_conf is None:
            raise ValueError("set_conf must be set")
        
        conf_mask = conf_pts > self.set_conf
        return conf_mask
    
    def project_input_mask(self,pred_dict,batch):
        input_mask = batch['input_cams']['original_valid_masks'][0][0] # shape H x W
        input_c2w = batch['input_cams']['c2ws'][0][0]
        input_w2c = torch.linalg.inv(input_c2w)
        input_K = batch['input_cams']['Ks'][0][0]
        H, W = input_mask.shape
        pointmaps_input_cam = torch.stack([self.transform_pointmap(pmap,input_w2c@c2w) for pmap,c2w in zip(pred_dict['pointmaps'][0],batch['new_cams']['c2ws'][0])]) # bp: Assuming batch size is 1!!
        img_coords = pointmaps_input_cam @ input_K.T
        img_coords = (img_coords[...,:2]/img_coords[...,2:3]).int()

        n_views, H, W = img_coords.shape[:3]
        device = input_mask.device
        if self.no_input_mask:
            combined_mask = torch.ones((n_views, H, W), device=device)
        else:
            combined_mask = torch.zeros((n_views, H, W), device=device)

            # Flatten spatial dims
            xs = img_coords[..., 0].view(n_views, -1)  # [V, H*W]
            ys = img_coords[..., 1].view(n_views, -1)  # [V, H*W]

            # Create base pixel coords (i, j)
            i_coords = torch.arange(H, device=device).view(-1, 1).expand(H, W).reshape(-1)  # [H*W]
            j_coords = torch.arange(W, device=device).view(1, -1).expand(H, W).reshape(-1)  # [H*W]
            mask_coords = torch.stack((i_coords, j_coords), dim=-1)  # [H*W, 2], shared across views

            # Mask for valid projections
            valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)  # [V, H*W]

            # Clip out-of-bounds coords for indexing (only valid will be used anyway)
            xs_clipped = torch.clamp(xs, 0, W-1)
            ys_clipped = torch.clamp(ys, 0, H-1)

            # input_mask lookup per view
            flat_input_mask = input_mask[ys_clipped, xs_clipped]  # [V, H*W]
            input_mask_mask = flat_input_mask & valid  # apply valid range mask

            # Apply mask to coords and depths
            depth_points = pointmaps_input_cam[..., -1].view(n_views, -1)  # [V, H*W]
            input_depths = batch['input_cams']['depths'][0][0][ys_clipped, xs_clipped]  # [V, H*W]

            depth_mask = (depth_points > input_depths) & input_mask_mask  # final mask [V, H*W]

            # Get final (i,j) coords to write
            final_i = mask_coords[:, 0].unsqueeze(0).expand(n_views, -1)[depth_mask]  # [N_mask]
            final_j = mask_coords[:, 1].unsqueeze(0).expand(n_views, -1)[depth_mask]  # [N_mask]
            final_view_idx = torch.arange(n_views, device=device).view(-1, 1).expand(-1, H*W)[depth_mask]  # [N_mask]

            # Scatter final mask
            combined_mask[final_view_idx, final_i, final_j] = 1 
        return combined_mask.unsqueeze(0).bool()

    def forward(self,pred_dict,batch):
        if self.mode == 'novel_views':
            project_masks = self.project_input_mask(pred_dict,batch)
            pred_mask_raw = torch.sigmoid(pred_dict['classifier'])
            if self.no_pred_mask:
                pred_masks = torch.ones_like(project_masks).bool()
            else:
                pred_masks = (pred_mask_raw > self.pred_mask_threshold).bool()
            
            conf_masks = self.reject_conf_points(pred_dict['conf_pointmaps'])
            combined_mask = project_masks & pred_masks & conf_masks
            batch['new_cams']['valid_masks'] = combined_mask 

        elif self.mode == 'input_view':
            conf_masks = self.reject_conf_points(pred_dict['conf_pointmaps'])
            if self.no_pred_mask:
                pred_masks = torch.ones_like(conf_masks).bool()
            else:
                pred_mask_raw = torch.sigmoid(pred_dict['classifier'])
                pred_masks = (pred_mask_raw > self.pred_mask_threshold).bool()
            combined_mask = conf_masks & batch['new_cams']['valid_masks'] & pred_masks
            batch['new_cams']['valid_masks'] = combined_mask # this is for visualization

        return pred_dict, batch

class CustomZarrLoader(torch.utils.data.Dataset):
    def __init__(self, zarr_path, episode_idx, frame_idx, depth_scale=0.001, 
                 dtype=torch.float32, n_pred_views=3, pred_input_only=False, pointmap_for_bb=None,
                 min_depth=0.1, use_segmentation_mask=True, mask_combination='all'):
        """
        Custom loader for zarr data with segmentation masks
        
        Args:
            zarr_path: Path to zarr dataset
            episode_idx: Episode index to load
            frame_idx: Frame index to load
            depth_scale: Scale factor for depth values
            dtype: Data type for tensors
            n_pred_views: Number of prediction views
            pred_input_only: Whether to use only input view for prediction
            min_depth: Minimum depth threshold
            use_segmentation_mask: Whether to use segmentation masks
            mask_combination: How to combine multiple object masks ('all', 'first', 'largest')
        """
        self.zarr_path = zarr_path
        self.episode_idx = episode_idx
        self.frame_idx = frame_idx
        self.depth_scale = depth_scale
        self.dtype = dtype
        self.rng = np.random.RandomState(seed=42)
        self.n_pred_views = n_pred_views
        self.min_depth = min_depth  # Keep as metric value, not uint16
        self.pred_input_only = pred_input_only
        self.use_segmentation_mask = use_segmentation_mask
        self.mask_combination = mask_combination
        self.pointmap_for_bb = pointmap_for_bb
        if self.pred_input_only:
            self.n_pred_views = 1
        self.desired_resolution = (480,640)
        self.resize_transform_rgb = transforms.Resize(self.desired_resolution)
        self.resize_transform_depth = transforms.Resize(self.desired_resolution,interpolation=transforms.InterpolationMode.NEAREST)
        
        # Load data from zarr
        self._load_zarr_data()
    
    def _load_zarr_data(self):
        """Load RGB, depth, and segmentation data from zarr"""
        register_codecs()
        
        # Open zarr buffer
        buffer = zarr.open(self.zarr_path, mode="r")
        episode = buffer[f"episode_{self.episode_idx}"]
        
        # Load RGB frame
        rgb_frames = parallel_reading(
            group=episode["camera_0"],
            array_name="rgb",
        )
        self.rgb = rgb_frames[self.frame_idx]
        
        # Load depth frame
        # depth_frames = parallel_reading(
        #     group=episode["camera_0"],
        #     array_name="depth",
        # )
        # self.depth = depth_raw.astype(np.float32) * self.depth_scale
        
        # opt 2: estimate depth
        depth_frames = episode[f"depth_estimate"]
        depth_raw = depth_frames[self.frame_idx]
        self.depth = depth_raw
        
        
        # Load segmentation mask if available
        if self.use_segmentation_mask and "sam_mask_sequence_multi_obj" in episode:
            seg_frames = parallel_reading(
                group=episode,
                array_name="sam_mask_sequence_multi_obj",
            )
            assert rgb_frames.shape[0] == depth_frames.shape[0] == seg_frames.shape[0]
            assert rgb_frames.shape[1:3] == depth_frames.shape[1:3] == seg_frames.shape[2:], f"Shape mismatch: {rgb_frames.shape} != {depth_frames.shape} != {seg_frames.shape}"
            # Check if frame_idx is valid for segmentation mask
            if self.frame_idx < seg_frames.shape[0]:
                self.seg_mask = seg_frames[self.frame_idx]  # Shape: [num_obj, H, W]
                print(f"Loaded segmentation mask with shape: {self.seg_mask.shape}")
                print(f"Segmentation mask time dimension: {seg_frames.shape[0]}, using frame {self.frame_idx}")
            else:
                print(f"Frame index {self.frame_idx} out of range for segmentation mask (max: {seg_frames.shape[0]-1})")
                self.seg_mask = None
        else:
            self.seg_mask = None
            if self.use_segmentation_mask:
                print("No segmentation mask found in episode")
            else:
                print("use_segmentation_mask=False")
    
    def depth_uint16_to_metric(self,depth):
        return depth / torch.iinfo(torch.uint16).max * 10.0

    def depth_metric_to_uint16(self,depth):
        return depth * torch.iinfo(torch.uint16).max / 10.0

    def resize(self,depth,img,mask,K):
        s_x = self.desired_resolution[1] / img.shape[1]
        s_y = self.desired_resolution[0] / img.shape[0]
        depth = self.resize_transform_depth(depth.unsqueeze(0)).squeeze(0)
        img = self.resize_transform_rgb(img.permute(-1,0,1)).permute(1,2,0)
        mask = self.resize_transform_depth(mask.unsqueeze(0)).squeeze(0)
        K[0] *= s_x
        K[1] *= s_y
        return depth, img, mask, K
    
    def look_at(self,cam_pos, center=(0,0,0), up=(0,0,1)):
        z = center - cam_pos
        z /= np.linalg.norm(z, axis=-1, keepdims=True)
        y = -np.float32(up)
        y = y - np.sum(y * z, axis=-1, keepdims=True) * z
        y /= np.linalg.norm(y, axis=-1, keepdims=True)
        x = np.cross(y, z, axis=-1)

        cam2w = np.r_[np.c_[x,y,z,cam_pos],[[0,0,0,1]]]
        return cam2w.astype(np.float32)

    def find_new_views(self,n_views,geometric_median = (0,0,0),r_min=0.4,r_max=0.9):
        rad = self.rng.uniform(r_min,r_max, size=n_views)
        azi = self.rng.uniform(0, 2*np.pi, size=n_views)
        ele = self.rng.uniform(-np.pi, np.pi, size=n_views)
        cam_centers = np.c_[np.cos(azi), np.sin(azi)] 
        cam_centers = rad[:,None] * np.c_[np.cos(ele)[:,None]*cam_centers, np.sin(ele)] + geometric_median
        
        c2ws = [self.look_at(cam_pos=cam_center,center=geometric_median) for cam_center in cam_centers]
        return c2ws

    def process_segmentation_mask(self, seg_mask):
        """
        Process segmentation mask based on combination strategy
        
        Args:
            seg_mask: [num_obj, H, W] segmentation mask (already indexed by frame)
            
        Returns:
            processed_mask: [H, W] boolean mask
        """
        if seg_mask is None:
            return None
        
        print(f"Processing segmentation mask with shape: {seg_mask.shape}")
        print(f"Number of objects: {seg_mask.shape[0]}")
        
        if self.mask_combination == 'all':
            # Combine all object masks with OR operation
            processed_mask = np.any(seg_mask, axis=0)  # [H, W]
            print(f"Combined all {seg_mask.shape[0]} object masks")
        elif self.mask_combination == 'first':
            # Use only the first object mask
            processed_mask = seg_mask[0]  # [H, W]
            print(f"Using first object mask only")
        elif self.mask_combination == 'largest':
            # Use the largest object mask (by area)
            areas = np.sum(seg_mask, axis=(1, 2))  # [num_obj]
            largest_obj_idx = np.argmax(areas)
            processed_mask = seg_mask[largest_obj_idx]  # [H, W]
            print(f"Using largest object mask (index {largest_obj_idx}, area: {areas[largest_obj_idx]})")
        else:
            raise ValueError(f"Unknown mask_combination: {self.mask_combination}")
        
        print(f"Final processed mask shape: {processed_mask.shape}, valid pixels: {np.sum(processed_mask)}")
        return processed_mask.astype(bool)

    def __len__(self):
        return 1  # Single scene
    
    def __getitem__(self,idx):
        data = dict(new_cams={},input_cams={})

        # Set up input camera data
        data['input_cams']['c2ws_original'] = [torch.eye(4).to(self.dtype)]
        data['input_cams']['c2ws'] = [torch.eye(4).to(self.dtype)]
        
        # Load camera intrinsics (you may need to adjust this based on your data)
        # For now, using a default camera matrix - you should load this from your zarr data
        K = np.array([
            [604.682922, 0.0, 328.062561],
            [0.0, 604.898438, 244.393188],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        data['input_cams']['Ks'] = [torch.from_numpy(K).to(self.dtype)]
        
        # Load depth and RGB
        data['input_cams']['depths'] = [torch.from_numpy(self.depth).to(self.dtype)]
        data['input_cams']['imgs'] = [torch.from_numpy(self.rgb).to(self.dtype)]
        
        # Debug: Print depth statistics
        depth_tensor = data['input_cams']['depths'][0]
        print(f"Depth statistics - Min: {depth_tensor.min():.6f}, Max: {depth_tensor.max():.6f}, Mean: {depth_tensor.mean():.6f}, Median: {depth_tensor.median():.6f}")
        
        # Process segmentation mask
        if self.use_segmentation_mask and self.seg_mask is not None:
            processed_mask = self.process_segmentation_mask(self.seg_mask)
            data['input_cams']['valid_masks'] = [torch.from_numpy(processed_mask).bool()]
            print(f"Using segmentation mask with {np.sum(processed_mask)} valid pixels")
        else:
            # Create a default mask (all pixels with valid depth)
            valid_mask = (self.depth > 0) & (self.depth < 10.0)  # 10m max depth
            data['input_cams']['valid_masks'] = [torch.from_numpy(valid_mask).bool()]
            print(f"Using default mask with {np.sum(valid_mask)} valid pixels")

        # Resize if needed
        if data['input_cams']['depths'][0].shape != self.desired_resolution:
            data['input_cams']['depths'][0], data['input_cams']['imgs'][0], data['input_cams']['valid_masks'][0], data['input_cams']['Ks'][0] = \
            self.resize(data['input_cams']['depths'][0], data['input_cams']['imgs'][0], data['input_cams']['valid_masks'][0], data['input_cams']['Ks'][0])
        
        data['input_cams']['original_valid_masks'] = [data['input_cams']['valid_masks'][0].clone()]
        # Apply depth threshold (convert to metric if needed)
        depth_threshold = self.min_depth
        data['input_cams']['valid_masks'][0] = data['input_cams']['valid_masks'][0] & \
            (data['input_cams']['depths'][0] > depth_threshold)

        # Generate new camera poses
        if self.pred_input_only:
            c2ws = [data['input_cams']['c2ws'][0].cpu().numpy()]
        else:
            input_mask = data['input_cams']['valid_masks'][0]
            if self.pointmap_for_bb is not None:
                pointmap_input = self.pointmap_for_bb
            else:
                pointmap_input = compute_pointmap_torch(data['input_cams']['depths'][0],data['input_cams']['c2ws'][0],data['input_cams']['Ks'][0],device='cpu')[input_mask]
            
            # Check if we have valid points
            assert len(pointmap_input) != 0
            c2ws = pointmap_to_poses(pointmap_input, self.n_pred_views, inner_radius=1.1, outer_radius=2.5, device='cpu')
            self.n_pred_views = len(c2ws)
        
        # Set up new camera data
        data['new_cams'] = {}
        data['new_cams']['c2ws'] = [torch.from_numpy(c2w).to(self.dtype) for c2w in c2ws]
        data['new_cams']['depths'] = [torch.zeros_like(data['input_cams']['depths'][0]) for _ in range(self.n_pred_views)]
        data['new_cams']['Ks'] = [data['input_cams']['Ks'][0] for _ in range(self.n_pred_views)]
        if self.pred_input_only:
            data['new_cams']['valid_masks'] = data['input_cams']['original_valid_masks']
        else:
            data['new_cams']['valid_masks'] = [torch.ones_like(data['input_cams']['valid_masks'][0]) for _ in range(self.n_pred_views)]
        
        return data

def dict_to_float(d):
    return {k: v.float() for k, v in d.items()}

def merge_dicts(d1,d2):
    # stack the tensors along dimension 1 
    for k,v in d1.items():
        d1[k] = torch.cat([d1[k],d2[k]],dim=1)
    return d1

def compute_all_points(pred_dict,batch):
    n_views = pred_dict['depths'].shape[1]
    all_points = None 
    for i in range(n_views):
        mask = batch['new_cams']['valid_masks'][0,i]
        pointmap = compute_pointmap_torch(pred_dict['depths'][0,i],batch['new_cams']['c2ws'][0,i],batch['new_cams']['Ks'][0,i])
        masked_points = pointmap[mask]
        if all_points is None:
            all_points = masked_points
        else:
            all_points = torch.cat([all_points,masked_points],dim=0)
    return all_points

def preprocess_point_cloud(pcd, remove_outliers=True, downsample=True, estimate_normals=True):
    """
    Preprocess point cloud to improve mesh quality
    
    Args:
        pcd: Open3D PointCloud object
        remove_outliers: Whether to remove statistical outliers
        downsample: Whether to downsample the point cloud
        estimate_normals: Whether to estimate normals
    
    Returns:
        processed_pcd: Preprocessed Open3D PointCloud object
    """
    print(f"Original point cloud: {len(pcd.points)} points")
    
    # Check point cloud bounds
    points = np.asarray(pcd.points)
    if len(points) > 0:
        print(f"Point cloud bounds: min={points.min(axis=0)}, max={points.max(axis=0)}")
        print(f"Point cloud center: {points.mean(axis=0)}")
        print(f"Point cloud scale: {np.linalg.norm(points.max(axis=0) - points.min(axis=0))}")
    
    # Remove outliers using statistical outlier removal (with more conservative parameters)
    if remove_outliers and len(pcd.points) > 100:
        # Use more conservative outlier removal parameters
        pcd, outlier_indices = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=3.0)
        print(f"After outlier removal: {len(pcd.points)} points (removed {len(outlier_indices)} outliers)")
    
    # Downsample using voxel grid (with adaptive voxel size)
    if downsample and len(pcd.points) > 1000:
        # Calculate adaptive voxel size based on point cloud scale
        if len(points) > 0:
            scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
            voxel_size = max(scale / 100, 0.001)  # Adaptive voxel size, minimum 1mm
        else:
            voxel_size = 0.01  # Default 1cm voxel size
        
        print(f"Using voxel size: {voxel_size}")
        pcd = pcd.voxel_down_sample(voxel_size)
        print(f"After downsampling: {len(pcd.points)} points")
    
    # Estimate normals with improved parameters
    if estimate_normals and len(pcd.points) > 0:
        # Calculate adaptive radius based on point cloud scale
        if len(points) > 0:
            scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
            radius = max(scale / 50, 0.0001)  # Adaptive radius, minimum 0.1mm for small scales
        else:
            radius = 0.1  # Default radius
        
        print(f"Using normal estimation radius: {radius}")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
        
        # Try to orient normals consistently, but handle small scale issues
        try:
            pcd.orient_normals_consistent_tangent_plane(100)
            print("Normals estimated and oriented")
        except RuntimeError as e:
            print(f"Normal orientation failed (likely due to small scale): {e}")
            print("Using estimated normals without orientation")
            # For very small scales, we'll use the normals as-is without orientation
    
    return pcd

def create_mesh_from_points(points, colors=None, save_path=None, method='poisson', 
                          preprocess=True, mesh_postprocess=True):
    """
    Create a mesh from 3D points using various methods
    
    Args:
        points: [N, 3] array of 3D points
        colors: [N, 3] array of RGB colors (optional)
        save_path: Path to save the mesh (optional)
        method: Mesh reconstruction method ('poisson', 'alpha_shape', 'ball_pivoting', 'convex_hull')
        preprocess: Whether to preprocess the point cloud
        mesh_postprocess: Whether to postprocess the mesh
    
    Returns:
        mesh: Open3D TriangleMesh object
    """
    if len(points) == 0:
        print("Warning: No points to create mesh from")
        return None
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Check if point cloud scale is too small for preprocessing
    points_array = np.asarray(pcd.points)
    if len(points_array) > 0:
        scale = np.linalg.norm(points_array.max(axis=0) - points_array.min(axis=0))
        print(f"Point cloud scale: {scale}")
        
        # If scale is very small (< 0.001), skip preprocessing to avoid issues
        if scale < 0.001:
            print("Point cloud scale is very small, skipping preprocessing to avoid numerical issues")
            preprocess = False
    
    # Preprocess point cloud
    if preprocess:
        pcd = preprocess_point_cloud(pcd)
    
    mesh = None
    
    try:
        if method == 'poisson':
            mesh = create_poisson_mesh(pcd)
        elif method == 'alpha_shape':
            mesh = create_alpha_shape_mesh(pcd)
        elif method == 'ball_pivoting':
            mesh = create_ball_pivoting_mesh(pcd)
        elif method == 'convex_hull':
            mesh = create_convex_hull_mesh(pcd)
        elif method == 'auto':
            # Try multiple methods in order of preference
            mesh_methods = ['poisson', 'alpha_shape', 'ball_pivoting']
            mesh = None
            
            for mesh_method in mesh_methods:
                print(f"Trying {mesh_method} method...")
                if mesh_method == 'poisson':
                    mesh = create_poisson_mesh(pcd)
                elif mesh_method == 'alpha_shape':
                    mesh = create_alpha_shape_mesh(pcd)
                elif mesh_method == 'ball_pivoting':
                    mesh = create_ball_pivoting_mesh(pcd)
                
                if mesh is not None and len(mesh.vertices) > 0:
                    print(f"Successfully created mesh using {mesh_method}")
                    break
                else:
                    print(f"Failed to create mesh using {mesh_method}")
            
            # If all methods failed, try convex hull as fallback
            if mesh is None:
                print("All methods failed, trying convex hull as fallback...")
                mesh = create_convex_hull_mesh(pcd)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if mesh is not None and mesh_postprocess:
            mesh = postprocess_mesh(mesh)
        
        if mesh is not None and save_path:
            o3d.io.write_triangle_mesh(save_path, mesh)
            print(f"Mesh saved to: {save_path}")
        
        return mesh
        
    except Exception as e:
        print(f"Mesh creation with {method} failed: {e}")
        # Fallback to convex hull
        try:
            print("Falling back to convex hull...")
            mesh = create_convex_hull_mesh(pcd)
            if mesh is not None and save_path:
                o3d.io.write_triangle_mesh(save_path, mesh)
                print(f"Convex hull mesh saved to: {save_path}")
            return mesh
        except Exception as e2:
            print(f"Convex hull also failed: {e2}")
            return None

def create_poisson_mesh(pcd, depth=8, width=0, scale=1.1, linear_fit=False):
    """Create mesh using Poisson reconstruction with improved parameters"""
    try:
        # Check if point cloud has normals
        if not pcd.has_normals():
            print("Point cloud has no normals, estimating them for Poisson reconstruction...")
            # Estimate normals with very small radius for small scale point clouds
            points = np.asarray(pcd.points)
            if len(points) > 0:
                point_scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
                radius = max(point_scale / 100, 0.0001)  # Very small radius for small scales
            else:
                radius = 0.0001
            
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=20)
            )
            
            # Try to orient normals, but don't fail if it doesn't work
            try:
                pcd.orient_normals_consistent_tangent_plane(50)
            except RuntimeError:
                print("Normal orientation failed, using estimated normals as-is")
        
        # Try different depth values for better quality
        for d in [depth, depth-1, depth+1, 7, 9]:
            try:
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=d, width=width, scale=scale, linear_fit=linear_fit
                )
                print(f"Poisson mesh created with depth={d}: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
                return mesh
            except Exception as e:
                print(f"Poisson with depth={d} failed: {e}")
                continue
        return None
    except Exception as e:
        print(f"Poisson reconstruction failed: {e}")
        return None

def create_alpha_shape_mesh(pcd, alpha=1):
    """Create mesh using alpha shapes"""
    try:
        # Try different alpha values
        for a in [alpha, alpha*0.5, alpha*2.0, 0.05, 0.2]:
            try:
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, a)
                if len(mesh.vertices) > 0:
                    print(f"Alpha shape mesh created with alpha={a}: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
                    return mesh
            except Exception as e:
                print(f"Alpha shape with alpha={a} failed: {e}")
                continue
        return None
    except Exception as e:
        print(f"Alpha shape reconstruction failed: {e}")
        return None

def create_ball_pivoting_mesh(pcd, radii=None):
    """Create mesh using ball pivoting algorithm"""
    try:
        if radii is None:
            # Estimate radii based on point cloud density
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radii = [avg_dist, avg_dist * 2, avg_dist * 4]
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        print(f"Ball pivoting mesh created: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        return mesh
    except Exception as e:
        print(f"Ball pivoting reconstruction failed: {e}")
        return None

def create_convex_hull_mesh(pcd):
    """Create mesh using convex hull"""
    try:
        hull, _ = pcd.compute_convex_hull()
        print(f"Convex hull mesh created: {len(hull.vertices)} vertices, {len(hull.triangles)} triangles")
        return hull
    except Exception as e:
        print(f"Convex hull reconstruction failed: {e}")
        return None

def postprocess_mesh(mesh):
    """Postprocess mesh to improve quality"""
    try:
        # Remove degenerate triangles
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        # Fill holes (if available in this Open3D version)
        try:
            mesh = mesh.fill_holes()
        except AttributeError:
            print("fill_holes method not available in this Open3D version, skipping...")
        
        # Smooth mesh
        try:
            mesh = mesh.filter_smooth_simple(number_of_iterations=1)
        except AttributeError:
            print("filter_smooth_simple method not available in this Open3D version, skipping...")
        
        print(f"Mesh postprocessed: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        return mesh
    except Exception as e:
        print(f"Mesh postprocessing failed: {e}")
        return mesh

def save_fused_meshes(fused_meshes, output_dir, prefix="fused_mesh"):
    """
    Save fused meshes from TSDF fusion to PLY files
    
    Args:
        fused_meshes: List of mesh dictionaries from fuse_batch
        output_dir: Output directory to save PLY files
        prefix: Prefix for the saved files
    """
    if not fused_meshes:
        print("No fused meshes to save")
        return
    
    print(f"Saving {len(fused_meshes)} fused mesh(es)...")
    
    for i, mesh_dict in enumerate(fused_meshes):
        try:
            # Extract mesh components
            verts = mesh_dict['verts']  # [N, 3] vertices
            faces = mesh_dict['faces']  # [M, 3] face indices
            norms = mesh_dict['norms']  # [N, 3] normals
            colors = mesh_dict['colors']  # [N, 3] colors (0-255)
            
            print(f"Mesh {i}: {len(verts)} vertices, {len(faces)} faces")
            
            # Create Open3D mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            
            # Add normals if available
            if norms is not None and len(norms) > 0:
                mesh.vertex_normals = o3d.utility.Vector3dVector(norms)
            
            # Add colors if available
            if colors is not None and len(colors) > 0:
                # Normalize colors to [0, 1] range if they're in [0, 255]
                if colors.max() > 1.0:
                    colors = colors / 255.0
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            
            # Save mesh
            if len(fused_meshes) == 1:
                save_path = os.path.join(output_dir, f"{prefix}.ply")
            else:
                save_path = os.path.join(output_dir, f"{prefix}_{i}.ply")
            
            o3d.io.write_triangle_mesh(save_path, mesh)
            print(f"Saved fused mesh {i} to: {save_path}")
            
        except Exception as e:
            print(f"Failed to save fused mesh {i}: {e}")
    
    print("Fused mesh saving completed!")

def save_input_images(rgb, depth, mask, output_dir, frame_idx):
    """
    Save input RGB, depth, and mask images as PNG files
    
    Args:
        rgb: RGB image tensor [H, W, 3]
        depth: Depth image tensor [H, W]
        mask: Mask tensor [H, W]
        output_dir: Output directory
        frame_idx: Frame index for filename
    """
    import os
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays and remove batch dimensions
    if hasattr(rgb, 'cpu'):
        rgb_np = rgb.cpu().numpy()
    else:
        rgb_np = rgb
    
    if hasattr(depth, 'cpu'):
        depth_np = depth.cpu().numpy()
    else:
        depth_np = depth
        
    if hasattr(mask, 'cpu'):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    # Remove batch dimensions if present
    if rgb_np.ndim == 4:  # [B, H, W, C]
        rgb_np = rgb_np[0]
    if depth_np.ndim == 3:  # [B, H, W]
        depth_np = depth_np[0]
    if mask_np.ndim == 3:  # [B, H, W]
        mask_np = mask_np[0]
    
    print(f"Data shapes - RGB: {rgb_np.shape}, Depth: {depth_np.shape}, Mask: {mask_np.shape}")
    
    # Save RGB image
    rgb_path = os.path.join(output_dir, f"input_rgb_frame_{frame_idx}.png")
    if rgb_np.max() <= 1.0:
        rgb_save = (rgb_np * 255).astype(np.uint8)
    else:
        rgb_save = rgb_np.astype(np.uint8)
    Image.fromarray(rgb_save).save(rgb_path)
    print(f"RGB image saved to: {rgb_path}")
    
    # Save depth image using colorize
    depth_path = os.path.join(output_dir, f"input_depth_frame_{frame_idx}.png")
    depth_colored = colorize(depth_np, cmap="magma_r")
    Image.fromarray(depth_colored).save(depth_path)
    print(f"Depth image saved to: {depth_path}")
    
    # Save mask image
    mask_path = os.path.join(output_dir, f"input_mask_frame_{frame_idx}.png")
    mask_save = (mask_np * 255).astype(np.uint8)
    Image.fromarray(mask_save, mode='L').save(mask_path)
    print(f"Mask image saved to: {mask_path}")
    
    # Save combined visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB
    axes[0].imshow(rgb_save)
    axes[0].set_title(f'RGB (Frame {frame_idx})')
    axes[0].axis('off')
    
    # Depth (using colorized version)
    axes[1].imshow(depth_colored)
    axes[1].set_title(f'Depth (Frame {frame_idx})')
    axes[1].axis('off')
    
    # Mask
    axes[2].imshow(mask_np, cmap='gray')
    axes[2].set_title(f'Mask (Frame {frame_idx})')
    axes[2].axis('off')
    
    plt.tight_layout()
    combined_path = os.path.join(output_dir, f"input_combined_frame_{frame_idx}.png")
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined visualization saved to: {combined_path}")

def create_3d_visualization(points, colors=None, save_path="output_3d.html"):
    """
    Create an interactive 3D visualization using plotly
    
    Args:
        points: [N, 3] array of 3D points
        colors: [N, 3] array of RGB colors (optional)
        save_path: Path to save the HTML visualization
    """
    try:
        import plotly.graph_objects as go
        import plotly.offline as pyo
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        if colors is not None:
            # Convert colors to hex strings
            colors_hex = []
            for color in colors:
                r, g, b = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                colors_hex.append(f'rgb({r},{g},{b})')
            
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=colors_hex,
                    opacity=0.8
                ),
                name='3D Points'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='lightblue',
                    opacity=0.8
                ),
                name='3D Points'
            ))
        
        fig.update_layout(
            title='3D Point Cloud Visualization',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            width=1000,
            height=800
        )
        
        pyo.plot(fig, filename=save_path, auto_open=False)
        print(f"3D visualization saved to: {save_path}")
        
    except ImportError:
        print("Plotly not available, skipping 3D visualization")

def eval_custom_scene(model, zarr_path, episode_idx, frame_idx, output_dir="custom_eval_output",
                     visualize=False, rr_addr=None, run_octmae=False, set_conf=5,
                     no_input_mask=False, no_pred_mask=False, no_filter_input_view=False,
                     n_pred_views=5, do_filter_all_masks=False, dino_model=None, tsdf=False,
                     use_segmentation_mask=True, mask_combination='all', depth_scale=0.001,
                     mesh_method='auto', no_preprocess=False, no_mesh_postprocess=False,
                     tsdf_voxel_size=0.005):
    """
    Evaluate a custom scene from zarr data
    
    Args:
        model: RayST3R model
        zarr_path: Path to zarr dataset
        episode_idx: Episode index
        frame_idx: Frame index
        output_dir: Output directory for results
        visualize: Whether to create visualizations
        rr_addr: Rerun address for visualization
        run_octmae: Whether to use OctMAE
        set_conf: Confidence threshold
        no_input_mask: Whether to ignore input masks
        no_pred_mask: Whether to ignore prediction masks
        no_filter_input_view: Whether to filter input view
        n_pred_views: Number of prediction views
        do_filter_all_masks: Whether to filter all masks
        dino_model: DINO model for features
        tsdf: Whether to create TSDF mesh
        use_segmentation_mask: Whether to use segmentation masks
        mask_combination: How to combine segmentation masks
        depth_scale: Scale factor for depth values
        mesh_method: Mesh reconstruction method ('auto', 'poisson', 'alpha_shape', 'ball_pivoting', 'convex_hull')
        no_preprocess: Whether to skip point cloud preprocessing
        no_mesh_postprocess: Whether to skip mesh postprocessing
        tsdf_voxel_size: Voxel size for TSDF fusion
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize timing dictionary
    timing_info = {}
    total_start_time = time.time()
    
    # Store frame index for saving images
    frame_idx_for_save = frame_idx
    
    if dino_model is None:
        # Loading DINOv2 model
        dino_load_start = time.time()
        dino_model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitl14_reg")
        dino_model.eval()
        dino_model.to("cuda")
        timing_info['dino_model_loading'] = time.time() - dino_load_start

    # Data loading timing
    data_load_start = time.time()
    dataloader_input_view = CustomZarrLoader(
        zarr_path, episode_idx, frame_idx, 
        depth_scale=depth_scale,
        n_pred_views=1, pred_input_only=True,
        use_segmentation_mask=use_segmentation_mask,
        mask_combination=mask_combination
    )
    input_view_loader = DataLoader(dataloader_input_view, batch_size=1, shuffle=False, collate_fn=collate)
    input_view_batch = next(iter(input_view_loader))
    timing_info['input_view_data_loading'] = time.time() - data_load_start
    
    # Save input images
    print("Saving input images...")
    # Get the data directly from the dataloader
    rgb_data = torch.from_numpy(dataloader_input_view.rgb).to(dataloader_input_view.dtype)
    depth_data = torch.from_numpy(dataloader_input_view.depth).to(dataloader_input_view.dtype)
    
    # Process segmentation mask for saving
    if dataloader_input_view.use_segmentation_mask and dataloader_input_view.seg_mask is not None:
        processed_mask = dataloader_input_view.process_segmentation_mask(dataloader_input_view.seg_mask)
        mask_data = torch.from_numpy(processed_mask).bool()
    else:
        valid_mask = (dataloader_input_view.depth > 0) & (dataloader_input_view.depth < 10.0)
        mask_data = torch.from_numpy(valid_mask).bool()
    
    save_input_images(
        rgb_data, 
        depth_data, 
        mask_data, 
        output_dir, 
        frame_idx_for_save
    )

    postprocessor_input_view = PostProcessWrapper(mode='input_view',set_conf=set_conf,
                                                  no_input_mask=no_input_mask,no_pred_mask=no_pred_mask)
    postprocessor_pred_views = PostProcessWrapper(mode='novel_views',debug=False,set_conf=set_conf,
                                                  no_input_mask=no_input_mask,no_pred_mask=no_pred_mask)
    
    with torch.no_grad():
        # Input view inference timing
        input_inference_start = time.time()
        pred_input_view, gt_input_view, _, scale_factor = model(input_view_batch,dino_model)
        print(f"Input view batch scale factor: {scale_factor}")
        timing_info['input_view_inference'] = time.time() - input_inference_start
        
        # Input view postprocessing timing
        input_postprocess_start = time.time()
        if no_filter_input_view:
            pred_input_view['pointmaps'] = input_view_batch['input_cams']['pointmaps']
            pred_input_view['depths'] = input_view_batch['input_cams']['depths']
        else: 
            pred_input_view, input_view_batch = postprocessor_input_view(pred_input_view,input_view_batch)
        timing_info['input_view_postprocessing'] = time.time() - input_postprocess_start

        input_points = pred_input_view['pointmaps'][0][0][input_view_batch['new_cams']['valid_masks'][0][0]] * (1.0/scale_factor)
        if input_points.shape[0] == 0:
            input_points = None
        
        # Prediction views data loading timing
        pred_data_load_start = time.time()
        dataloader_pred_views = CustomZarrLoader(
            zarr_path, episode_idx, frame_idx,
            depth_scale=depth_scale,
            n_pred_views=n_pred_views, pred_input_only=False,
            pointmap_for_bb=input_points,
            use_segmentation_mask=use_segmentation_mask,
            mask_combination=mask_combination
        )
        pred_views_loader = DataLoader(dataloader_pred_views, batch_size=1, shuffle=False, collate_fn=collate)
        pred_views_batch = next(iter(pred_views_loader))
        timing_info['pred_views_data_loading'] = time.time() - pred_data_load_start

        # Prediction views inference timing
        pred_inference_start = time.time()
        pred_new_views, gt_new_views, _, scale_factor = model(pred_views_batch,dino_model)
        print(f"Prediction views batch scale factor: {scale_factor}")
        timing_info['pred_views_inference'] = time.time() - pred_inference_start
        
        # Prediction views postprocessing timing
        pred_postprocess_start = time.time()
        pred_new_views, pred_views_batch = postprocessor_pred_views(pred_new_views,pred_views_batch)
        timing_info['pred_views_postprocessing'] = time.time() - pred_postprocess_start
    
    # Final processing timing
    final_process_start = time.time()
    pred = merge_dicts(dict_to_float(pred_input_view),dict_to_float(pred_new_views))
    gt = merge_dicts(dict_to_float(gt_input_view),dict_to_float(gt_new_views))

    batch = copy.deepcopy(input_view_batch)
    batch['new_cams'] = merge_dicts(input_view_batch['new_cams'],pred_views_batch['new_cams'])
    gt['pointmaps'] = None # make sure it's not used in viz
    
    if do_filter_all_masks:
        batch = filter_all_masks(pred,input_view_batch,max_outlier_views=1)

    # scale factor is the scale we applied to the input view for inference
    all_points = compute_all_points(pred,batch)
    all_points = all_points*(1.0/scale_factor)
    
    # transform all_points to the original coordinate system
    all_points_h = torch.cat([all_points,torch.ones(all_points.shape[:-1]+(1,)).to(all_points.device)],dim=-1)
    all_points_original = all_points_h @ batch['input_cams']['c2ws_original'][0][0].T
    all_points = all_points_original[...,:3]
    
    # Create mesh and visualizations
    if tsdf:
        print(f"Creating TSDF fusion with voxel size: {tsdf_voxel_size}")
        fused_meshes = fuse_batch(pred,gt,batch,voxel_size=tsdf_voxel_size)
        # Save fused meshes to PLY files
        save_fused_meshes(fused_meshes, output_dir, prefix="tsdf_fused_mesh")
    else:
        fused_meshes = None
    
    # Save point cloud
    points_save_path = os.path.join(output_dir, "inference_points.ply")
    o3d_pc = npy2ply(all_points.cpu().numpy(), colors=None, normals=None)
    o3d.io.write_point_cloud(points_save_path, o3d_pc)
    print(f"Point cloud saved to: {points_save_path}")
    
    # Create mesh from points
    mesh_save_path = os.path.join(output_dir, "inference_mesh.ply")
    
    if mesh_method == 'auto':
        # Try multiple methods in order of preference
        mesh_methods = ['poisson', 'alpha_shape', 'ball_pivoting']
        mesh = None
        
        for method in mesh_methods:
            print(f"\nTrying mesh creation with method: {method}")
            method_save_path = os.path.join(output_dir, f"inference_mesh_{method}.ply")
            mesh = create_mesh_from_points(
                all_points.cpu().numpy(), 
                save_path=method_save_path,
                method=method,
                preprocess=not no_preprocess,
                mesh_postprocess=not no_mesh_postprocess
            )
            
            if mesh is not None and len(mesh.vertices) > 0:
                print(f"Successfully created mesh using {method}")
                # Also save as the main mesh file
                o3d.io.write_triangle_mesh(mesh_save_path, mesh)
                break
            else:
                print(f"Failed to create mesh using {method}")
        
        if mesh is None:
            print("All mesh creation methods failed, trying convex hull as final fallback...")
            mesh = create_mesh_from_points(
                all_points.cpu().numpy(), 
                save_path=os.path.join(output_dir, "inference_mesh_convex_hull.ply"),
                method='convex_hull',
                preprocess=not no_preprocess,
                mesh_postprocess=False
            )
            if mesh is not None:
                o3d.io.write_triangle_mesh(mesh_save_path, mesh)
    else:
        # Use specified method
        print(f"\nCreating mesh using method: {mesh_method}")
        mesh = create_mesh_from_points(
            all_points.cpu().numpy(), 
            save_path=mesh_save_path,
            method=mesh_method,
            preprocess=not no_preprocess,
            mesh_postprocess=not no_mesh_postprocess
        )
    
    # Create 3D visualization
    viz_save_path = os.path.join(output_dir, "3d_visualization.html")
    create_3d_visualization(all_points.cpu().numpy(), save_path=viz_save_path)
    
    if visualize:
        just_load_viz(pred, gt, batch, addr=rr_addr, fused_meshes=fused_meshes)
    
    timing_info['final_processing'] = time.time() - final_process_start
    timing_info['total_time'] = time.time() - total_start_time
    
    # Print timing summary
    print("\n" + "="*50)
    print("INFERENCE TIMING SUMMARY")
    print("="*50)
    for key, value in timing_info.items():
        print(f"{key.replace('_', ' ').title()}: {value:.4f} seconds")
    print("="*50)
    print(f"Total Inference Time: {timing_info['total_time']:.4f} seconds")
    print("="*50 + "\n")
    
    return all_points, mesh

def main():
    parser = argparse.ArgumentParser(description="Custom evaluation with zarr data and segmentation masks")
    parser.add_argument("zarr_path", type=str, help="Path to zarr dataset")
    parser.add_argument("--episode_idx", type=int, default=0, help="Episode index")
    parser.add_argument("--frame_idx", type=int, default=30, help="Frame index")
    parser.add_argument("--output_dir", type=str, default="custom_eval_output", help="Output directory")
    parser.add_argument("--rr_addr", type=str, default="0.0.0.0:"+os.getenv("RERUN_RECORDING","9876"))
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--run_octmae", action="store_true", default=False)
    parser.add_argument("--set_conf", type=float, default=5)
    parser.add_argument("--n_pred_views", type=int, default=5)
    parser.add_argument("--filter_all_masks", action="store_true", default=False)
    parser.add_argument("--tsdf", action="store_true", default=True,
                       help="Enable TSDF fusion and save fused meshes to PLY files")
    parser.add_argument("--tsdf_voxel_size", type=float, default=0.005,
                       help="Voxel size for TSDF fusion (default: 0.005)")
    parser.add_argument("--no_input_mask", action="store_true", default=False)
    parser.add_argument("--no_pred_mask", action="store_true", default=False)
    parser.add_argument("--no_filter_input_view", action="store_true", default=False)
    parser.add_argument("--use_segmentation_mask", action="store_true", default=True)
    parser.add_argument("--mask_combination", type=str, default="first", 
                       choices=["all", "first", "largest"], 
                       help="How to combine multiple object masks")
    parser.add_argument("--depth_scale", type=float, default=0.001, 
                       help="Scale factor for depth values (mm to m = 0.001)")
    parser.add_argument("--mesh_method", type=str, default="auto", 
                       choices=["auto", "poisson", "alpha_shape", "ball_pivoting", "convex_hull"],
                       help="Mesh reconstruction method (auto tries multiple methods)")
    parser.add_argument("--no_preprocess", action="store_true", default=False,
                       help="Skip point cloud preprocessing")
    parser.add_argument("--no_mesh_postprocess", action="store_true", default=False,
                       help="Skip mesh postprocessing")
    args = parser.parse_args()
    
    print("Loading checkpoint from Huggingface")
    rayst3r_checkpoint = hf_hub_download("bartduis/rayst3r", "rayst3r.pth")
    
    model = EvalWrapper(rayst3r_checkpoint, distributed=False)
    
    print(f"Evaluating custom scene:")
    print(f"  Zarr path: {args.zarr_path}")
    print(f"  Episode: {args.episode_idx}, Frame: {args.frame_idx}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Use segmentation mask: {args.use_segmentation_mask}")
    print(f"  Mask combination: {args.mask_combination}")
    print(f"  Depth scale: {args.depth_scale}")
    
    all_points, mesh = eval_custom_scene(
        model, args.zarr_path, args.episode_idx, args.frame_idx,
        output_dir=args.output_dir, visualize=args.visualize, rr_addr=args.rr_addr,
        run_octmae=args.run_octmae, set_conf=args.set_conf,
        no_input_mask=args.no_input_mask, no_pred_mask=args.no_pred_mask,
        no_filter_input_view=args.no_filter_input_view, n_pred_views=args.n_pred_views,
        do_filter_all_masks=args.filter_all_masks, tsdf=args.tsdf,
        use_segmentation_mask=args.use_segmentation_mask,
        mask_combination=args.mask_combination, depth_scale=args.depth_scale,
        mesh_method=args.mesh_method, no_preprocess=args.no_preprocess,
        no_mesh_postprocess=args.no_mesh_postprocess, tsdf_voxel_size=args.tsdf_voxel_size
    )
    
    print(f"\nEvaluation completed!")
    print(f"Generated {len(all_points)} 3D points")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
