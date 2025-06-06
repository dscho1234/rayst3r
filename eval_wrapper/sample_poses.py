import numpy as np
import torch
import open3d as o3d


def look_at(cam_pos, target=(0,0,0), up=(0,0,1)):
    # Forward vector
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)

    # Default up vector
    right = np.cross(up, forward)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([1, 0, 0])
        right = np.cross(up, forward)

    right /= np.linalg.norm(right)
    up = np.cross(forward, right)

    # Build rotation and translation matrices
    rotation = np.eye(4)
    rotation[:3, :3] = np.vstack([right, up, -forward]).T

    
    translation = np.eye(4)
    translation[:3, 3] = cam_pos

    cam_to_world = translation @ rotation
    cam_to_world[:3,2] = -cam_to_world[:3,2]
    cam_to_world[:3,1] = -cam_to_world[:3,1]
    # rotate 90 degrees around z axis 
    return cam_to_world


def sample_camera_poses(target: np.ndarray, inner_radius: float, outer_radius: float, n: int,seed: int = 42,mode: str = 'grid') -> np.ndarray:
    """
    Samples `n` camera poses uniformly on a sphere of given `radius` around `target`.
    The cameras are positioned randomly and oriented to look at `target`.

    Args:
        target (np.ndarray): 3D point (x, y, z) that cameras should look at.
        inner_radius (float): Radius of the sphere.
        outer_radius (float): Radius of the sphere.
        n (int): Number of camera poses to sample.
    
    Returns:
        torch.Tensor: (n, 4, 4) array of transformation matrices (camera-to-world).
    """
    cameras = []
    np.random.seed(seed)
    
    u_1 = np.linspace(0,1,n,endpoint=False)
    u_2 = np.linspace(0,0.7,n)
    u_1, u_2 = np.meshgrid(u_1, u_2)
    u_1 = u_1.flatten()
    u_2 = u_2.flatten()
    theta = np.arccos(1-2*u_2)
    phi = 2*np.pi*u_1
    n_poses = len(phi)
    
    radii = np.random.uniform(inner_radius, outer_radius, n_poses)
    cameras = []
    
    r_z = np.array([[0,-1,0],[1,0,0],[0,0,1]])
   
    for i in range(n_poses):
        # Camera position on the sphere
        x = target[0] + radii[i] * np.sin(theta[i]) * np.cos(phi[i])
        y = target[1] + radii[i] * np.sin(theta[i]) * np.sin(phi[i])
        z = target[2] + radii[i] * np.cos(theta[i])
        cam_pos = np.array([x, y, z])
        cam2world = look_at(cam_pos, target)
        if theta[i] == 0:
            cam2world[:3,:3] = cam2world[:3,:3] @ r_z # rotate 90 degrees around z axis for the camera opposite to the input
        cameras.append(cam2world)
    cameras = np.unique(cameras, axis=0)
    return np.stack(cameras)


def pointmap_to_poses(pointmaps: torch.Tensor, n_poses: int, inner_radius: float = 1.1, outer_radius: float = 2.5, device: str = 'cuda',
bb_mode: str='bb',run_octmae: bool = False) -> np.ndarray:
    """
    Samples `n_poses` camera poses uniformly on a sphere of given `radius` around `target`.
    The cameras are positioned randomly and oriented to look at `target`.
    """

    bb_min_corner = pointmaps.min(dim=0)[0].cpu().numpy()
    bb_max_corner = pointmaps.max(dim=0)[0].cpu().numpy()
    center = (bb_min_corner + bb_max_corner) / 2    #inner_radius = inner_radius * np.linalg.norm(bb_max_corner - bb_min_corner) / 2 # minimum radius is scalar multiple of bounding box radius
    bb_radius = np.linalg.norm(bb_max_corner - bb_min_corner) / 2
    cam2center_dist = np.linalg.norm(center)
        
    if run_octmae:
        radius = max(1.2*cam2center_dist,2.5*bb_radius) 
    else:
        radius = max(0.7*cam2center_dist,1.3*bb_radius)
    inner_radius = radius
    outer_radius = radius   
    camera_poses = sample_camera_poses(center, inner_radius, outer_radius, n_poses)
    return camera_poses
