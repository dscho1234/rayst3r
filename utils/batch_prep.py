import torch
import torchvision.transforms as tvf

dino_patch_size = 14

def batch_to_device(batch,device='cuda'):
    for key in batch:
        if isinstance(batch[key],torch.Tensor):
            batch[key] = batch[key].to(device)
        elif isinstance(batch[key],dict):
            batch[key] = batch_to_device(batch[key],device)
    return batch


def compute_pointmap(depth: torch.Tensor, intrinsics: torch.Tensor, cam2world: torch.Tensor = None) -> torch.Tensor:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    h, w = depth.shape
        
    i, j = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
    i = i.to(depth.device)
    j = j.to(depth.device)

    x_cam = (i - cx) * depth / fx
    y_cam = (j - cy) * depth / fy

    points_cam = torch.stack([x_cam, y_cam, depth], axis=-1)

    if cam2world is not None:
        points_cam = torch.matmul(cam2world[:3, :3], points_cam.reshape(-1, 3).T).T + cam2world[:3, 3]
    points_cam = points_cam.reshape(h, w, 3)

    return points_cam

def compute_pointmaps(depths: torch.Tensor, intrinsics: torch.Tensor, cam2worlds: torch.Tensor) -> torch.Tensor:
    pointmaps = []
    depth_shape = depths.shape
    pointmaps_shape = depths.shape + (3,)
    for depth, K, c2w in zip(depths, intrinsics, cam2worlds):
        n_views = depth.shape[0]
        for i in range(n_views):
            pointmaps.append(compute_pointmap(depth[i], K[i],c2w[i]))
    return torch.stack(pointmaps).reshape(pointmaps_shape)

def depth_to_metric(depth):
    # depth: shape H x W
    # we want to convert the depth to a metric depth
    depth_max = 10.0
    depth_scaled = depth_max * (depth / 65535.0)
    
    return depth_scaled

def make_rgb_transform() -> tvf.Compose:
    return tvf.Compose([
        #tvf.ToTensor(),
        #lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        tvf.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])

rgb_transform = make_rgb_transform()

def compute_dino_and_store_features(dino_model : torch.nn.Module, rgb: torch.Tensor, mask: torch.Tensor,dino_layers: list[int] = None) -> torch.Tensor:
    """Computes the DINO features given an RGB image."""
    rgb = rgb.squeeze(1)
    mask = mask.squeeze(1)
    rgb = rgb.permute(0,3,1,2)
    mask = mask.unsqueeze(1).repeat(1,3,1,1)
    rgb = rgb * mask

    rgb = rgb.float()
    H, W = rgb.shape[-2:]
    goal_H, goal_W = H//dino_patch_size*dino_patch_size, W//dino_patch_size*dino_patch_size
    resize_transform = tvf.CenterCrop([goal_H, goal_W])
    with torch.no_grad():
        rgb = resize_transform(rgb)
        rgb = rgb_transform(rgb)
        all_feat = dino_model.get_intermediate_layers(rgb, dino_layers)
        dino_feat = torch.cat(all_feat, dim=-1)
    return dino_feat


def prepare_fast_batch(batch,dino_model = None,dino_layers = None, convert_depth_to_metric = True):
    # depth to metric    
    if convert_depth_to_metric:
        batch['new_cams']['depths'] = depth_to_metric(batch['new_cams']['depths'])
        batch['input_cams']['depths'] = depth_to_metric(batch['input_cams']['depths'])

    # compute pointmaps
    batch['new_cams']['pointmaps'] = compute_pointmaps(batch['new_cams']['depths'],batch['new_cams']['Ks'],batch['new_cams']['c2ws'])
    batch['input_cams']['pointmaps'] = compute_pointmaps(batch['input_cams']['depths'],batch['input_cams']['Ks'],batch['input_cams']['c2ws'])
    
    # compute dino features
    if dino_model is not None and len(dino_layers) > 0:
        batch['input_cams']['dino_features'] = compute_dino_and_store_features(dino_model,batch['input_cams']['imgs'],batch['input_cams']['valid_masks'],dino_layers)
    
    return batch


def normalize_batch(batch,normalize_mode):
    scale_factors = []
    if normalize_mode == 'None':
        pass
    elif normalize_mode == 'median':
        B = batch['input_cams']['valid_masks'].shape[0]
        for b in range(B):
            input_mask = batch['input_cams']['valid_masks'][b]
            depth_median = batch['input_cams']['depths'][b][input_mask].median()
            scale_factor = 1.0 / depth_median
            scale_factors.append(scale_factor)
            batch['input_cams']['depths'][b] = scale_factor * batch['input_cams']['depths'][b]
            batch['input_cams']['pointmaps'][b] = scale_factor * batch['input_cams']['pointmaps'][b]
            batch['input_cams']['c2ws'][b][0,:3,-1] = scale_factor * batch['input_cams']['c2ws'][b][0,:3,-1]

            batch['new_cams']['depths'][b] = scale_factor * batch['new_cams']['depths'][b]
            batch['new_cams']['pointmaps'][b] = scale_factor * batch['new_cams']['pointmaps'][b]
            batch['new_cams']['c2ws'][b][:,:3,-1] = scale_factor * batch['new_cams']['c2ws'][b][:,:3,-1]

    return batch, scale_factors

def denormalize_batch(batch,pred,gt,scale_factors):
    B = len(scale_factors)
    n_new_cams = batch['new_cams']['c2ws'].shape[1]
    for b in range(B):
        new_scale_factor = 1.0 / scale_factors[b]
        batch['input_cams']['depths'][b] = new_scale_factor * batch['input_cams']['depths'][b]
        batch['input_cams']['pointmaps'][b] = new_scale_factor * batch['input_cams']['pointmaps'][b]
        batch['input_cams']['c2ws'][b][:,:3,-1] = new_scale_factor * batch['input_cams']['c2ws'][b][:,:3,-1]
        batch['new_cams']['depths'][b] = new_scale_factor * batch['new_cams']['depths'][b]
        batch['new_cams']['pointmaps'][b] = new_scale_factor * batch['new_cams']['pointmaps'][b]
        batch['new_cams']['c2ws'][b][:,:3,-1] = new_scale_factor * batch['new_cams']['c2ws'][b][:,:3,-1]

        pred['depths'][b] = new_scale_factor * pred['depths'][b]

        gt['c2ws'][b][:,:3,-1] = new_scale_factor * gt['c2ws'][b][:,:3,-1]
        gt['depths'][b] = new_scale_factor * gt['depths'][b]
        
        gt['pointmaps'][b] = compute_pointmaps(gt['depths'][b].unsqueeze(1),gt['Ks'][b].unsqueeze(1),gt['c2ws'][b].unsqueeze(1)).squeeze(1)
        pred['pointmaps'][b] = compute_pointmaps(pred['depths'][b].unsqueeze(1),gt['Ks'][b].unsqueeze(1),gt['c2ws'][b].unsqueeze(1)).squeeze(1)
    return batch, pred, gt
