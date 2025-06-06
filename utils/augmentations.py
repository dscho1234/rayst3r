import random
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from torchvision.transforms import GaussianBlur
from utils.batch_prep import compute_pointmaps
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np

class ChangeBright(torch.nn.Module):
  def __init__(self,prob=0.5,mag=[0.5,2.0]):
    super().__init__()
    self.mag = mag
    self.prob = prob

  def forward(self,rgb):
    #if np.random.uniform()>=self.prob:
      #return rgb
    n = rgb.shape[0]
    apply_aug = np.random.uniform(0,1,size=n) < self.prob
    aug = iaa.MultiplyBrightness(np.random.uniform(self.mag[0],self.mag[1]))  #NOTE iaa has bug about deterministic, we sample ourselves
    rgb[apply_aug] = aug(images=rgb[apply_aug])
    return rgb

class ChangeContrast(torch.nn.Module):
  def __init__(self,prob=0.5,mag=[0.5,2.0]):
    self.mag = mag
    self.prob = prob

  def __call__(self,rgb):
    n = rgb.shape[0]
    apply_aug = np.random.uniform(0,1,size=n) < self.prob

    aug = iaa.GammaContrast(np.random.uniform(self.mag[0],self.mag[1]))
    rgb[apply_aug] = aug(images=rgb[apply_aug])
    return rgb

class SaltAndPepper:
  def __init__(self, prob=0.3, ratio=0.1, per_channel=True):
    self.prob = prob
    self.ratio = ratio
    self.per_channel = per_channel

  def __call__(self, rgb):
    n = rgb.shape[0]
    apply_aug = np.random.uniform(0,1,size=n) < self.prob
    aug = iaa.SaltAndPepper(self.ratio, per_channel=self.per_channel).to_deterministic()
    rgb[apply_aug] = aug(images=rgb[apply_aug])
    return rgb

class RGBGaussianNoise:
  def __init__(self, max_noise=10, prob=0.5):
    self.max_noise = max_noise
    self.prob = prob

  def __call__(self, rgb):
    n = rgb.shape[0]
    apply_aug = np.random.uniform(0,1,size=n) < self.prob

    shape = rgb.shape
    noise = np.random.normal(0, self.max_noise, size=shape).clip(-self.max_noise, self.max_noise)
    rgb[apply_aug] = (rgb[apply_aug].astype(float) + noise[apply_aug]).clip(0,255).astype(np.uint8)
    return rgb

# from https://github.com/mihdalal/manipgen/blob/master/manipgen/utils/obs_utils.py
class DepthWarping(torch.nn.Module):
    def __init__(self, std=0.5, prob=0.8):
        super().__init__()
        self.std = std
        self.prob = prob
    
    def forward(self, depths, device=None):
        if device is None:
            device = depths.device

        n, _, h, w = depths.shape

        # Generate Gaussian shifts
        gaussian_shifts = torch.normal(mean=0, std=self.std, size=(n, h, w, 2), device=device).float()
        apply_shifts = torch.rand(n, device=device) < self.prob
        gaussian_shifts[~apply_shifts] = 0.0

        # Create grid for the original coordinates
        xx = torch.linspace(0, w - 1, w, device=device)
        yy = torch.linspace(0, h - 1, h, device=device)
        xx = xx.unsqueeze(0).repeat(h, 1)
        yy = yy.unsqueeze(1).repeat(1, w)
        grid = torch.stack((xx, yy), 2).unsqueeze(0)  # Add batch dimension

        # Apply Gaussian shifts to the grid
        grid = grid + gaussian_shifts

        # Normalize grid values to the range [-1, 1] for grid_sample
        grid[..., 0] = (grid[..., 0] / (w - 1)) * 2 - 1
        grid[..., 1] = (grid[..., 1] / (h - 1)) * 2 - 1

        # Perform the remapping using grid_sample
        depth_interp = F.grid_sample(depths, grid, mode='bilinear', padding_mode='border', align_corners=True)

        # Remove the batch and channel dimensions
        depth_interp = depth_interp.squeeze(0).squeeze(0)

        return depth_interp

class DepthHoles(torch.nn.Module):
    def __init__(self, prob=0.5, kernel_size_lower=3, kernel_size_upper=27, sigma_lower=1.0, 
    sigma_upper=7.0, thresh_lower=0.6, thresh_upper=0.9):
        super().__init__()
        self.prob = prob
        self.kernel_size_lower = kernel_size_lower
        self.kernel_size_upper = kernel_size_upper
        self.sigma_lower = sigma_lower
        self.sigma_upper = sigma_upper
        self.thresh_lower = thresh_lower
        self.thresh_upper = thresh_upper

    def forward(self, depths, device=None):
        if device is None:
            device = depths.device

        n, _, h, w = depths.shape
        # generate random noise
        noise = torch.rand(n, 1, h, w, device=device)

        # apply gaussian blur
        k = random.choice(list(range(self.kernel_size_lower, self.kernel_size_upper+1, 2)))
        noise = GaussianBlur(kernel_size=k, sigma=(self.sigma_lower, self.sigma_upper))(noise)

        # normalize noise
        noise = (noise - noise.min()) / (noise.max() - noise.min())

        # apply thresholding
        thresh = torch.rand(n, 1, 1, 1, device=device) * (self.thresh_upper - self.thresh_lower) + self.thresh_lower
        mask = (noise > thresh)
        prob = self.prob
        keep_mask = torch.rand(n, device=device) < prob
        mask[~keep_mask, :] = 0

        return mask

class DepthNoise(torch.nn.Module):
    def __init__(self, std=0.005,prob=1.0):
        super().__init__()
        self.std = std
        self.prob = prob
        
    def forward(self, depths, device=None):
        if device is None:
            device = depths.device

        n, _, h, w = depths.shape
        apply_noise = torch.rand(n, device=device) < self.prob
        noise = torch.randn(n, 1, h, w, device=device) * self.std
        noise[~apply_noise] = 0.0
        return depths + noise

class Augmentor(torch.nn.Module):
    def __init__(self, depth_holes=DepthHoles(), depth_warping=DepthWarping(),depth_noise=DepthNoise(),
    rgb_operators=[ChangeBright(),SaltAndPepper(),ChangeContrast(),RGBGaussianNoise()]):
        super().__init__()
        self.depth_holes = depth_holes
        self.depth_warping = depth_warping
        self.depth_noise = depth_noise
        self.rgb_operators = rgb_operators

    def forward(self, batch):
        input_depths = batch['input_cams']['depths']
        if self.depth_holes.prob > 0:
            masks = self.depth_holes(input_depths)
            batch['input_cams']['valid_masks'][masks] = False
        #if self.depth_warping.prob > 0:
            #input_depths = self.depth_warping(input_depths)
        if self.depth_noise.prob > 0:
            input_depths = self.depth_noise(input_depths)
        
        input_rgbs = batch['input_cams']['imgs'].squeeze(1).cpu().numpy() # this is a bit inefficient, but it's ok..
        for op in self.rgb_operators:
            input_rgbs = op(input_rgbs)
        batch['input_cams']['imgs'] = torch.from_numpy(input_rgbs).cuda().unsqueeze(1)

        batch['input_cams']['depths'] = input_depths
        batch['input_cams']['pointmaps'] = compute_pointmaps(batch['input_cams']['depths'],batch['input_cams']['Ks'],batch['input_cams']['c2ws']) # now we're doing this twice, but alas
        return batch