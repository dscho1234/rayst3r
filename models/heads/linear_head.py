import torch
import torch.nn as nn
import torch.nn.functional as F
from .postprocess import postprocess

class LinearPts3d (nn.Module):
    """ 
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, net, has_conf=False,mode='pts3d'):
        super().__init__()
        self.patch_size = net.patch_size
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.has_conf = has_conf
        self.mode = mode
        self.classifier_mode = None
        if self.mode == 'pts3d':
            self.proj = nn.Linear(net.dec_embed_dim, (3 + has_conf)*self.patch_size**2)
        elif self.mode == 'depth':
            self.proj = nn.Linear(net.dec_embed_dim, (1 + has_conf)*self.patch_size**2)
        elif self.mode == 'classifier':
            self.proj = nn.Linear(net.dec_embed_dim, (1 + has_conf)*self.patch_size**2)
            self.classifier_mode = net.classifier_mode

    def setup(self, croconet):
        pass

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        # extract 3D points
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        # permute + norm depth
        return postprocess(feat, self.depth_mode, self.conf_mode,self.classifier_mode)
