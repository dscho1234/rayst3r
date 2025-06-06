bb = breakpoint
import torch 
import torch.nn as nn
from models.blocks import DecoderBlock, Block, PatchEmbed, PositionGetter
from models.pos_embed import get_2d_sincos_pos_embed, RoPE2D
from models.losses import *
from utils.geometry import center_pointmaps, compute_rays
from models.heads import head_factory

def init_weights(m):
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Parameter):
        nn.init.normal_(m, std=0.02)

class RayEncoder(nn.Module):
    def __init__(self,
                 dim=256,
                 patch_size=8,
                 img_size=(128,128),
                 depth=3,
                 num_heads=4,
                 pos_embed='RoPE100',
                 ):
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=patch_size, in_chans=2, embed_dim=dim)
        self.dim = dim
        if pos_embed.startswith('RoPE'):
            freq = float(pos_embed[len('RoPE'):])
            self.rope = RoPE2D(freq=freq)
        else:
            self.rope = None
        self.blocks = nn.ModuleList([Block(dim=dim, num_heads=num_heads,rope=self.rope) for _ in range(depth)])
        self.initialize_weights()

    def initialize_weights(self):
        # patch embed 
        self.patch_embed._init_weights()
        
        # linears and layer norms
        self.apply(init_weights)

    def forward(self, rays):
        rays = rays.permute(0,3,1,2)
        rays, pos = self.patch_embed(rays)
        for blk in self.blocks:
            rays = blk(rays, pos)
        return rays, pos

class PointmapEncoder(nn.Module):
    def __init__(self,
                 dim=256,
                 patch_size=8,
                 img_size=(128,128),
                 depth=3,
                 num_heads=4,
                 pos_embed='RoPE100',
                 ):
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=patch_size, in_chans=3, embed_dim=dim)
        self.dim = dim
        self.patch_size = patch_size

        if pos_embed.startswith('RoPE'):
            freq = float(pos_embed[len('RoPE'):])
            self.rope = RoPE2D(freq=freq)
        else:
            self.rope = None
        self.blocks = nn.ModuleList([Block(dim=dim, num_heads=num_heads,rope=self.rope) for _ in range(depth)])
        self.masked_token = nn.Parameter(torch.randn(1,1,3))
        self.initialize_weights()

    def initialize_weights(self):
        # patch embed 
        self.patch_embed._init_weights()
        
        # linears and layer norms
        self.apply(init_weights)

    def forward(self, pointmaps,masks=None):
        # replace masked points (not on object) with a learned token
        pointmaps[~masks] = self.masked_token.to(pointmaps.dtype).to(pointmaps.device)
        pointmaps = pointmaps.permute(0,3,1,2)
        pointmaps, pos = self.patch_embed(pointmaps)

        for blk in self.blocks:
            pointmaps = blk(pointmaps, pos)
        return pointmaps, pos

class RayQuery(nn.Module):
    def __init__(self,
                 ray_enc=RayEncoder(),
                 pointmap_enc=PointmapEncoder(),
                 dec_pos_embed='RoPE100',
                 decoder_dim=256,
                 decoder_depth=3,
                 decoder_num_heads=4,
                 imshape=(128,128),
                 pts_head_type='dpt',
                 classifier_head_type='dpt_mask',
                 criterion=ConfLoss(L21),
                 return_all_blocks=True,
                 depth_mode=('exp',-float('inf'),float('inf')),
                 conf_mode=('exp',1,float('inf')),
                 classifier_mode=('raw',0,1),
                 dino_layers=[23],
                 ):
        super().__init__()
        self.ray_enc = ray_enc
        self.pointmap_enc = pointmap_enc
        self.dec_depth = decoder_depth
        self.dec_embed_dim = decoder_dim
        self.enc_embed_dim = ray_enc.dim
        self.patch_size = pointmap_enc.patch_size
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.classifier_mode = classifier_mode
        self.skip_dino = len(dino_layers) == 0
        self.pts_head_type = pts_head_type
        self.classifier_head_type = classifier_head_type

        if dec_pos_embed.startswith('RoPE'):
            self.dec_pos_embed = RoPE2D(freq=100.0)
        else:
            raise NotImplementedError(f'{dec_pos_embed} not implemented')
        self.decoder_blocks = nn.ModuleList([DecoderBlock(dim=decoder_dim, num_heads=decoder_num_heads,
                                                          rope=self.dec_pos_embed) for _ in range(decoder_depth)])
        self.pts_head = head_factory(pts_head_type, 'pts3d', self, has_conf=True)
        
        self.classifier_head = head_factory(classifier_head_type, 'pts3d', self, has_conf=True)
        self.imshape = imshape
        self.criterion = criterion
        self.return_all_blocks = return_all_blocks

        # dino projection
        self.dino_layers = dino_layers
        self.dino_proj = nn.Linear(1024 * len(dino_layers), decoder_dim)
        self.dino_pos_getter = PositionGetter()

        self.initialize_weights()
    
    def initialize_weights(self):
        self.apply(init_weights)

    def forward_encoders(self, rays, pointmaps,masks=None):
        # encode rays
        rays, rays_pos = self.ray_enc(rays)
        
        # encode pointmaps
        B, H, W, C = pointmaps.shape
        pointmaps = pointmaps.reshape(B,H,W,C) # each pointmap is encoded separately
        pointmaps, pointmaps_pos = self.pointmap_enc(pointmaps,masks=masks)
        new_shape = pointmaps.shape
        pointmaps = pointmaps.reshape(new_shape[0],*new_shape[1:])
        pointmaps_pos = pointmaps_pos[:B]
        
        return rays, rays_pos, pointmaps, pointmaps_pos

    def forward_decoder(self, rays, rays_pos, pointmaps, pointmaps_pos):
        if self.return_all_blocks:
            all_blocks = []
            for blk in self.decoder_blocks:
                rays, pointmaps = blk(rays, pointmaps, rays_pos, pointmaps_pos)
                all_blocks.append(rays)
            return all_blocks
        else:
            for blk in self.decoder_blocks:
                rays, pointmaps = blk(rays, pointmaps, rays_pos, pointmaps_pos)
            return rays
    
    def get_dino_pos(self,dino_features):
        # dino runs on 14x14 patches
        # note: assuming we cropped or resized down!
        dino_H = self.imshape[0]//14
        dino_W = self.imshape[1]//14
        dino_pos = self.dino_pos_getter(dino_features.shape[0],dino_H,dino_W,dino_features.device)
        return dino_pos

    def forward(self,batch,mode='loss'):
        # prep for encoders
        rays = compute_rays(batch) # we are querying the first camera
        pointmaps_context = batch['input_cams']['pointmaps'] # we are using the other cameras as context
        input_masks = batch['input_cams']['valid_masks']
        
        # run the encoders
        rays, rays_pos, pointmaps, pointmaps_pos = self.forward_encoders(rays, pointmaps_context,masks=input_masks)
        ## adding dino features 
        if not self.skip_dino:
            dino_features = batch['input_cams']['dino_features']
            dino_features = self.dino_proj(dino_features)
            if len(dino_features.shape) == 4:
                dino_features = dino_features.squeeze(1)
            dino_pos = self.get_dino_pos(dino_features)
            pointmaps = torch.cat([pointmaps,dino_features],dim=1)
            pointmaps_pos = torch.cat([pointmaps_pos,dino_pos],dim=1)
        else:
            dino_features = None
            dino_pos = None
        # decoder
        rays = self.forward_decoder(rays, rays_pos, pointmaps, pointmaps_pos)
        pts_pred_dict = self.pts_head(rays, self.imshape)
        classifier_pred_dict = self.classifier_head(rays, self.imshape)
        
        pred_dict = {**pts_pred_dict,**classifier_pred_dict}
        gt_dict = batch['new_cams']
        loss_dict = self.criterion(pred_dict, gt_dict)

        del rays, rays_pos, pointmaps, pointmaps_pos, dino_features, dino_pos, pointmaps_context, input_masks, pts_pred_dict, classifier_pred_dict

        if mode == 'loss':
            # delete all the variables that are not needed
            del pred_dict, gt_dict
            return loss_dict
        elif mode == 'viz':
            return pred_dict, gt_dict, loss_dict
        else:
            raise ValueError(f"Invalid mode: {mode}")