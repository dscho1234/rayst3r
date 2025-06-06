import torch

def postprocess(out, depth_mode, conf_mode,classifier_mode=None):
    """
    extract 3D points/confidence from prediction head output
    """
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,3
    if classifier_mode is None:
        if fmap.shape[-1] == 4:
            res = dict(pointmaps=reg_dense_pts3d(fmap[:, :, :, :-1], mode=depth_mode))
        else:
            res = dict(depths=reg_dense_depth(fmap[:, :, :, 0], mode=depth_mode))
        if conf_mode is not None:
            res['conf_pointmaps'] = reg_dense_conf(fmap[:, :, :, -1], mode=conf_mode)
    else:
        res = dict(classifier=reg_dense_classifier(fmap[:, :, :, 0], mode=classifier_mode))
        if conf_mode is not None:
            res['conf_classifier'] = reg_dense_conf(fmap[:, :, :, 1], mode=conf_mode)
    
    return res

def reg_dense_classifier(x, mode):
    """
    extract classifier from prediction head output
    """
    mode, vmin, vmax = mode
    #return torch.sigmoid(x)
    return x

def reg_dense_depth(x, mode):
    """
    extract depth from prediction head output
    """
    mode, vmin, vmax = mode
    no_bounds = (vmin == -float('inf')) and (vmax == float('inf'))
    assert no_bounds
    if mode == 'linear':
        return x
    elif mode == 'square':
        return x.square().clip(min=vmin, max=vmax)
    elif mode == 'exp':
        return torch.exp(x).clip(min=vmin, max=vmax)    
    else:
        raise ValueError(f'bad {mode=}')

def reg_dense_pts3d(xyz, mode):
    """
    extract 3D points from prediction head output
    """
    mode, vmin, vmax = mode

    no_bounds = (vmin == -float('inf')) and (vmax == float('inf'))
    assert no_bounds

    if mode == 'linear':
        if no_bounds:
            return xyz  # [-inf, +inf]
        return xyz.clip(min=vmin, max=vmax)

    # distance to origin
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    if mode == 'square':
        return xyz * d.square()

    if mode == 'exp':
        return xyz * torch.expm1(d)
    raise ValueError(f'bad {mode=}')

def reg_dense_conf(x, mode):
    """
    extract confidence from prediction head output
    """
    mode, vmin, vmax = mode
    if mode == 'exp':
        return vmin + x.exp().clip(max=vmax-vmin)
    if mode == 'sigmoid':
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    raise ValueError(f'bad {mode=}')

