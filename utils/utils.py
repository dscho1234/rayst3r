import torch
import numpy as np

def to_tensor(x,dtype=torch.float64):
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x.copy()).to(dtype)
    else:
        raise ValueError(f"Unsupported type: {type(x)}")

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise ValueError(f"Unsupported type: {type(x)}")
    
def invalid_to_nans( arr, valid_mask, ndim=999 ):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = float('nan')
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr

def invalid_to_zeros( arr, valid_mask, ndim=999 ):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = 0
        nnz = valid_mask.view(len(valid_mask), -1).sum(1)
    else:
        nnz = arr.numel() // len(arr) if len(arr) else 0 # number of point per image
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr, nnz

def scenes_to_batch(scenes,repeat=None):
    batch = {}
    n_cams = None
    
    if 'new_cams' in scenes:
        n_cams = scenes['new_cams']['depths'].shape[1]
        batch['new_cams'], n_cams = scenes_to_batch(scenes['new_cams'])
        batch['input_cams'],_ = scenes_to_batch(scenes['input_cams'],repeat=n_cams)
    else:
        for key in scenes.keys():
            shape = scenes[key].shape
            if len(shape) > 3 :
                n_cams = shape[1]
                if repeat is not None:
                    # repeat the 2nd dimension by repeat times to also have the inputs repeated in the batch
                    repeat_dims = (1,) * len(shape)  # (1,1,1,...) for all dimensions
                    repeat_dims = list(repeat_dims)
                    repeat_dims[1] = repeat 
                    batch[key] = scenes[key].repeat(*repeat_dims)
                    batch[key] = batch[key].reshape(-1, *shape[2:])
                else:
                    batch[key] = scenes[key].reshape(-1, *shape[2:])
            elif key == 'dino_features':
                repeat_shape = (repeat,) + (1,) * (len(shape) - 1)
                batch[key] = scenes[key].repeat(*repeat_shape)
            else:
                batch[key] = scenes[key]
    return batch, n_cams

def dict_to_scenes(input_dict,n_cams):
    scenes = {}
    for key in input_dict.keys():
        if isinstance(input_dict[key],dict):
            scenes[key] = dict_to_scenes(input_dict[key],n_cams)
        else:
            scenes[key] = input_dict[key].reshape(-1, n_cams, *input_dict[key].shape[1:])
    return scenes

def batch_to_scenes(pred,gt,batch,n_cams):
    # pred
    batch = dict_to_scenes(batch,n_cams)
    pred = dict_to_scenes(pred,n_cams)
    gt = dict_to_scenes(gt,n_cams)
    return pred, gt, batch