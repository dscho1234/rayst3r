bb=breakpoint
import torch
from utils.geometry import center_pointmaps, uncenter_pointmaps
from utils.utils import scenes_to_batch, batch_to_scenes
from utils.batch_prep import prepare_fast_batch, normalize_batch, denormalize_batch
from utils.viz import save_pointmaps
from tqdm import tqdm
import wandb
from utils import misc
from torch.amp import GradScaler
from utils.eval import eval_pred
from utils.geometry import depth2pts

def batch_to_device(batch,device='cuda'):
    for key in batch:
        if isinstance(batch[key],torch.Tensor):
            batch[key] = batch[key].to(device)
        elif isinstance(batch[key],dict):
            batch[key] = batch_to_device(batch[key],device)
    return batch

def eval_model(model,batch,mode='loss',device='cuda',dino_model=None,args=None,augmentor=None,return_scale=False):
    batch = batch_to_device(batch,device)
    # check if model is distributed
    if isinstance(model,torch.nn.parallel.DistributedDataParallel):
        dino_layers = model.module.dino_layers
    else:
        dino_layers = model.dino_layers
    if 'pointmaps' not in list(batch['input_cams'].keys()):
        batch = prepare_fast_batch(batch,dino_model,dino_layers)

    normalize_mode = args.normalize_mode if args is not None else 'median'
    batch, scale_factors = normalize_batch(batch,normalize_mode)
    if augmentor is not None:
        batch = augmentor(batch)

    batch, n_cams = scenes_to_batch(batch)
    batch = center_pointmaps(batch) # centering around first camera

    device = args.device if args is not None else 'cuda'
    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        pred, gt, loss_dict = model(batch,mode='viz')
    
    if 'pointmaps' not in list(pred.keys()):
        pred['pointmaps'] = depth2pts(pred['depths'].squeeze(-1),batch['new_cams']['Ks'])
    elif 'depths' not in list(pred.keys()):
        pred['depths'] = pred['pointmaps'][...,-1]
    loss_dict = {**loss_dict,**eval_pred(pred, gt)}
    if mode == 'loss':
        return loss_dict
    elif mode == 'viz':
        pred, gt, batch = uncenter_pointmaps(pred, gt, batch)
        pred, gt, batch = batch_to_scenes(pred, gt,batch, n_cams)
        if return_scale:
            return pred, gt, loss_dict, scale_factors[0].item()
        else:
            return pred, gt, loss_dict
    else:
        raise ValueError(f"Invalid mode: {mode}")

def update_loss_dict(loss_dict,loss_dict_new,sample_count):
    for key in loss_dict_new:
        if key not in loss_dict:
            loss_dict[key] = loss_dict_new[key]
        else:
            # previously stored value in loss_dict is average from sample_count samples
            # so we need to update it to include the new sample
            loss_dict[key] = (loss_dict[key] * sample_count + loss_dict_new[key]) / (sample_count + 1)
    return loss_dict

def train_epoch(model, train_loader, optimizer, device='cuda', max_norm=1.0,log_wandb=False,epoch=0,batch_size=None,args=None,dino_model=None,augmentor=None):
    model.train()
    all_losses_dict = {}
    
    sample_idx = epoch * batch_size * len(train_loader)
    scaler = GradScaler()
    for i, batch in tqdm(enumerate(train_loader),total=len(train_loader)):
        optimizer.zero_grad()
        new_loss_dict = eval_model(model, batch, mode='loss', device=device,dino_model=dino_model,args=args,augmentor=augmentor)
        loss = new_loss_dict['loss']
        if loss is None:
            continue
        
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        
        grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in model.parameters() if p.grad is not None]))
        if grad_norm.isnan():
            breakpoint()
        
        ## Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        new_loss_dict['grad_norm'] = grad_norm.detach().cpu().item()

        misc.adjust_learning_rate(optimizer, epoch + i/len(train_loader), args)
        optimizer.step()
        
        new_loss_dict = {k: (v.detach().cpu().item() if isinstance(v, torch.Tensor) else v) for k, v in new_loss_dict.items()}
        if log_wandb:
            wandb_dict = {f"train_{k}":v for k,v in new_loss_dict.items()}
            wandb.log(wandb_dict, step=sample_idx + (i+1)*batch_size)
            # log learning rate
            wandb.log({"train_lr": optimizer.param_groups[0]['lr']}, step=sample_idx + (i+1)*batch_size)
        
        all_losses_dict = update_loss_dict(all_losses_dict, new_loss_dict,sample_count=i)
        # Clear cache and delete variables to free memory
        torch.cuda.empty_cache()
        del loss
        del new_loss_dict
        del grad_norm
        del batch
    
    return all_losses_dict

def eval_epoch(model,test_loader,device='cuda',dino_model=None,args=None,augmentor=None):
    model.eval()
    all_losses_dict = {}
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader),total=len(test_loader)):
            new_loss_dict = eval_model(model,batch,mode='loss',device=device,dino_model=dino_model,args=args,augmentor=augmentor)
            if new_loss_dict is None:
                continue
            all_losses_dict = update_loss_dict(all_losses_dict,new_loss_dict,sample_count=i)

            torch.cuda.empty_cache()
            del new_loss_dict
            del batch
    
    return all_losses_dict