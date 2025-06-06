bb = breakpoint
import torch
from torch.utils.data import DataLoader
import wandb
from argparse import ArgumentParser
from datasets.octmae import OctMae
from datasets.foundation_pose import FoundationPose
from datasets.generic_loader import GenericLoader

from utils.collate import collate
from models.rayquery import RayQuery
from engine import train_epoch, eval_epoch, eval_model
import torch.nn as nn
from models.rayquery import RayQuery, PointmapEncoder, RayEncoder
from models.losses import *
import utils.misc as misc
import os
from utils.viz import just_load_viz
from utils.fusion import fuse_batch
import socket
import time
from utils.augmentations import *

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_train", type=str, default="TableOfCubes(size=10,n_views=2,seed=747)")
    parser.add_argument("--dataset_test", type=str, default="TableOfCubes(size=10,n_views=2,seed=787)")
    parser.add_argument("--dataset_just_load", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--model", type=str, default="RayQuery(ray_enc=RayEncoder(),pointmap_enc=PointmapEncoder(),criterion=RayCompletion(ConfLoss(L21)))")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_every", type=int, default=3)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default="init")
    parser.add_argument("--just_load", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rr_addr", type=str, default="0.0.0.0:"+os.getenv("RERUN_RECORDING","9876"))
    parser.add_argument("--mesh", action="store_true")
    parser.add_argument("--max_norm", type=float, default=-1)
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10) 
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--normalize_mode',type=str,default='None')
    parser.add_argument('--start_from',type=str,default=None)
    parser.add_argument('--augmentor',type=str,default='None')
    return parser.parse_args()

def main(args):
    load_dino = False
    if not args.just_load:
        dataset_train = eval(args.dataset_train)
        dataset_test = eval(args.dataset_test)
        if not dataset_train.prefetch_dino:
            load_dino = True
        rank, world_size, local_rank = misc.setup_distributed()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=world_size, rank=rank, shuffle=True
        )

        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=world_size, rank=rank, shuffle=False
        )

        train_loader = DataLoader(
            dataset_train, sampler=sampler_train, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
            num_workers=args.n_workers,
            pin_memory=True,
            prefetch_factor=2,
            drop_last=True
        )
        test_loader = DataLoader(
            dataset_test, sampler=sampler_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
            num_workers=args.n_workers,
            pin_memory=True,
            prefetch_factor=2,
            drop_last=True
        )

        n_scenes_epoch = len(train_loader) * args.batch_size * world_size
        print(f"Number of scenes in epoch: {n_scenes_epoch}")
    else:
        if args.dataset_just_load is None:
            dataset = eval(args.dataset_train)
        else:
            dataset = eval(args.dataset_just_load)
        if not dataset.prefetch_dino:
            load_dino = True
        rank, world_size, local_rank = misc.setup_distributed()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        just_loader = DataLoader(dataset, sampler=sampler_train, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
            pin_memory=True,
            drop_last=True
        )
    
    model = eval(args.model).to(args.device)
    if args.augmentor != 'None':
        augmentor = eval(args.augmentor)
    else:
        augmentor = None
    
    if load_dino and len(model.dino_layers) > 0:
        dino_model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitl14_reg")
        dino_model.eval()
        dino_model.to("cuda")
    else:
        dino_model = None
    # distribute model
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)
    model_without_ddp = model.module if hasattr(model, 'module') else model

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    os.makedirs(args.logdir,exist_ok=True)
    start_epoch = 0
    print("Running on host %s" % socket.gethostname())
    if args.resume and os.path.exists(os.path.join(args.resume, "checkpoint-latest.pth")):
        checkpoint = torch.load(os.path.join(args.resume, "checkpoint-latest.pth"), map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        model_params = list(model.parameters())
        print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            print("With optim & sched!")
        del checkpoint
    elif args.start_from is not None:
        checkpoint = torch.load(args.start_from, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Start from checkpoint %s" % args.start_from)
    if args.just_load:
        with torch.no_grad():
            while True:
                #test_log_dict = eval_epoch(model,just_loader,device=args.device,dino_model=dino_model,args=args)
                for data in just_loader:
                    pred, gt, loss_dict, batch = eval_model(model,data,mode='viz',args=args,dino_model=dino_model,augmentor=augmentor)
                    # cast to float32 for visualization
                    gt = {k: v.float() for k, v in gt.items()}
                    pred = {k: v.float() for k, v in pred.items()}
                    #loss_dict = eval_model(model,data,mode='loss',device=args.device)
                    #print(f"Loss: {loss_dict['loss']:.4f}")
                    # summarize all keys in loss_dict in table
                    print(f"{'Key':<10} {'Value':<10}")
                    print("-"*20)
                    for key, value in loss_dict.items():
                        print(f"{key:<10}: {value:.4f}")
                    print("-"*20)
                    name = args.logdir
                    addr = args.rr_addr
                    if args.mesh:
                        fused_meshes = fuse_batch(pred,gt,data, voxel_size=0.002)
                    else:
                        fused_meshes = None
                    just_load_viz(pred,gt,batch,addr=addr,name=name,fused_meshes=fused_meshes)
                    breakpoint()
        return
    else: 
        if args.wandb_project and misc.get_rank() == 0:
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)
            log_wandb = args.wandb_project
        else:
            log_wandb = None
        for epoch in range(start_epoch,args.n_epochs):
            start_time = time.time()
            log_dict = train_epoch(model,train_loader,optimizer,device=args.device,max_norm=args.max_norm,epoch=epoch,
                                   log_wandb=log_wandb,batch_size=eff_batch_size,args=args,dino_model=dino_model,augmentor=augmentor)
            end_time = time.time()
            print(f"Epoch {epoch} train loss: {log_dict['loss']:.4f} grad_norm: {log_dict['grad_norm']:.4f} \n")
            print(f"Time taken for epoch {epoch}: {end_time - start_time:.2f} seconds")
            
            if epoch % args.eval_every == 0:
                test_log_dict = eval_epoch(model,test_loader,device=args.device,dino_model=dino_model,args=args,augmentor=augmentor)
                print(f"Epoch {epoch} test loss: {test_log_dict['loss']:.4f} \n")
                if log_wandb:
                    wandb_dict = {f"test_{k}":v for k,v in test_log_dict.items()}
                    wandb.log(wandb_dict, step=(epoch+1)*n_scenes_epoch)
            if epoch % args.save_every == 0:
                # this saves the model every epoch and doesn't overwrite but it becomes tremendous, huge
                #misc.save_model(args, epoch, model, optimizer)
                misc.save_model(args, epoch, model_without_ddp, optimizer, epoch_name=f"latest")

if __name__ == "__main__":
    args = parse_args()
    main(args)