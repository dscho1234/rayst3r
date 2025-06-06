bb = breakpoint
import torch
import trimesh 
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pickle
import tqdm
import json 
from PIL import Image

class GenericLoader(torch.utils.data.Dataset):
    def __init__(self,dir="octmae_data/tiny_train/train_processed",seed=747,size=10,datasets=["fp_objaverse"],split="train",dtype=torch.float32,mode="slow",
                 prefetch_dino=False,dino_features=[23],view_select_mode="new_zoom",noise_std=0.0,rendered_views_mode="None",**kwargs):
        super().__init__(**kwargs)
        self.dir = dir
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.datasets = datasets
        self.split = split
        self.dtype = dtype
        self.mode = mode
        self.prefetch_dino = prefetch_dino
        self.view_select_mode = view_select_mode
        self.noise_std = noise_std * torch.iinfo(torch.uint16).max / 10.0 # variance in the range of the depth map, uint16 normalized to 10
        if self.mode == 'slow':
            self.prefetch_dino = True
        self.find_scenes()
        self.dino_features = dino_features
        self.rendered_views_mode = rendered_views_mode

    def find_dataset_location_list(self,dataset):
        data_dir = None
        for d in self.dir:
            datasets = os.listdir(d)
            if dataset in datasets:
                if data_dir is not None:
                    raise ValueError(f"Dataset {dataset} found in multiple locations: {self.dir}")
                else:
                    data_dir = os.path.join(d,dataset)
        if data_dir is None:
            raise ValueError(f"Dataset {dataset} not found in {self.dir}")
        return data_dir

    def find_dataset_location(self,dataset):
        if isinstance(self.dir,list):
            data_dir = self.find_dataset_location_list(dataset)
        else:
            data_dir = os.path.join(self.dir,dataset)
            if not os.path.exists(data_dir):
                raise ValueError(f"Dataset {dataset} not found in {self.dir}")
        return data_dir

    def find_scenes(self):
        all_scenes = {}
        print("Loading scenes...")
        for dataset in self.datasets:
            dataset_dir = self.find_dataset_location(dataset)
            scenes =  json.load(open(os.path.join(dataset_dir, f"{self.split}_scenes.json")))
            scene_ids = [dataset + "_" + f.split("/")[-2] + "_" +  f.split("/")[-1] for f in scenes]
            all_scenes.update(dict(zip(scene_ids, scenes)))
        self.scenes = all_scenes
        self.scene_ids = list(self.scenes.keys())
        # shuffle the scene ids
        self.rng.shuffle(self.scene_ids)
        if self.size > 0:
            self.scene_ids = self.scene_ids[:self.size]
        self.size = len(self.scene_ids)
        return scenes

    def __len__(self):
        return self.size

    def decide_context_view(self,cam_dir):
        # we pick the view furthest away from the origin as the view for conditioning
        cam_dirs = [d for d in os.listdir(cam_dir) if os.path.isdir(os.path.join(cam_dir,d)) and not d.startswith("gen")] # input cam needs rgb
        
        extrinsics = {c:torch.load(os.path.join(cam_dir,c,'cam2world.pt'),map_location='cpu',weights_only=True) for c in cam_dirs}
        dist_origin = {c:torch.linalg.norm(extrinsics[c][:3,3]) for c in extrinsics}
        
        if self.view_select_mode == 'new_zoom':
            # find the view with the maximum distance to the origin
            input_cam = max(dist_origin,key=dist_origin.get)
            # pick another random view to predict, excluding the context view
        elif self.view_select_mode == 'random':
            # pick a random view
            input_cam = str(self.rng.choice(list(dist_origin.keys())))
            # pick another random view to predict, excluding the context view
        else:
            raise ValueError(f"Invalid mode: {self.view_select_mode}")

        if self.rendered_views_mode == "None":
            pass
        elif self.rendered_views_mode == "random":
            cam_dirs = [d for d in os.listdir(cam_dir) if os.path.isdir(os.path.join(cam_dir,d))]
        elif self.rendered_views_mode == "always":
            cam_dirs_gen = [d for d in os.listdir(cam_dir) if os.path.isdir(os.path.join(cam_dir,d)) and d.startswith("gen")]
            if len(cam_dirs_gen) > 0:
                cam_dirs = cam_dirs_gen
        else:
            raise ValueError(f"Invalid mode: {self.rendered_views_mode}")

        possible_views = [v for v in cam_dirs if v != input_cam]
        new_cam = str(self.rng.choice(possible_views))
        return input_cam,new_cam

    def transform_pointmap(self,pointmap_cam,c2w):
        # pointmap: shape H x W x 3
        # cw2: shape 4 x 4
        # we want to transform the pointmap to the world frame
        pointmap_cam_h = torch.cat([pointmap_cam,torch.ones(pointmap_cam.shape[:-1]+(1,)).to(pointmap_cam.device)],dim=-1)
        pointmap_world_h = pointmap_cam_h @ c2w.T
        pointmap_world = pointmap_world_h[...,:3]/pointmap_world_h[...,3:4]
        return pointmap_world

    def load_scene_slow(self,input_cam,new_cam,cam_dir):
        
        data = dict(new_cams={},input_cams={})

        data['new_cams']['c2ws'] = [torch.load(os.path.join(cam_dir,new_cam,'cam2world.pt'),map_location='cpu',weights_only=True).to(self.dtype)]
        data['new_cams']['depths'] = [torch.load(os.path.join(cam_dir,new_cam,'depth.pt'),map_location='cpu',weights_only=True).to(self.dtype)]
        data['new_cams']['pointmaps'] = [self.transform_pointmap(torch.load(os.path.join(cam_dir,new_cam,'pointmap.pt'),map_location='cpu',weights_only=True).to(self.dtype),data['new_cams']['c2ws'][0])]
        data['new_cams']['Ks'] = [torch.load(os.path.join(cam_dir,new_cam,'intrinsics.pt'),map_location='cpu',weights_only=True).to(self.dtype)]
        data['new_cams']['valid_masks'] = [torch.load(os.path.join(cam_dir,new_cam,'mask.pt'),map_location='cpu',weights_only=True).to(torch.bool)]

        # add the context views
        data['input_cams']['c2ws'] = [torch.load(os.path.join(cam_dir,input_cam,'cam2world.pt'),map_location='cpu',weights_only=True).to(self.dtype)]
        data['input_cams']['depths'] = [torch.load(os.path.join(cam_dir,input_cam,'depth.pt'),map_location='cpu',weights_only=True).to(self.dtype)]
        data['input_cams']['pointmaps'] = [self.transform_pointmap(torch.load(os.path.join(cam_dir,input_cam,'pointmap.pt'),map_location='cpu',weights_only=True).to(self.dtype),data['input_cams']['c2ws'][0])]
        data['input_cams']['Ks'] = [torch.load(os.path.join(cam_dir,input_cam,'intrinsics.pt'),map_location='cpu',weights_only=True).to(self.dtype)]
        data['input_cams']['valid_masks'] = [torch.load(os.path.join(cam_dir,input_cam,'mask.pt'),map_location='cpu',weights_only=True).to(torch.bool)]
        data['input_cams']['imgs'] = [torch.load(os.path.join(cam_dir,input_cam,'rgb.pt'),map_location='cpu',weights_only=True).to(self.dtype)]
        data['input_cams']['dino_features'] = [torch.load(os.path.join(cam_dir,input_cam,f'dino_features_layer_{l}.pt'),map_location='cpu',weights_only=True).to(self.dtype) for l in self.dino_features]
        return data

    def depth_to_metric(self,depth):
        # depth: shape H x W
        # we want to convert the depth to a metric depth
        depth_max = 10.0
        depth_scaled = depth_max * (depth / 65535.0)
        return depth_scaled

    def load_scene_fast(self,input_cam,new_cam,cam_dir):
        data = dict(new_cams={},input_cams={})
        data['new_cams']['c2ws'] = [torch.load(os.path.join(cam_dir,new_cam,'cam2world.pt'),map_location='cpu',weights_only=True).to(self.dtype)]
        data['new_cams']['Ks'] = [torch.load(os.path.join(cam_dir,new_cam,'intrinsics.pt'),map_location='cpu',weights_only=True).to(self.dtype)]
        data['new_cams']['depths'] = [torch.from_numpy(np.array(Image.open(os.path.join(cam_dir,new_cam,'depth.png'))).astype(np.float32))]
        data['new_cams']['valid_masks'] = [torch.from_numpy(np.array(Image.open(os.path.join(cam_dir,new_cam,'mask.png'))))]

        data['input_cams']['c2ws'] = [torch.load(os.path.join(cam_dir,input_cam,'cam2world.pt'),map_location='cpu',weights_only=True).to(self.dtype)]
        data['input_cams']['Ks'] = [torch.load(os.path.join(cam_dir,input_cam,'intrinsics.pt'),map_location='cpu',weights_only=True).to(self.dtype)]
        data['input_cams']['depths'] = [torch.from_numpy(np.array(Image.open(os.path.join(cam_dir,input_cam,'depth.png'))).astype(np.float32))]
        data['input_cams']['valid_masks'] = [torch.from_numpy(np.array(Image.open(os.path.join(cam_dir,input_cam,'mask.png'))))]
        data['input_cams']['imgs'] = [torch.from_numpy(np.array(Image.open(os.path.join(cam_dir,input_cam,'rgb.png'))))]
        if self.prefetch_dino:
            data['input_cams']['dino_features'] = [torch.cat([torch.load(os.path.join(cam_dir,input_cam,f'dino_features_layer_{l}.pt'),map_location='cpu',weights_only=True).to(self.dtype) for l in self.dino_features],dim=-1)]
        return data

    def __getitem__(self,idx):
        cam_dir = os.path.join(self.scenes[self.scene_ids[idx]],'cameras')    
        #data['input_cams'] = {k:[v[0].unsqueeze(0)] for k,v in data['input_cams'].items()}
        input_cam,new_cam = self.decide_context_view(cam_dir)
        if self.mode == 'slow':
            data = self.load_scene_slow(input_cam,new_cam,cam_dir)
        else:
            data = self.load_scene_fast(input_cam,new_cam,cam_dir)
        return data
