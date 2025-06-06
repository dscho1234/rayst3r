import torch

def collate(batch):
    if isinstance(batch[0],dict):
        return {k: collate([d[k] for d in batch]) for k in batch[0].keys()}
    else:
        return torch.stack([torch.stack(t) for t in batch])