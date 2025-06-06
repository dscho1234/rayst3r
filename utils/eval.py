import torch

def eval_pred(pred_dict, gt_dict,accuracy_tresh=[0.001,0.01,0.02,0.05,0.1,0.5]):
    pointmaps_pred = pred_dict['pointmaps']
    pointmaps_gt = gt_dict['pointmaps']
    mask = gt_dict['valid_masks'].unsqueeze(-1).repeat(1,1,1,3)

    points_pred = pointmaps_pred[mask].reshape(-1,3)
    points_gt = pointmaps_gt[mask].reshape(-1,3)
    dists = torch.norm(points_pred - points_gt, dim=1)
    results = {'dist':dists.mean().detach().item()}
    if 'classifier' in pred_dict:
        classifier_pred = (torch.sigmoid(pred_dict['classifier']) > 0.5).bool()
        classifier_gt = gt_dict['valid_masks']
        results['classifier_acc'] = (classifier_pred == classifier_gt).float().mean().detach().item()
    
    for tresh in accuracy_tresh:
        acc = (dists < tresh).float().mean()
        results[f'acc_{tresh}'] = acc.detach().item()
    return results
