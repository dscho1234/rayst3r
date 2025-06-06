bb = breakpoint
import torch
import torch.nn as nn
import copy
from utils.geometry import normalize_pointcloud

class Criterion (nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        self.criterion = copy.deepcopy(criterion)

    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'

class CrocoLoss (nn.Module):
    def __init__(self,mode='vanilla',eps=1e-4):
        super().__init__()
        self.mode = mode
    def get_name(self):
        return f'CrocoLoss({self.mode})'

    def forward(self, pred, gt, **kw):        
        pred_pts = pred['pointmaps']
        conf = pred['conf']
        
        if self.mode == 'vanilla':
            loss = torch.abs(gt-pred_pts)/(torch.exp(conf)) + conf
        elif self.mode == 'bounded_1':
            a=0.25
            b=4.
            conf = (b-a)*torch.sigmoid(conf) + a
            loss = torch.abs(gt-pred_pts)/(conf) + torch.log(conf)
        elif self.mode == 'bounded_2':
            a = 3.0
            b = 3.0
            conf = 2*a * (torch.sigmoid(conf/b)-0.5)
            loss = torch.abs(gt-pred_pts)/torch.exp(conf) + conf
        return loss.mean()

class SMDLoss (nn.Module):
    def __init__(self,raw_loss,mode='linear'):
        super().__init__()
        self.mode = mode
        self.raw_loss = raw_loss
    def get_name(self):
        return f'SMDLoss({self.raw_loss},{self.mode})'

    def forward(self, pred, gt,eps, **kw):        
        p_gt = compute_probs(pred,gt,eps=eps)
        # filtering out nan values
        loss = self.raw_loss(p_gt)
        loss_mask = ~torch.isnan(p_gt) & (loss != torch.inf).bool() 
        loss = loss[loss_mask]
        return loss.mean()

# https://github.com/naver/dust3r/blob/c9e9336a6ba7c1f1873f9295852cea6dffaf770d/dust3r/losses.py#L197
class ConfLoss (nn.Module):
    """ Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10) 

        alpha: hyperparameter
    """

    def __init__(self, raw_loss, alpha=0.2,skip_conf=False):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.raw_loss = raw_loss
        self.skip_conf = skip_conf
    
    def get_name(self):
        return f'ConfLoss({self.raw_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def forward(self, pred, gt,conf, **kw):
        # compute per-pixel loss
        loss = self.raw_loss(gt, pred, **kw)
        # weight by confidence
        if not self.skip_conf:
            conf, log_conf = self.get_conf_log(conf)
            conf_loss = loss * conf - self.alpha * log_conf
            ## average + nan protection (in case of no valid pixels at all)
            conf_loss = conf_loss.mean() if conf_loss.numel() > 0 else 0
            return conf_loss
        else:
            return loss.mean()


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return f'BCELoss()'
    
    def forward(self, gt, pred):
   #     return torch.nn.functional.binary_cross_entropy(pred, gt)
        return torch.nn.functional.binary_cross_entropy_with_logits(pred, gt)

class ClassifierLoss(nn.Module):
    def __init__(self,criterion):
        super().__init__()
        self.criterion = criterion

    def get_name(self):
        return f'ClassifierLoss({self.criterion})'

    def forward(self, pred, gt):
        return self.criterion(pred, gt)

class BaseCriterion(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

class NLLLoss (BaseCriterion):
    """ Negative log likelihood loss """
    def forward(self, pred):
        # assuming the pred is already a log (for stability sake)
        return -pred
        #return -torch.log(pred)

class LLoss (BaseCriterion):
    """ L-norm loss
    """
    def forward(self, a, b):
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, f'Bad shape = {a.shape}'
        dist = self.distance(a, b)
        assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f'bad {self.reduction=} mode')

    def distance(self, a, b):
        raise NotImplementedError()

class L21Loss (LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1) 

L21 = L21Loss()

def apply_log_to_norm(xyz):
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    xyz = xyz * torch.log1p(d)
    return xyz

class DepthCompletion (Criterion):
    def __init__(self, criterion, classifier_criterion=None,norm_mode='?None', loss_in_log=False,device='cuda',lambda_classifier=1.0):
        super().__init__(criterion)
        self.criterion.reduction = 'none' 
        self.loss_in_log = loss_in_log
        self.device = device
        self.lambda_classifier = lambda_classifier
        self.classifier_criterion = classifier_criterion

        if norm_mode.startswith('?'):
            # do no norm pts from metric scale datasets
            self.norm_all = False
            self.norm_mode = norm_mode[1:]
        else:
            self.norm_all = True
            self.norm_mode = norm_mode
    
    def forward(self, pred_dict, gt_dict,**kw):
        gt_depths = gt_dict['depths']
        pred_depths = pred_dict['depths']
        gt_masks = gt_dict['valid_masks']
        if gt_masks.sum() == 0:
            return None
        else:
            gt_depths_masked = gt_depths[gt_masks].view(-1,1)
            pred_depths_masked = pred_depths[gt_masks].view(-1,1)
            # this is a loss on the points on the objects
            loss_dict =  {'loss_points':self.criterion(pred_depths_masked, gt_depths_masked,pred_dict['conf_pointmaps'][gt_masks])}
            # loss on predicting a mask for the points on the objects
            if 'classifier' in pred_dict and self.classifier_criterion is not None:
                loss_dict['loss_classifier'] = self.classifier_criterion(pred_dict['classifier'], gt_dict['valid_masks'].float(),pred_dict['conf_classifier'])
                loss_dict['loss'] = loss_dict['loss_points'] + self.lambda_classifier * loss_dict['loss_classifier']
            else:
                loss_dict['loss'] = loss_dict['loss_points']

            return loss_dict


class RayCompletion (Criterion):
    def __init__(self, criterion, classifier_criterion=None,norm_mode='?None', loss_in_log=False,device='cuda',lambda_classifier=1.0):
        super().__init__(criterion)
        self.criterion.reduction = 'none' 
        self.loss_in_log = loss_in_log
        self.device = device
        self.lambda_classifier = lambda_classifier
        self.classifier_criterion = classifier_criterion

        if norm_mode.startswith('?'):
            # do no norm pts from metric scale datasets
            self.norm_all = False
            self.norm_mode = norm_mode[1:]
        else:
            self.norm_all = True
            self.norm_mode = norm_mode
    
    def get_all_pts3d(self, gt_dict, pred_dict):
        gt_pts1 = gt_dict['pointmaps']
        #gt_pts_context = gt_dict['pointmaps_context'][:,0] # we use the first camera given as input for normalization, in our current case that's the only cam
        if 'pointmaps' in pred_dict:
            pr_pts1 = pred_dict['pointmaps']
        else:
            pr_pts1 = None
        mask = gt_dict['valid_masks'].clone()
        # normalize 3d points
        norm_factor = None 

        return gt_pts1, pr_pts1, mask, norm_factor

    def forward(self, pred_dict, gt_dict, eps=None,**kw):
        gt_pts1, pred_pts1, mask, norm_factor = \
            self.get_all_pts3d(gt_dict, pred_dict, **kw)
        if mask.sum() == 0:
            return None
        else:
            mask_repeated = mask.unsqueeze(-1).repeat(1,1,1,3)
            if norm_factor is not None:
                pred_pts1 = pred_pts1 / norm_factor
                gt_pts1 = gt_pts1 / norm_factor

            pred_pts1 = pred_pts1[mask_repeated].reshape(-1,3)
            gt_pts1 = gt_pts1[mask_repeated].reshape(-1,3)
            
            if self.loss_in_log and self.loss_in_log != 'before':
                # this only make sense when depth_mode == 'exp'
                pred_pts1 = apply_log_to_norm(pred_pts1)
                gt_pts1 = apply_log_to_norm(gt_pts1)
            
            # this is a loss on the points on the objects
            loss_dict =  {'loss_points':self.criterion(pred_pts1, gt_pts1,pred_dict['conf_pointmaps'][mask])}
            # loss on predicting a mask for the points on the objects
            if 'classifier' in pred_dict and self.classifier_criterion is not None:
                loss_dict['loss_classifier'] = self.classifier_criterion(pred_dict['classifier'], gt_dict['valid_masks'].float(),pred_dict['conf_classifier'])
                loss_dict['loss'] = loss_dict['loss_points'] + self.lambda_classifier * loss_dict['loss_classifier']
            else:
                loss_dict['loss'] = loss_dict['loss_points']

            return loss_dict
