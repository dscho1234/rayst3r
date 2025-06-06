# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# head factory
# --------------------------------------------------------
from .linear_head import LinearPts3d
from .dpt_head import create_dpt_head, create_dpt_head_mask, create_dpt_head_depth

def head_factory(head_type, output_mode, net, has_conf=False):
    """" build a prediction head for the decoder 
    """
    if head_type == 'linear' and output_mode == 'pts3d':
        return LinearPts3d(net, has_conf)
    if head_type == 'linear_depth' and output_mode == 'pts3d':
        return LinearPts3d(net, has_conf,mode='depth')
    if head_type == 'linear_classifier' and output_mode == 'pts3d':
        return LinearPts3d(net, has_conf,mode='classifier')
    elif head_type == 'dpt' and output_mode == 'pts3d':
        return create_dpt_head(net, has_conf=has_conf)
    elif head_type == 'dpt_depth' and output_mode == 'pts3d':
        return create_dpt_head_depth(net, has_conf=has_conf)
    elif head_type == 'dpt_mask' and output_mode == 'pts3d':
        return create_dpt_head_mask(net, has_conf=has_conf)
    else:
        raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")