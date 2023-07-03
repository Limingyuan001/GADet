# Copyright (c) OpenMMLab. All rights reserved.
from .accuracy import Accuracy, accuracy
from .ae_loss import AssociativeEmbeddingLoss
from .balanced_l1_loss import BalancedL1Loss, balanced_l1_loss
from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy)
from .dice_loss import DiceLoss
from .focal_loss import FocalLoss, sigmoid_focal_loss
from .gaussian_focal_loss import GaussianFocalLoss
from .gfocal_loss import DistributionFocalLoss, QualityFocalLoss
from .ghm_loss import GHMC, GHMR
from .iou_loss import (BoundedIoULoss, CIoULoss, DIoULoss, GIoULoss, IoULoss,
                       bounded_iou_loss, iou_loss)
from .kd_loss import KnowledgeDistillationKLDivLoss
from .mse_loss import MSELoss, mse_loss
from .pisa_loss import carl_loss, isr_p
from .seesaw_loss import SeesawLoss
from .smooth_l1_loss import L1Loss, SmoothL1Loss, l1_loss, smooth_l1_loss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss
from .varifocal_loss import VarifocalLoss
# from .area_loss import AreaLoss
# from .area_loss02 import AreaLoss
# from .area_loss05 import AreaLoss
# from .area_loss06 import AreaLoss
# from .area_loss08 import AreaLoss
# from .area_loss09 import AreaLoss
# from .area_loss10 import AreaLoss
# from .area_loss11 import AreaLoss
# from .area_loss12 import AreaLoss
# from .area_loss13 import AreaLoss
# from .area_loss14 import AreaLoss
# from .area_loss15 import AreaLoss
# from .area_loss16 import AreaLoss
# from .area_loss17 import AreaLoss
# from .area_loss18 import AreaLoss
# from .area_loss19 import AreaLoss
# from .area_loss20 import AreaLoss
# from .area_loss21 import AreaLoss
# from .area_loss22 import AreaLoss
# from .area_loss23 import AreaLoss

# from .area_loss25 import AreaLoss
# from .area_loss26 import AreaLoss
# from .area_loss27 import AreaLoss

# from .area_loss24 import AreaLoss
'''
别tm改这个了！！！这个没用!
改下面的AreaLoss_PDLC
'''
from .area_loss181 import AreaLoss #必须得有，但其实没用，因为yolox_head_PDLC149.py这些文件引用的是AreaLoss_PDLC！！！！！！！！！！
'''
别tm改这个了！！！这个没用!
改下面的AreaLoss_PDLC
'''
from .area_loss_L1 import AreaLoss_L1

# from .area_loss_PDLC24 import AreaLoss_PDLC
# from .area_loss_PDLC181_ijcai import AreaLoss_PDLC #用于jicai的pdlc181 他跟area_loss_PDLC24一样都是alaha=0.5，beta=1
# from .area_loss_PDLC181 import AreaLoss_PDLC # 195
# from .area_loss_PDLC182 import AreaLoss_PDLC # 196
# from .area_loss_PDLC183 import AreaLoss_PDLC # 197
# from .area_loss_PDLC184 import AreaLoss_PDLC # 198
# from .area_loss_PDLC185 import AreaLoss_PDLC # 199
# from .area_loss_PDLC181_OPIXray import AreaLoss_PDLC
from .area_loss_PDLC24_OPIXray import AreaLoss_PDLC



__all__ = [
    'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy',
    'mask_cross_entropy', 'CrossEntropyLoss', 'sigmoid_focal_loss',
    'FocalLoss', 'smooth_l1_loss', 'SmoothL1Loss', 'balanced_l1_loss',
    'BalancedL1Loss', 'mse_loss', 'MSELoss', 'iou_loss', 'bounded_iou_loss',
    'IoULoss', 'BoundedIoULoss', 'GIoULoss', 'DIoULoss', 'CIoULoss', 'GHMC',
    'GHMR', 'reduce_loss', 'weight_reduce_loss', 'weighted_loss', 'L1Loss',
    'l1_loss', 'isr_p', 'carl_loss', 'AssociativeEmbeddingLoss',
    'GaussianFocalLoss', 'QualityFocalLoss', 'DistributionFocalLoss',
    'VarifocalLoss', 'KnowledgeDistillationKLDivLoss', 'SeesawLoss', 'DiceLoss'
    ,'AreaLoss','AreaLoss_L1'
]
