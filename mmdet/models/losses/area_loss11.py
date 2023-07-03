# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def mse_loss(pred, target):
    """Wrapper of mse loss."""
    return F.mse_loss(pred, target, reduction='none')

#利用@weight_loss能解决weight即positive的问题，然后avg_factor=‘none’进行sum即可
#问题一：target是否跟尺度有关，应该是有 但是现在全是零没法判断
#问题二：pred的尺寸没法判定，需要根据得到每次的步长进而进行计算area 这里可以根据dim=1的长度获取
#问题三：需要获得各类别的预测值
means=[15131., 10559.,  3592.,  2859., 15826., 11306., 16746.,  8510., 14498.,  1492., 81744., 12914., 23296.,  1400., 28223.]
stds=[ 3264.,  5439.,  1271.,  1717.,  4845.,  2899.,  6152.,  2679.,  6524.,   734., 36801.,  3526., 11786.,   372.,  9155.]
@weighted_loss
def area_loss(Pred_wh,Pred_label,true_w=320,strides=[32, 16, 8],
              means=[15131., 10559., 3592., 2859., 15826., 11306., 16746., 8510., 14498., 1492., 81744., 12914., 23296.,
                     1400., 28223.],
              stds=[ 3264.,  5439.,  1271.,  1717.,  4845.,  2899.,  6152.,  2679.,  6524.,   734., 36801.,  3526., 11786.,   372.,  9155.]

              ):
    # if true_w/(Pred_wh.size(1)/3)**0.5 in strides:
    #     stride=true_w/(Pred_wh.size(1)/3)**0.5
    # else:
    #     print(true_w/(Pred_wh.size(1)/3)**0.5)
    stride = true_w / (Pred_wh.size(1) / 3) ** 0.5
    # if stride not in strides:
    #     print(stride)
    wh=torch.chunk(Pred_wh,2,dim=-1)
    w=wh[0]
    h=wh[1]
    area=stride**2*w*h
    means=torch.tensor(means).cuda()
    stds=torch.tensor(stds).cuda()
    index=torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,13,14]).cuda()


    Pred_label=torch.index_select(Pred_label,-1,index)

    means=torch.index_select(means,-1,index)
    stds= torch.index_select(stds,-1,index)
    return ((Pred_label*(area-means)/(stds**2))**2).sum(dim=-1,keepdim=True)



@LOSSES.register_module()
class AreaLoss(nn.Module):
    """MSELoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, reduction='mean', loss_weight=1.0,true_w=320,strides=[],means=[],stds=[]):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.strides=strides
        self.means=means
        self.stds=stds
        self.true_w=true_w

    def forward(self,
                Pred_wh,
                Pred_label,


                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * area_loss(
            Pred_wh, Pred_label,weight,true_w=self.true_w,strides=self.strides,means=self.means,stds=self.stds,  reduction=reduction, avg_factor=avg_factor)
        return loss
