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
index=torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]).cuda()
means=[15131., 10559.,  3592.,  2859., 15826., 11306., 16746.,  8510., 14498.,  1492., 81744., 12914., 23296.,  1400., 28223.]
stds=[ 3264.,  5439.,  1271.,  1717.,  4845.,  2899.,  6152.,  2679.,  6524.,   734., 36801.,  3526., 11786.,   372.,  9155.]
means_d=[189., 196.,  97.,  78., 194., 160., 223., 139., 216.,  71., 594., 168., 291.,  54., 283.]
stds_d=[22., 23.,  8., 18., 18., 17., 24., 21., 31.,  8., 13., 19., 24.,  6., 20.]

softmax=nn.Softmax(dim=-1)
def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss
@weighted_loss
def area_loss(box_label,Pred_label,true_w=320,strides=[32, 16, 8],
                means=[189., 196.,  97.,  78., 194., 160., 223., 139., 216.,  71., 594., 168., 291.,  54., 283.],
              stds=[22., 23.,  8., 18., 18., 17., 24., 21., 31.,  8., 13., 19., 24.,  6., 20.]
             ):
    # with torch.autograd.set_detect_anomaly(True):
    # if true_w/(Pred_wh.size(1)/3)**0.5 in strides:
    #     stride=true_w/(Pred_wh.size(1)/3)**0.5
    # else:
    #     print(true_w/(Pred_wh.size(1)/3)**0.5)
    # stride = true_w / (Pred_wh.size(1) / 3) ** 0.5
    # # if stride not in strides:
    # #     print(stride)
    # wh=torch.chunk(Pred_wh,2,dim=-1)
    # w=wh[0]
    # h=wh[1]
    # area=stride**2*w*h
    # means=torch.tensor(means).cuda()
    # stds=torch.tensor(stds).cuda()
    means = torch.tensor(
        [189., 196., 97., 78., 194., 160., 223., 139., 216., 71., 594., 168., 291., 54., 283.]).cuda()
    stds = torch.tensor([22., 23., 8., 18., 18., 17., 24., 21., 31., 8., 13., 19., 24., 6., 20.]).cuda()
    flatten_bboxes = box_label[0]
    P_gt = box_label[1]
    w = flatten_bboxes[..., 2] - flatten_bboxes[..., 0]
    h = flatten_bboxes[..., 3] - flatten_bboxes[..., 1]
    # area=w*h
    # area_=torch.unsqueeze(area,2)
    d = torch.sqrt(torch.pow(w, 2) + torch.pow(h, 2))
    d = torch.unsqueeze(torch.unsqueeze(d, 0), 2)
    Pred_label = torch.index_select(Pred_label, -1, index)
    means = torch.index_select(means, -1, index)
    stds = torch.index_select(stds, -1, index)
    eps = 0.000000001
    Pred_label = torch.unsqueeze(Pred_label, 0)
    batch = Pred_label.size(0)
    boxes = Pred_label.size(1)
    classes = Pred_label.size(2)
    index_p = torch.argmax(softmax(Pred_label), dim=-1, keepdim=True)
    index_p = index_p.expand(batch, boxes, classes)
    index_p_long = index_p.long()
    alpha=1
    beta=2
    # mask_p=torch.zeros_like(Pred_label).to(torch.bool)

    # for i in range(batch):
    #     for j in range(boxes):
    #
    #
    #         mask_p[i,j,index_p[i,j]]=True

    # mask_p=softmax(Pred_label)>0.5
    p_diff = softmax(Pred_label) - P_gt
    d_diff=(torch.pow(torch.log(d),alpha)-torch.pow(torch.log(means),alpha))/stds
    # loss_p=torch.pow(p_diff,2)
    d_diff=torch.gather(d_diff,2,index_p_long)[...,0].unsqueeze(2)
    loss_d= torch.abs(d_diff)**beta
    loss = loss_d
    # return (((torch.sigmoid(Pred_label)*torch.abs(area_-means)/(stds))**2)).sum(dim=-1,keepdim=True)
    # return ((torch.sigmoid(Pred_label)*((torch.sqrt(torch.log(torch.abs(area_)+1+eps))-torch.sqrt(torch.log(torch.abs(means)+1+eps)))**2)/(stds))).sum(dim=-1,keepdim=True)
    return loss




@LOSSES.register_module()
class AreaLoss_PDLC(nn.Module):
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
