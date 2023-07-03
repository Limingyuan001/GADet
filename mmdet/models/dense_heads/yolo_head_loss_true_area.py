# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, bias_init_with_prob, constant_init, is_norm,
                      normal_init)
from mmcv.runner import force_fp32

from mmdet.core import (build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, images_to_levels,
                        multi_apply, multiclass_nms)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin


@HEADS.register_module()
class YOLOV3Head_true_area(BaseDenseHead, BBoxTestMixin):
    """YOLOV3Head Paper link: https://arxiv.org/abs/1804.02767.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (List[int]): Number of input channels per scale.
        out_channels (List[int]): The number of output channels per scale
            before the final 1x1 layer. Default: (1024, 512, 256).
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        featmap_strides (List[int]): The stride of each scale.
            Should be in descending order. Default: (32, 16, 8).
        one_hot_smoother (float): Set a non-zero value to enable label-smooth
            Default: 0.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        loss_cls (dict): Config of classification loss.
        loss_conf (dict): Config of confidence loss.
        loss_xy (dict): Config of xy coordinate loss.
        loss_wh (dict): Config of wh coordinate loss.
        train_cfg (dict): Training config of YOLOV3 head. Default: None.
        test_cfg (dict): Testing config of YOLOV3 head. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 out_channels=(1024, 512, 256),
                 anchor_generator=dict(
                     type='YOLOAnchorGenerator',
                     base_sizes=[[(116, 90), (156, 198), (373, 326)],
                                 [(30, 61), (62, 45), (59, 119)],
                                 [(10, 13), (16, 30), (33, 23)]],
                     strides=[32, 16, 8]),
                 bbox_coder=dict(type='YOLOBBoxCoder'),
                 featmap_strides=[32, 16, 8],
                 one_hot_smoother=0.,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_conf=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_xy=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_wh=dict(type='MSELoss', loss_weight=1.0),
                 loss_area=dict(type='AreaLoss',loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Normal', std=0.01,
                     override=dict(name='convs_pred'))):
        super(YOLOV3Head_true_area, self).__init__(init_cfg)
        # Check params
        assert (len(in_channels) == len(out_channels) == len(featmap_strides))

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            if hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

        self.one_hot_smoother = one_hot_smoother

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.prior_generator = build_prior_generator(anchor_generator)

        self.loss_cls = build_loss(loss_cls)
        self.loss_conf = build_loss(loss_conf)
        self.loss_xy = build_loss(loss_xy)
        self.loss_wh = build_loss(loss_wh)
        self.loss_area = build_loss(loss_area)
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        assert len(
            self.prior_generator.num_base_priors) == len(featmap_strides)
        self._init_layers()

    @property
    def anchor_generator(self):

        warnings.warn('DeprecationWarning: `anchor_generator` is deprecated, '
                      'please use "prior_generator" instead')
        return self.prior_generator

    @property
    def num_anchors(self):
        """
        Returns:
            int: Number of anchors on each point of feature map.
        """
        warnings.warn('DeprecationWarning: `num_anchors` is deprecated, '
                      'please use "num_base_priors" instead')
        return self.num_base_priors

    @property
    def num_levels(self):
        return len(self.featmap_strides)

    @property
    def num_attrib(self):
        """int: number of attributes in pred_map, bboxes (4) +
        objectness (1) + num_classes"""

        return 5 + self.num_classes

    def _init_layers(self):
        self.convs_bridge = nn.ModuleList()
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_bridge = ConvModule(
                self.in_channels[i],
                self.out_channels[i],
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            conv_pred = nn.Conv2d(self.out_channels[i],
                                  self.num_base_priors * self.num_attrib, 1)

            self.convs_bridge.append(conv_bridge)
            self.convs_pred.append(conv_pred)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)

        # Use prior in model initialization to improve stability
        for conv_pred, stride in zip(self.convs_pred, self.featmap_strides):
            bias = conv_pred.bias.reshape(self.num_base_priors, -1)
            # init objectness with prior of 8 objects per feature map
            # refer to https://github.com/ultralytics/yolov3
            nn.init.constant_(bias.data[:, 4],
                              bias_init_with_prob(8 / (608 / stride)**2))
            nn.init.constant_(bias.data[:, 5:], bias_init_with_prob(0.01))

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        assert len(feats) == self.num_levels
        pred_maps = []
        for i in range(self.num_levels):
            x = feats[i]
            x = self.convs_bridge[i](x)
            pred_map = self.convs_pred[i](x)
            pred_maps.append(pred_map)

        return tuple(pred_maps),

    @force_fp32(apply_to=('pred_maps', ))
    def get_bboxes(self,
                   pred_maps,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions. It has
        been accelerated since PR #5991.

        Args:
            pred_maps (list[Tensor]): Raw predictions for a batch of images.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(pred_maps) == self.num_levels
        cfg = self.test_cfg if cfg is None else cfg
        scale_factors = np.array(
            [img_meta['scale_factor'] for img_meta in img_metas])

        num_imgs = len(img_metas)
        featmap_sizes = [pred_map.shape[-2:] for pred_map in pred_maps]

        mlvl_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=pred_maps[0].device)
        flatten_preds = []
        flatten_strides = []
        for pred, stride in zip(pred_maps, self.featmap_strides):
            pred = pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    self.num_attrib)
            pred[..., :2].sigmoid_()
            flatten_preds.append(pred)
            flatten_strides.append(
                pred.new_tensor(stride).expand(pred.size(1)))

        flatten_preds = torch.cat(flatten_preds, dim=1)
        flatten_bbox_preds = flatten_preds[..., :4]
        flatten_objectness = flatten_preds[..., 4].sigmoid()
        flatten_cls_scores = flatten_preds[..., 5:].sigmoid()
        flatten_anchors = torch.cat(mlvl_anchors)
        flatten_strides = torch.cat(flatten_strides)
        flatten_bboxes = self.bbox_coder.decode(flatten_anchors,
                                                flatten_bbox_preds,
                                                flatten_strides.unsqueeze(-1))

        if with_nms and (flatten_objectness.size(0) == 0):
            return torch.zeros((0, 5)), torch.zeros((0, ))

        if rescale:
            flatten_bboxes /= flatten_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        # 对预测进行修正
        w = flatten_bboxes[..., 2] - flatten_bboxes[..., 0]
        h = flatten_bboxes[..., 3] - flatten_bboxes[..., 1]
        # area=w*h
        # area_=torch.unsqueeze(area,2)
        d = torch.sqrt(torch.pow(w, 2) + torch.pow(h, 2))
        d = torch.unsqueeze(d, 2)
        batch = flatten_cls_scores.size(0)
        boxes = flatten_cls_scores.size(1)
        classes = flatten_cls_scores.size(2)
        d = d.expand(batch, boxes, classes)
        softmax = nn.Softmax(dim=-1)
        # index_p = torch.argmax(softmax(pred_label), dim=-1, keepdim=True)
        # index_p = index_p.expand(batch, boxes, classes)
        # index_p_long = index_p.long()
        means = torch.tensor(
            [189., 196., 97., 78., 194., 160., 223., 139., 216., 71., 594., 168., 291., 54., 283.]).cuda()
        means = means.expand(batch, boxes, classes)
        d_and_means = torch.cat((d.unsqueeze(-1), means.unsqueeze(-1)), dim=-1)
        similar = torch.min(d_and_means, dim=-1)[0] / torch.max(d_and_means, dim=-1)[0]
        sum = torch.sum(torch.bernoulli(similar) * torch.abs(flatten_cls_scores), dim=-1).unsqueeze(-1)+1
        flatten_cls_scores = flatten_cls_scores * torch.bernoulli(similar) / sum
        # sum = torch.sum(torch.exp(similar) * torch.abs(flatten_cls_scores), dim=-1).unsqueeze(-1)
        # flatten_cls_scores = (flatten_cls_scores * torch.exp(similar) * torch.sum(flatten_cls_scores,dim=-1).unsqueeze(-1)) / sum
        # flatten_cls_scores = (flatten_cls_scores * torch.exp(similar) ) / sum

        flatten_cls_scores = F.softmax(flatten_cls_scores,dim=-1)
        #至此flattern_cls_scores修正完毕
        padding = flatten_bboxes.new_zeros(num_imgs, flatten_bboxes.shape[1],
                                           1)
        flatten_cls_scores = torch.cat([flatten_cls_scores, padding], dim=-1)

        det_results = []
        for (bboxes, scores, objectness) in zip(flatten_bboxes,
                                                flatten_cls_scores,
                                                flatten_objectness):
            # Filtering out all predictions with conf < conf_thr
            conf_thr = cfg.get('conf_thr', -1)
            if conf_thr > 0:
                conf_inds = objectness >= conf_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=objectness)
            det_results.append(tuple([det_bboxes, det_labels]))
        return det_results

    @force_fp32(apply_to=('pred_maps', ))
    def loss(self,
             pred_maps,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            pred_maps (list[Tensor]): Prediction map for each scale level,
                shape (N, num_anchors * num_attrib, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(img_metas)
        device = pred_maps[0][0].device

        featmap_sizes = [
            pred_maps[i].shape[-2:] for i in range(self.num_levels)
        ]
        mlvl_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)
        anchor_list = [mlvl_anchors for _ in range(num_imgs)]

        responsible_flag_list = []
        for img_id in range(len(img_metas)):
            responsible_flag_list.append(
                self.prior_generator.responsible_flags(featmap_sizes,
                                                       gt_bboxes[img_id],
                                                       device))

        target_maps_list, neg_maps_list = self.get_targets(
            anchor_list, responsible_flag_list, gt_bboxes, gt_labels)
        # losses_cls, losses_conf, losses_xy, losses_wh = multi_apply(
        #     self.loss_single, pred_maps, target_maps_list, neg_maps_list)
        losses_cls, losses_conf, losses_xy, losses_wh ,losses_area= multi_apply(
            self.loss_single, pred_maps, target_maps_list, neg_maps_list,img_metas,featmap_sizes )

        return dict(
            loss_cls=losses_cls,
            loss_conf=losses_conf,
            loss_xy=losses_xy,
            loss_wh=losses_wh,
            loss_area=losses_area
        )

    def loss_single(self, pred_map, target_map, neg_map,img_metas,featmap_sizes):
        """Compute loss of a single image from a batch.

        Args:
            pred_map (Tensor): Raw predictions for a single level.
            target_map (Tensor): The Ground-Truth target for a single level.
            neg_map (Tensor): The negative masks for a single level.

        Returns:
            tuple:
                loss_cls (Tensor): Classification loss.
                loss_conf (Tensor): Confidence loss.
                loss_xy (Tensor): Regression loss of x, y coordinate.
                loss_wh (Tensor): Regression loss of w, h coordinate.
        """

        num_imgs = len(pred_map)
        pred_map = pred_map.permute(0, 2, 3,
                                    1).reshape(num_imgs, -1, self.num_attrib)
        neg_mask = neg_map.float()
        pos_mask = target_map[..., 4]
        pos_and_neg_mask = neg_mask + pos_mask
        pos_mask = pos_mask.unsqueeze(dim=-1)
        if torch.max(pos_and_neg_mask) > 1.:
            warnings.warn('There is overlap between pos and neg sample.')
            pos_and_neg_mask = pos_and_neg_mask.clamp(min=0., max=1.)

        pred_xy = pred_map[..., :2]
        pred_wh = pred_map[..., 2:4]
        pred_conf = pred_map[..., 4]
        pred_label = pred_map[..., 5:]

        target_xy = target_map[..., :2]
        target_wh = target_map[..., 2:4]
        target_conf = target_map[..., 4]
        target_label = target_map[..., 5:]

        loss_cls = self.loss_cls(pred_label, target_label, weight=pos_mask)
        loss_conf = self.loss_conf(
            pred_conf, target_conf, weight=pos_and_neg_mask)
        loss_xy = self.loss_xy(pred_xy, target_xy, weight=pos_mask)
        loss_wh = self.loss_wh(pred_wh, target_wh, weight=pos_mask)

        #这里主要是想办法获取到真实的面积，pred_wh需要解码才有意义之前的面积有问题
        # scale_factors = np.array(
        #     [img_meta['scale_factor'] for img_meta in img_metas])

        scale_factors=img_metas['scale_factor']
        # featmap_size = pred_map.shape[-2:]
        #确定featmap_sizes

        if featmap_sizes[-1]==10:
            idx=0
            # print(featmap_sizes[-1])
        if featmap_sizes[-1] == 20:
            idx=1
            # print(featmap_sizes[-1])
        if featmap_sizes[-1] == 40:
            idx=2
            # print(featmap_sizes[-1])
        if featmap_sizes[-1] not in [10,20,40]:
            idx = 2
            print(featmap_sizes[-1])
        mlvl_anchors = self.prior_generator.grid_priors_single_layer(
            featmap_sizes, level_idx=idx,device=torch.device('cuda:0'))
        stride = 320. / (pred_wh.size(1) / 3) ** 0.5
        # pred_map_sig=pred_map
        pred_map[..., :2].sigmoid()
        # pred_map_sig=torch.cat((pred_map[..., :2].sigmoid(),pred_map[...,2:]),dim=-1)
        # 两个问题，1 sigmoid没起作用 2 没使用permute（0231）和reshape
        flatten_strides = []
        flatten_strides.append(
            pred_map.new_tensor(stride).expand(pred_map.size(1)))
        flatten_strides = torch.cat(flatten_strides)#这个地方可能有问题，对照263,应该有180 720 2800个32 16 8
        # flatten_preds = pred_map[:4]
        flatten_bbox_preds = pred_map[..., :4]
        flatten_bboxes = self.bbox_coder.decode(mlvl_anchors,
                                                flatten_bbox_preds,
                                                flatten_strides.unsqueeze(-1))
        flatten_bboxes = flatten_bboxes/flatten_bboxes.new_tensor(
            scale_factors)
        #flatten_bboxes是真实的的x1，y1x2y2
        # loss_area= self.loss_area(flatten_bboxes,pred_label,weight=pos_mask)

        # 只能同时用一个形参传两个值
        box_label=(flatten_bboxes,target_label,target_wh,pred_wh)
        loss_area= self.loss_area(box_label,pred_label, weight=pos_mask)

        # # 对分类置信度进行修正
        # w = flatten_bboxes[..., 2] - flatten_bboxes[..., 0]
        # h = flatten_bboxes[..., 3] - flatten_bboxes[..., 1]
        # # area=w*h
        # # area_=torch.unsqueeze(area,2)
        # d = torch.sqrt(torch.pow(w, 2) + torch.pow(h, 2))
        # d = torch.unsqueeze(d, 2)
        #
        # batch = pred_label.size(0)
        # boxes = pred_label.size(1)
        # classes = pred_label.size(2)
        # d = d.expand(batch, boxes, classes)
        # softmax = nn.Softmax(dim=-1)
        # # index_p = torch.argmax(softmax(pred_label), dim=-1, keepdim=True)
        # # index_p = index_p.expand(batch, boxes, classes)
        # # index_p_long = index_p.long()
        # means = torch.tensor(
        #     [189., 196., 97., 78., 194., 160., 223., 139., 216., 71., 594., 168., 291., 54., 283.]).cuda()
        # means=means.expand(batch, boxes, classes)
        # d_and_means=torch.cat((d.unsqueeze(-1),means.unsqueeze(-1)),dim=-1)
        # similar=torch.min(d_and_means,dim=-1)[0]/torch.max(d_and_means,dim=-1)[0]
        # sum=torch.sum(torch.bernoulli(similar)*torch.abs(pred_label),dim=-1).unsqueeze(-1)+1
        # pred_label_new=pred_label*torch.bernoulli(similar)/sum
        # # sum=torch.sum(torch.exp(similar)*torch.abs(pred_label),dim=-1).unsqueeze(-1)
        # # pred_label_new=((pred_label*torch.exp(similar))*torch.sum(pred_label,dim=-1).unsqueeze(-1))/sum
        # # pred_label_new=(pred_label*torch.exp(similar))/sum
        #
        # pred_label_new=F.softmax(pred_label_new,dim=-1)
        #
        # #pred_label_new修正完毕
        #
        # # loss_cls = self.loss_cls(pred_label_new, target_label, weight=pos_mask)#这里原版应该用pred_label对应lossarea24的代码
        return loss_cls, loss_conf, loss_xy, loss_wh,loss_area
    # def grid_priors(self, featmap_sizes, dtype=torch.float32, device='cuda'):
    #     """Generate grid anchors in multiple feature levels.
    #
    #     Args:
    #         featmap_sizes (list[tuple]): List of feature map sizes in
    #             multiple feature levels.
    #         dtype (:obj:`torch.dtype`): Dtype of priors.
    #             Default: torch.float32.
    #         device (str): The device where the anchors will be put on.
    #
    #     Return:
    #         list[torch.Tensor]: Anchors in multiple feature levels. \
    #             The sizes of each tensor should be [N, 4], where \
    #             N = width * height * num_base_anchors, width and height \
    #             are the sizes of the corresponding feature level, \
    #             num_base_anchors is the number of anchors for that level.
    #     """
    #     assert self.num_levels == len(featmap_sizes)
    #     multi_level_anchors = []
    #     for i in range(self.num_levels):
    #         anchors = self.single_level_grid_priors(
    #             featmap_sizes[i], level_idx=i, dtype=dtype, device=device)
    #         multi_level_anchors.append(anchors)
    #     return multi_level_anchors
    def get_targets(self, anchor_list, responsible_flag_list, gt_bboxes_list,
                    gt_labels_list):
        """Compute target maps for anchors in multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_total_anchors, 4).
            responsible_flag_list (list[list[Tensor]]): Multi level responsible
                flags of each image. Each element is a tensor of shape
                (num_total_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.

        Returns:
            tuple: Usually returns a tuple containing learning targets.
                - target_map_list (list[Tensor]): Target map of each level.
                - neg_map_list (list[Tensor]): Negative map of each level.
        """
        num_imgs = len(anchor_list)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        results = multi_apply(self._get_targets_single, anchor_list,
                              responsible_flag_list, gt_bboxes_list,
                              gt_labels_list)

        all_target_maps, all_neg_maps = results
        assert num_imgs == len(all_target_maps) == len(all_neg_maps)
        target_maps_list = images_to_levels(all_target_maps, num_level_anchors)
        neg_maps_list = images_to_levels(all_neg_maps, num_level_anchors)

        return target_maps_list, neg_maps_list

    def _get_targets_single(self, anchors, responsible_flags, gt_bboxes,
                            gt_labels):
        """Generate matching bounding box prior and converted GT.

        Args:
            anchors (list[Tensor]): Multi-level anchors of the image.
            responsible_flags (list[Tensor]): Multi-level responsible flags of
                anchors
            gt_bboxes (Tensor): Ground truth bboxes of single image.
            gt_labels (Tensor): Ground truth labels of single image.

        Returns:
            tuple:
                target_map (Tensor): Predication target map of each
                    scale level, shape (num_total_anchors,
                    5+num_classes)
                neg_map (Tensor): Negative map of each scale level,
                    shape (num_total_anchors,)
        """

        anchor_strides = []
        for i in range(len(anchors)):
            anchor_strides.append(
                torch.tensor(self.featmap_strides[i],
                             device=gt_bboxes.device).repeat(len(anchors[i])))
        concat_anchors = torch.cat(anchors)
        concat_responsible_flags = torch.cat(responsible_flags)

        anchor_strides = torch.cat(anchor_strides)
        assert len(anchor_strides) == len(concat_anchors) == \
               len(concat_responsible_flags)
        assign_result = self.assigner.assign(concat_anchors,
                                             concat_responsible_flags,
                                             gt_bboxes)
        sampling_result = self.sampler.sample(assign_result, concat_anchors,
                                              gt_bboxes)

        target_map = concat_anchors.new_zeros(
            concat_anchors.size(0), self.num_attrib)

        target_map[sampling_result.pos_inds, :4] = self.bbox_coder.encode(
            sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes,
            anchor_strides[sampling_result.pos_inds])

        target_map[sampling_result.pos_inds, 4] = 1

        gt_labels_one_hot = F.one_hot(
            gt_labels, num_classes=self.num_classes).float()
        if self.one_hot_smoother != 0:  # label smooth
            gt_labels_one_hot = gt_labels_one_hot * (
                1 - self.one_hot_smoother
            ) + self.one_hot_smoother / self.num_classes
        target_map[sampling_result.pos_inds, 5:] = gt_labels_one_hot[
            sampling_result.pos_assigned_gt_inds]

        neg_map = concat_anchors.new_zeros(
            concat_anchors.size(0), dtype=torch.uint8)
        neg_map[sampling_result.neg_inds] = 1

        return target_map, neg_map

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)

    @force_fp32(apply_to=('pred_maps'))
    def onnx_export(self, pred_maps, img_metas, with_nms=True):
        num_levels = len(pred_maps)
        pred_maps_list = [pred_maps[i].detach() for i in range(num_levels)]

        cfg = self.test_cfg
        assert len(pred_maps_list) == self.num_levels

        device = pred_maps_list[0].device
        batch_size = pred_maps_list[0].shape[0]

        featmap_sizes = [
            pred_maps_list[i].shape[-2:] for i in range(self.num_levels)
        ]
        mlvl_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)

        multi_lvl_bboxes = []
        multi_lvl_cls_scores = []
        multi_lvl_conf_scores = []
        for i in range(self.num_levels):
            # get some key info for current scale
            pred_map = pred_maps_list[i]
            stride = self.featmap_strides[i]
            # (b,h, w, num_anchors*num_attrib) ->
            # (b,h*w*num_anchors, num_attrib)
            pred_map = pred_map.permute(0, 2, 3,
                                        1).reshape(batch_size, -1,
                                                   self.num_attrib)
            # Inplace operation like
            # ```pred_map[..., :2] = \torch.sigmoid(pred_map[..., :2])```
            # would create constant tensor when exporting to onnx
            pred_map_conf = torch.sigmoid(pred_map[..., :2])
            pred_map_rest = pred_map[..., 2:]
            pred_map = torch.cat([pred_map_conf, pred_map_rest], dim=-1)
            pred_map_boxes = pred_map[..., :4]
            multi_lvl_anchor = mlvl_anchors[i]
            multi_lvl_anchor = multi_lvl_anchor.expand_as(pred_map_boxes)
            bbox_pred = self.bbox_coder.decode(multi_lvl_anchor,
                                               pred_map_boxes, stride)
            # conf and cls
            conf_pred = torch.sigmoid(pred_map[..., 4])
            cls_pred = torch.sigmoid(pred_map[..., 5:]).view(
                batch_size, -1, self.num_classes)  # Cls pred one-hot.

            # Get top-k prediction
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                _, topk_inds = conf_pred.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
                transformed_inds = (
                    bbox_pred.shape[1] * batch_inds + topk_inds)
                bbox_pred = bbox_pred.reshape(-1,
                                              4)[transformed_inds, :].reshape(
                                                  batch_size, -1, 4)
                cls_pred = cls_pred.reshape(
                    -1, self.num_classes)[transformed_inds, :].reshape(
                        batch_size, -1, self.num_classes)
                conf_pred = conf_pred.reshape(-1, 1)[transformed_inds].reshape(
                    batch_size, -1)

            # Save the result of current scale
            multi_lvl_bboxes.append(bbox_pred)
            multi_lvl_cls_scores.append(cls_pred)
            multi_lvl_conf_scores.append(conf_pred)

        # Merge the results of different scales together
        batch_mlvl_bboxes = torch.cat(multi_lvl_bboxes, dim=1)
        batch_mlvl_scores = torch.cat(multi_lvl_cls_scores, dim=1)
        batch_mlvl_conf_scores = torch.cat(multi_lvl_conf_scores, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        from mmdet.core.export import add_dummy_nms_for_onnx
        conf_thr = cfg.get('conf_thr', -1)
        score_thr = cfg.get('score_thr', -1)
        # follow original pipeline of YOLOv3
        if conf_thr > 0:
            mask = (batch_mlvl_conf_scores >= conf_thr).float()
            batch_mlvl_conf_scores *= mask
        if score_thr > 0:
            mask = (batch_mlvl_scores > score_thr).float()
            batch_mlvl_scores *= mask
        batch_mlvl_conf_scores = batch_mlvl_conf_scores.unsqueeze(2).expand_as(
            batch_mlvl_scores)
        batch_mlvl_scores = batch_mlvl_scores * batch_mlvl_conf_scores
        if with_nms:
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            # keep aligned with original pipeline, improve
            # mAP by 1% for YOLOv3 in ONNX
            score_threshold = 0
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(
                batch_mlvl_bboxes,
                batch_mlvl_scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                nms_pre,
                cfg.max_per_img,
            )
        else:
            return batch_mlvl_bboxes, batch_mlvl_scores
