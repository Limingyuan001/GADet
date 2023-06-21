_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='GFL',
    init_cfg=dict(type='Pretrained',
                  checkpoint='D:\Projects\mmdetection\checkpoint\gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth'),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        # frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFLHead',
        num_classes=5,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# dataset settings
# dataset_type = 'VOCDataset'
# data_root = 'data/VOCdevkit/'
data_root = 'D:\Projects\data\OPIXray_voc/'
dataset_type = 'VOCDataset'
CLASSES = ('Folding_Knife','Straight_Knife','Scissor','Utility_Knife','Multi-tool_Knife')
img_scale=(320,320)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(


            classes=CLASSES,


            type=dataset_type,
            ann_file=data_root + 'VOC2007/ImageSets/Main/train.txt',
            img_prefix=data_root + 'VOC2007/',
            pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root +'VOC2007/' ,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root +'VOC2007/' ,
        pipeline=test_pipeline))
# evaluation = dict(interval=1, metric='mAP')

# 使用说明：1检查加载预训练模型是不是只加载backbone，2更换base中的数据类型，3加上下面的这段，4并改一下head中的种类数量
# 针对PADet统一更改更改 包括训练epoch，评估方式，保存方式，batch，image size
runner = dict(type='EpochBasedRunner', max_epochs=60)
checkpoint_config = dict(interval=-1)  # -1代表不保存
evaluation = dict(interval=5, metric='mAP',save_best='auto')
# data = dict(
#     samples_per_gpu=16,
#     workers_per_gpu=4)
img_scale=(320,320)
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
