_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/OPIXray_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# model settings
model = dict(
    type='RetinaNet',
init_cfg=dict(type='Pretrained',
                  checkpoint=r'D:\Projects\mmdetection\checkpoint\retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'),
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
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=5,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# 使用说明：1检查加载预训练模型是不是只加载backbone，2更换base中的数据类型，3加上下面的这段，4并改一下head中的种类数量
# 针对PADet统一更改更改 包括训练epoch，评估方式，保存方式，batch，image size
runner = dict(type='EpochBasedRunner', max_epochs=60)
checkpoint_config = dict(interval=-1)  # -1代表不保存
evaluation = dict(interval=5, metric='mAP',dynamic_intervals=[(45,1)],save_best='auto')
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4)
img_scale=(320,320)
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)