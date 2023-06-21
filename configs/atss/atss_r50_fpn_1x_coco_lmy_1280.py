_base_ = [
 '../_base_/default_runtime.py'
]
model = dict(
    type='ATSS',
    init_cfg=dict(type='Pretrained',
                  checkpoint=r'D:\Projects\mmdetection\checkpoint\atss_r50_fpn_1x_coco_20200209-985f7bd0.pth'),
    # init_cfg=dict(type='Pretrained',
    #               checkpoint=r'D:\Projects\mmdetection\checkpoint\atss\053\best_bbox_mAP_epoch_28.pth'),

    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        # frozen_stages=4,


        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        # 放在backbonr和neck处预训练一下

        # init_cfg=dict(type='Pretrained',
        #            checkpoint=r'D:\Projects\mmdetection\checkpoint\atss_r50_fpn_1x_coco_20200209-985f7bd0.pth'),

        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
         ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        # 放在backbonr和neck处预训练一下
        # init_cfg=dict(type='Pretrained',
        #            checkpoint=r'D:\Projects\mmdetection\checkpoint\atss_r50_fpn_1x_coco_20200209-985f7bd0.pth'),

    ),
    # atss原配
    bbox_head=dict(
        type='ATSSHead',

        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
    #    loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # 放head处预训练一下
        # init_cfg = dict(type='Pretrained',
        #                 checkpoint=r'D:\Projects\mmdetection\checkpoint\atss_r50_fpn_1x_coco_20200209-985f7bd0.pth'),
        ),

    # fcos
    # bbox_head=dict(
    #         type='FCOSHead',
    #         num_classes=15,
    #         in_channels=256,
    #         stacked_convs=4,
    #         feat_channels=256,
    #         strides=[8, 16, 32, 64, 128],
    #         loss_cls=dict(
    #             type='FocalLoss',
    #             use_sigmoid=True,
    #             gamma=2.0,
    #             alpha=0.25,
    #             loss_weight=1.0),
    #         loss_bbox=dict(type='IoULoss', loss_weight=1.0),
    #         loss_centerness=dict(
    #             type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    # ATSS原配
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),

    # fcos
    # train_cfg=dict(
    #     assigner=dict(
    #         type='MaxIoUAssigner',
    #         pos_iou_thr=0.5,
    #         neg_iou_thr=0.4,
    #         min_pos_iou=0,
    #         ignore_iof_thr=-1),
    #     allowed_border=-1,
    #     pos_weight=-1,
    #     debug=False),
    # TOOD原配
    # train_cfg=dict(
    #     initial_epoch=4,
    #     initial_assigner=dict(type='ATSSAssigner', topk=9),
    #     assigner=dict(type='TaskAlignedAssigner', topk=13),
    #     alpha=1,
    #     beta=6,
    #     allowed_border=-1,
    #     pos_weight=-1,
    #     debug=False),
    # simOTA
    # train_cfg=dict(
    #     assigner=dict(
    #         type='SimOTAAssigner',
    #         center_radius=2.5)
    # ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))


# dataset settings
data_root = 'D:\Projects\data\PIXray_coco/'
# data_root = 'E:\Datasets/coco'
dataset_type = 'CocoDataset'
img_scale=(448,1000)#由于取消resize所以随便写了
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=img_scale, keep_ratio=True),
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




CLASSES = ('Gun', 'Knife', 'Lighter', 'Battery', 'Pliers', 'Scissors', 'Wrench', 'Hammer', 'Screwdriver', 'Dart', 'Bat', 'Fireworks', 'Saw_blade',
           'Razor_blade', 'Pressure_vessel')
data = dict(
    samples_per_gpu=24,
    workers_per_gpu=4,
    # persistent_workers=True,
    train=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + 'annotations/pixray_train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + 'annotations/pixray_test.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + 'annotations/pixray_test.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline))

# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer = dict(
    type='SGD',
    lr=0.0025,
    momentum=0.9,
    weight_decay=1e-4,
    # nesterov=True,
    # paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.)
)
optimizer_config = dict(grad_clip=None)

max_epochs = 60
# max_epochs = 30

num_last_epochs = 15
resume_from = None
interval = 10
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
checkpoint_config = dict(interval=-1)  # -1代表不保存
# evaluation = dict(interval=1, metric=['bbox'])
last_itervals=5 # 最后面每轮进行评估的n个epoch
evaluation = dict(interval=1,  metric='bbox', dynamic_intervals=[(max_epochs-last_itervals,1)],save_best='auto')#这里的iterval指的就是epoch
log_config = dict(interval=50)
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='exp',
    warmup_ratio=0.1,
    warmup_iters=5 * 169,
    warmup_by_epoch=False)
# evaluation = dict(interval=1, metric='mAP_dual',save_best='mAP_dual')#这里的iterval指的就是epoch

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=24)