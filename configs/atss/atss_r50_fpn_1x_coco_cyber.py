_base_ = [
    # '../_base_/datasets/coco_detection.py',
    # '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
img_scale = (320, 320)  # height, width
# img_scale = (640, 640)  # height, width

model = dict(
    type='ATSS',
    init_cfg=dict(type='Pretrained',
                  checkpoint=r'D:\Projects\mmdetection\checkpoint\atss_r50_fpn_1x_coco_20200209-985f7bd0.pth'),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
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
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
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
# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# dataset settings
dataset_type = 'CocoDataset'
CLASSES = ('Gun', 'Knife', 'Lighter', 'Battery', 'Pliers', 'Scissors', 'Wrench', 'Hammer', 'Screwdriver', 'Dart', 'Bat', 'Fireworks', 'Saw_blade',
           'Razor_blade', 'Pressure_vessel')
data_root = 'D:\Projects\data\PIXray_coco/'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile', to_float32=True),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='Expand',
#         mean=img_norm_cfg['mean'],
#         to_rgb=img_norm_cfg['to_rgb'],
#         ratio_range=(1, 2)),
#     dict(
#         type='MinIoURandomCrop',
#         min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
#         min_crop_size=0.3),
#     dict(type='Resize', img_scale=[(320, 320)], keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(320, 320),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img'])
#         ])
# ]
# data = dict(
#     samples_per_gpu=24,
#     workers_per_gpu=1,
#
#     train=dict(
#         type=dataset_type,
#         classes=CLASSES,
#         ann_file=data_root + 'annotations/pixray_train.json',
#         img_prefix=data_root + 'train/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         classes=CLASSES,
#         ann_file=data_root + 'annotations/pixray_test.json',
#         img_prefix=data_root + 'test/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         classes=CLASSES,
#         ann_file=data_root + 'annotations/pixray_test.json',
#         img_prefix=data_root + 'test/',
#         pipeline=test_pipeline))

train_pipeline = [
    # dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    # dict(type='Resize', img_scale=img_scale, keep_ratio=False),#keep_ratio=False 代表是否将图片按照原始尺寸进行缩放，还是严格按照size进行改变

    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + 'annotations/pixray_train.json',
        img_prefix=data_root + 'train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='Resize', keep_ratio=False),

            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=6,
    persistent_workers=True,
    train=train_dataset,
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
# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
# optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001) #mmclassification中的学习率
# # optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001) #为了lossarea
#
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(grad_clip=None)
num_last_epochs = 15
resume_from = None
interval = 10
custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
optimizer = dict(
    type='SGD',
    # lr=0.0025,
    lr=0.01,

    momentum=0.9,
    weight_decay=1e-4,
    # nesterov=True,
    # paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.)
)
# optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=2000,  # same as burn-in in darknet
#     warmup_ratio=0.1,
#     step=[218, 246])

lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='exp',
    warmup_ratio=0.1,
    warmup_iters=5 * 169,
    warmup_by_epoch=False)
# runtime settings
max_epochs=60
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
checkpoint_config = dict(interval=-1)  # -1代表不保存
# evaluation = dict(interval=1, metric=['bbox'])
last_itervals=5 # 最后面每轮进行评估的n个epoch
evaluation = dict(interval=1,  metric='bbox', dynamic_intervals=[(max_epochs-last_itervals,1)],save_best='auto')#这里的iterval指的就是epoch
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=24)

# 加速训练
cudnn_benchmark=True #CNN (卷积神经网络) 特有的加速方式
pin_memory=True  #但是在这里加没用 我直接取datasets/builder。py改了dataloader中的pin_memoray