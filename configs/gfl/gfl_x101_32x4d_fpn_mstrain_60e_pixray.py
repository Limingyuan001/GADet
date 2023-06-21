_base_ = ['./gfl_r50_fpn_mstrain_2x_coco.py']
model = dict(
    type='GFL',
    init_cfg=dict(type='Pretrained',
                  checkpoint='D:\Projects\mmdetection\checkpoint\gfl_x101_32x4d_fpn_mstrain_2x_coco_20200630_102002-50c1ffdb.pth'),
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        # frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        # init_cfg=dict(
        #     type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')
    ),
bbox_head = dict(
    type='GFLHead',
    num_classes=15)
)



# dataset settings
data_root = 'D:\Projects\data\PIXray_coco/'
# data_root = 'E:\Datasets/coco'
dataset_type = 'CocoDataset'
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
CLASSES = ('Gun', 'Knife', 'Lighter', 'Battery', 'Pliers', 'Scissors', 'Wrench', 'Hammer', 'Screwdriver', 'Dart', 'Bat', 'Fireworks', 'Saw_blade',
           'Razor_blade', 'Pressure_vessel')
data = dict(
    samples_per_gpu=16,
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
# 使用说明：1检查加载预训练模型是不是只加载backbone，2更换base中的数据类型，3加上下面的这段，4并改一下head中的种类数量
# 针对PADet统一更改更改 包括训练epoch，评估方式，保存方式，batch，image size
runner = dict(type='EpochBasedRunner', max_epochs=60)
checkpoint_config = dict(interval=-1)  # -1代表不保存
evaluation = dict(interval=5, metric='bbox',save_best='auto')
# data = dict(
#     samples_per_gpu=16,
#     workers_per_gpu=4)
img_scale=(320,320)
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
