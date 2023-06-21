# _base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']
_base_ = [ '../_base_/default_runtime.py']

img_scale = (320, 320)  # height, width
# img_scale = (640, 640)  # height, width

# model settings
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    init_cfg = dict(type='Pretrained',checkpoint='D:\Projects\mmdetection\checkpoint\yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'),
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead_PDLC', num_classes=5, in_channels=128, feat_channels=128),
    train_cfg=dict(
        assigner=dict(
            type='SimOTAAssignerV2',
            center_radius=2.5)
    ),
# sim25+ pdlc181 +head149

    # train_cfg=dict(
    #     initial_epoch=4,
    #     initial_assigner=dict(type='ATSSAssigner', topk=9),
    #     assigner=dict(type='TaskAlignedAssigner', topk=13),
    #     alpha=1,
    #     beta=6,
    #     allowed_border=-1,
    #     pos_weight=-1,
    #     debug=False),

    # train_cfg=dict(
    #     assigner=dict(type='ATSSAssigner', topk=9),
    #     allowed_border=-1,
    #     pos_weight=-1,
    #     debug=False),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# dataset settings
# data_root = 'D:\Projects\data\PIXray_coco/'
# data_root = 'E:\Datasets/coco/'
# dataset_type = 'PixCocoDataset'
# dataset_type = 'CocoDataset'

data_root = 'D:\Projects\data\OPIXray_voc/'
dataset_type = 'VOCDataset'
CLASSES = ('Folding_Knife','Straight_Knife','Scissor','Utility_Knife','Multi-tool_Knife')
train_pipeline = [
    # dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    # dict(
    #     type='RandomAffine',
    #     scaling_ratio_range=(0.1, 2),
    #     border=(-img_scale[0] // 2, -img_scale[1] // 2)),
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
        ann_file=data_root + 'VOC2007/ImageSets/Main/train.txt',
        img_prefix=data_root +'VOC2007/',
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
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
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
        img_prefix=data_root  +'VOC2007/',
        pipeline=test_pipeline))

#voc
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type='RepeatDataset',
#         times=3,
#         dataset=dict(
#             type=dataset_type,
#             ann_file=[
#                 data_root + 'VOC2007/ImageSets/Main/trainval.txt',
#                 data_root + 'VOC2012/ImageSets/Main/trainval.txt'
#             ],
#             img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
#             pipeline=train_pipeline)),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
#         img_prefix=data_root + 'VOC2007/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
#         img_prefix=data_root + 'VOC2007/',
#         pipeline=test_pipeline))
# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.0025,
    momentum=0.9,
    weight_decay=1e-4,
    # nesterov=True,
    # paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.)
)
optimizer_config = dict(grad_clip=None)

max_epochs = 30
num_last_epochs = 15


resume_from = None
interval = 10

# learning policy
# lr_config = dict(
#     _delete_=True,
#     policy='YOLOX',
#     warmup='exp',
#     by_epoch=False,
#     warmup_by_epoch=True,
#     warmup_ratio=1,
#     warmup_iters=5,  # 5 epoch
#     num_last_epochs=num_last_epochs,
#     min_lr_ratio=0.05)

# runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
checkpoint_config = dict(interval=-1)  # -1代表不保存
# evaluation = dict(interval=1, metric=['bbox'])
last_itervals=15 # 最后面每轮进行评估的n个epoch
evaluation = dict(interval=5,  metric='mAP', dynamic_intervals=[(max_epochs-last_itervals,1)],save_best='auto')#voc用mAP,coco用bbox
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
# auto_scale_lr = dict(base_batch_size=24)
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
# checkpoint_config = dict(interval=interval)
# evaluation = dict(
#     save_best='auto',
#     # The evaluation interval is 'interval' when running epoch is
#     # less than ‘max_epochs - num_last_epochs’.
#     # The evaluation interval is 1 when running epoch is greater than
#     # or equal to ‘max_epochs - num_last_epochs’.
#     interval=interval,
#     dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
#     metric='bbox')
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
auto_scale_lr = dict(base_batch_size=16)
