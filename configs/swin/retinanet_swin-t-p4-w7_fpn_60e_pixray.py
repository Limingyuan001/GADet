_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/PIXray_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
pretrained = r'D:\Projects\mmdetection\checkpoint\swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768], start_level=0, num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=15)
    )
# 使用说明：1检查加载预训练模型是不是只加载backbone，2更换base中的数据类型，3加上下面的这段，4并改一下head中的种类数量
# 针对PADet统一更改更改 包括训练epoch，评估方式，保存方式，batch，image size
runner = dict(type='EpochBasedRunner', max_epochs=60)
checkpoint_config = dict(interval=-1)  # -1代表不保存
evaluation = dict(interval=1, metric='bbox',save_best='auto')
# data = dict(
#     samples_per_gpu=16,
#     workers_per_gpu=4)
img_scale=(320,320)
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
