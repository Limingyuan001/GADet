_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/PIXray_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
init_cfg=dict(type='Pretrained',
                  checkpoint=r'D:\Projects\mmdetection\checkpoint\faster_rcnn_x101_32x4d_fpn_1x_coco_20200203-cff10310.pth'),
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        # frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        # init_cfg=dict(
        #     type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')
    ),
roi_head=dict(
    bbox_head=dict(

        num_classes=15))
)

# 使用说明：1检查加载预训练模型是不是只加载backbone，2更换base中的数据类型，3加上下面的这段，4并改一下head中的种类数量
# 针对PADet统一更改更改 包括训练epoch，评估方式，保存方式，batch，image size
runner = dict(type='EpochBasedRunner', max_epochs=60)
checkpoint_config = dict(interval=-1)  # -1代表不保存
evaluation = dict(interval=5, metric='bbox',save_best='auto')
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4)
img_scale=(320,320)
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
