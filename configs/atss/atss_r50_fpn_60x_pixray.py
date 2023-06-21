_base_ = [
    './atss_r50_fpn_1x_coco.py',
    '../_base_/datasets/PIXray_detection.py',
    ]


# 针对PADet统一更改更改 包括训练epoch，评估方式，保存方式，batch，image size
runner = dict(type='EpochBasedRunner', max_epochs=60)
checkpoint_config = dict(interval=-1)  # -1代表不保存
evaluation = dict(interval=1, metric='bbox')
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4)
img_scale=(320,320)
model = dict(
    bbox_head=dict(
        num_classes=80))
# 跑不起来，好像是继承除了问题，只能使用primary的形式，更换base，加上下面的，并改一下head中的种类数量
