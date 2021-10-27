_base_ = [
    '../_base_/models/fcn_r50-d8.py',
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64)
)

actnn = True
data = dict(
    samples_per_gpu=8, # 8*1 = 8
    workers_per_gpu=4,
)
optimizer_config = dict(
    grad_clip=dict(
        max_norm=35,
        norm_type=2
    )
)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='segmentation',
                entity='actnn',
                name='fcn_r18-d8_512x1024_80k_cityscapes_1gpu',
            )
        )
    ]
)
