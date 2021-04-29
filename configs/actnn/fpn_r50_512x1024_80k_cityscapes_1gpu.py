_base_ = [
    '../_base_/models/fpn_r50.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

conv_cfg = dict(type='QConv2d')
norm_cfg = dict(type='QBN2d', requires_grad=True)
act_cfg = dict(type='QReLU')
model = dict(
    backbone=dict(
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg),
    neck=dict(
        conv_cfg=conv_cfg,
        norm_cfg=None,
        act_cfg=None),
    decode_head=dict(
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg),
)

data = dict(samples_per_gpu=8, workers_per_gpu=4)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                entity='actnn',
                project='segmentation',
                name='fpn_r50_512x1024_80k_cityscapes_1gpu',
            )
        )
    ]
)
