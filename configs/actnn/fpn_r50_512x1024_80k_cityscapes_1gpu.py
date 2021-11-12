_base_ = [
    '../_base_/models/fpn_r50.py',
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

actnn = dict(
    default_bit=4,
    auto_prec=False,
)
custom_hooks = [
    dict(
        type="ActNNHook",
        interval=1
    )
]
data = dict(
    samples_per_gpu=8, # 8*1 = 8
    workers_per_gpu=4,
)
# optimizer_config = dict(
#     grad_clip=dict(
#         mode='agc',
#         clip_factor=0.01,
#         eps=1e-3,
#         norm_type=2.0,
#     )
# )
# optimizer_config = dict(
#     grad_clip=dict(
#         max_norm=35,
#         norm_type=2
#     )
# )
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='segmentation',
                entity='actnn',
                name='fpn_r50_512x1024_80k_cityscapes_1gpu',
            )
        )
    ]
)
