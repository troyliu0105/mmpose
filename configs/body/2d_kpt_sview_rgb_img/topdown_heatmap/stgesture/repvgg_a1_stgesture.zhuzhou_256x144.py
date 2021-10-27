_base_ = ['../../../../_base_/datasets/st_rail_gesture.py']
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=10, metric='mAP', save_best='AP')

optimizer = dict(
    type='Adam',
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[66, 90])
total_epochs = 100
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=13,
    dataset_joints=13,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# model settings
model = dict(
    type='TopDown',
    backbone=dict(type='RepVGG', arch="a1", out_indices=[3]),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=256,
        in_index=-1,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=False,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    # 横着
    image_size=[256, 144],
    heatmap_size=[64, 36],
    # 竖着
    # image_size=[144, 256],
    # heatmap_size=[36, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    oks_thr=0.9,
    vis_thr=0.2,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/rails/rm_zhuzhou'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='TopDownSTGestureDataset',
        ann_file=f'{data_root}/zhuzhou.train.json',
        img_prefix=f'{data_root}/JpgImages/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='TopDownSTGestureDataset',
        ann_file=f'{data_root}/zhuzhou.val.json',
        img_prefix=f'{data_root}/JpgImages/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownSTGestureDataset',
        ann_file=f'{data_root}/zhuzhou.val.json',
        img_prefix=f'{data_root}/JpgImages/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
