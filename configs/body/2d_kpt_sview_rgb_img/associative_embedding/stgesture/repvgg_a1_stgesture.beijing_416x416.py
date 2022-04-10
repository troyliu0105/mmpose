_base_ = ['../../../../_base_/datasets/st_rail_gesture.py']
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='mAP', save_best='AP')

optimizer = dict(
    type='Adam',
    lr=0.005,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.005,
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
    type='AssociativeEmbedding',
    backbone=dict(type='RepVGG', arch="a1", out_indices=[3]),
    keypoint_head=dict(
        type='AESimpleHead',
        in_channels=256,
        num_joints=channel_cfg['num_output_channels'],
        in_index=-1,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        tag_per_joint=True,
        with_ae_loss=[True],
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=channel_cfg['num_output_channels'],
            supervise_empty=False,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[True],
            push_loss_factor=[0.01],
            pull_loss_factor=[0.01],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0],
        )),
    train_cfg=dict(),
    test_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        max_num_people=30,
        scale_factor=[1, 0.5],
        with_heatmaps=[True],
        with_ae=[True],
        project2image=False,
        align_corners=False,
        nms_kernel=5,
        nms_padding=2,
        tag_per_joint=True,
        detection_threshold=0.1,
        tag_threshold=1,
        use_detection_val=True,
        ignore_too_much=False,
        adjust=True,
        refine=True,
        flip_test=True))

data_cfg = dict(
    # 横着
    # image_size=[216, 384],
    # heatmap_size=[54, 96],
    image_size=416,
    heatmap_size=[104],
    # 竖着
    # image_size=[384, 216],
    # heatmap_size=[96, 54],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    base_sigma=2,
    base_size=208,
    num_scales=1,
    scale_aware_sigma=False,
    oks_thr=0.9,
    vis_thr=0.2,
)

train_pipeline = [
    dict(type='LoadImageAsThreeChannelGrayFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=30,
        scale_factor=[0.75, 1.5],
        scale_type='short',
        trans_factor=40),
    dict(type='BottomUpRandomFlip', flip_prob=0.5),
    dict(type='ToTensor'),
    dict(
        type='BottomUpGenerateTarget',
        sigma=2,
        max_num_people=30,
    ),
    dict(
        type='Collect',
        keys=['img', 'joints', 'targets', 'masks'],
        meta_keys=[]),
]

val_pipeline = [
    dict(type='LoadImageAsThreeChannelGrayFromFile'),
    dict(type='BottomUpGetImgSize', test_scale_factor=[1, 0.5]),
    dict(type='BottomUpResizeAlign', transforms=[
        dict(type='ToTensor'),
    ]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'aug_data', 'test_scale_factor', 'base_size',
            'center', 'scale', 'flip_index'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/rails/rm_beijing'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    # val_dataloader=dict(samples_per_gpu=32),
    # test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='BottomUpSTGestureDataset',
        ann_file=f'{data_root}/rm_beijing.20211028.train.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='BottomUpSTGestureDataset',
        ann_file=f'{data_root}/rm_beijing.20211028.val.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='BottomUpSTGestureDataset',
        ann_file=f'{data_root}/rm_beijing.20211028.val.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
