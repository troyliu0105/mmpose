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
    lr=0.0015,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[70, 90])
total_epochs = 110
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=13,
    dataset_joints=13,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ],
    # inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    inference_channel=[0, 1, 2, 3, 4, 8, 5, 6, 7])

data_cfg = dict(
    image_size=512,
    base_size=256,
    base_sigma=2,
    heatmap_size=[128],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    num_scales=1,
    scale_aware_sigma=False,
)

# model settings
model = dict(
    type='DisentangledKeypointRegressor',
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(3,),
                num_channels=(32,)),
            stage2=dict(
                num_modules=2,
                num_branches=2,
                block='BASIC',
                num_blocks=(2, 2),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=3,
                num_branches=3,
                block='BASIC',
                num_blocks=(2, 2, 2),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=2,
                num_branches=2,
                block='BASIC',
                num_blocks=(2, 2),
                num_channels=(32, 64),
                multiscale_output=True)),
    ),
    keypoint_head=dict(
        type='DEKRHead',
        in_channels=[32, 64],
        in_index=[0, 1],
        num_heatmap_filters=32,
        num_joints=len(channel_cfg['dataset_channel'][0]),
        num_offset_filters_per_joint=15,
        num_offset_filters_layers=1,
        offset_layer_type="BasicBlock",
        input_transform="resize_concat",
        heatmap_loss=dict(
            type='JointsMSELoss',
            use_target_weight=True,
            supervise_empty=False,
            loss_weight=1.0,
        ),
        offset_loss=dict(
            type='SoftWeightSmoothL1Loss',
            use_target_weight=True,
            supervise_empty=False,
            loss_weight=0.002,
            beta=1 / 9.0,
        )),
    train_cfg=dict(),
    test_cfg=dict(
        num_joints=len(channel_cfg['inference_channel']),
        max_num_people=30,
        scale_factor=[1],
        project2image=False,
        align_corners=False,
        max_pool_kernel=3,
        use_nms=True,
        nms_dist_thr=0.05,
        nms_joints_thr=5,
        keypoint_threshold=0.01,
        flip_test=False
    ))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=30,
        scale_factor=[0.75, 1.5],
        scale_type='short',
        trans_factor=40),
    dict(type='BottomUpRandomFlip', flip_prob=0.5),
    dict(type='PhotometricDistortion',
         brightness_delta=32,
         contrast_range=(0.8, 1.2),
         saturation_range=(0.8, 1.2),
         hue_delta=18),
    dict(type='ToTensor'),
    dict(type='GetKeypointCenterArea'),
    dict(
        type='BottomUpGenerateHeatmapTarget',
        sigma=(2, 4),
        gen_center_heatmap=True,
        bg_weight=0.1,
    ),
    dict(
        type='BottomUpGenerateOffsetTarget',
        radius=4,
    ),
    dict(
        type='Collect',
        keys=['img', 'heatmaps', 'masks', 'offsets', 'offset_weights'],
        meta_keys=[]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BottomUpGetImgSize', test_scale_factor=[1]),
    dict(
        type='BottomUpResizeAlign',
        transforms=[
            dict(type='ToTensor'),
        ]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'aug_data', 'test_scale_factor', 'base_size',
            'center', 'scale', 'flip_index', 'num_joints', 'skeleton',
            'image_size', 'heatmap_size'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data'
data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=32),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=[
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=f'{data_root}/rails/rm_beijing/rm_beijing.20211123.train.json',
            img_prefix=f'{data_root}/rails/rm_beijing/images/',
            data_cfg=data_cfg,
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=f'{data_root}/rails/rm_dalian/rm_dalian.20211123.train.json',
            img_prefix=f'{data_root}/rails/rm_dalian/images/',
            data_cfg=data_cfg,
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=f'{data_root}/rails/rm_shuohuang/rm_shuohuang.20211223.total.train.json',
            img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
            data_cfg=data_cfg,
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        # dict(
        #     type='BottomUpSTGestureDataset',
        #     ann_file=f'{data_root}/coco/annotations/stgesture_person_keypoints_val2017.json',
        #     img_prefix=f'{data_root}/coco/val2017/',
        #     data_cfg=data_cfg,
        #     pipeline=train_pipeline,
        #     dataset_info={{_base_.dataset_info}})
    ],
    val=dict(
        type='BottomUpSTGestureDataset',
        ann_file=f'{data_root}/rails/rm_shuohuang/rm_shuohuang.20211223.total.val.json',
        img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='BottomUpSTGestureDataset',
        ann_file=f'{data_root}/rails/rm_shuohuang/rm_shuohuang.20211223.total.val.json',
        img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
