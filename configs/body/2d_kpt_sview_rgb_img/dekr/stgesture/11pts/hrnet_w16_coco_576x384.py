_base_ = [
    '../../../../../_base_/default_runtime.py',
    '../../../../../_base_/datasets/st_rail_gesture.py'
]
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
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=13,
    dataset_joints=13,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
# inference_channel=[11, 0, 1, 2, 3, 4, 8, 5, 6, 7, 9])

data_cfg = dict(
    image_size=[576, 384],
    base_size=[288, 192],
    base_sigma=2,
    heatmap_size=[[144, 96]],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    num_scales=1,
    scale_aware_sigma=False,
)

# model settings
model = dict(
    type='DEKR',
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(32,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(16, 32)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(16, 32, 64)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(16, 32, 64, 128),
                multiscale_output=True)),
    ),
    keypoint_head=dict(
        type='DEKRHead',
        in_channels=[16, 128],
        in_index=[0, 3],
        num_joints=channel_cfg['num_output_channels'],
        transition_head_channels=32,
        offset_pre_kpt=15,
        offset_pre_blocks=1,
        offset_feature_type="BasicBlock",
        input_transform="resize_concat",
        loss_keypoint=dict(
            type='DEKRMultiLossFactory',
            supervise_empty=True,
            num_joints=channel_cfg['num_output_channels'],
            num_stages=1,
            bg_weight=0.1,
            heatmaps_loss_factor=1.0,
            offset_loss_factor=0.03,
        )),
    train_cfg=dict(),
    test_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        max_num_people=30,
        scale_factor=[1],
        with_heatmaps=[True],
        project2image=False,
        align_corners=True,
        detection_threshold=0.2,
        nms_kernel=5,
        nms_padding=2,
        ignore_too_much=False,
        adjust=True,
        refine=True,
        flip_test=False))

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
    dict(
        type='BottomUpGenerateDEKRTargets',
        sigma=2,
    ),
    dict(
        type='Collect',
        keys=['img', 'offset', 'offset_w', 'target', 'mask'],
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
            'center', 'scale', 'flip_index'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/coco'
data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=32),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='BottomUpSTGestureDataset',
        ann_file=f'{data_root}/annotations/stgesture_person_keypoints_train2017.json',
        img_prefix=f'{data_root}/train2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='BottomUpSTGestureDataset',
        ann_file=f'{data_root}/annotations/stgesture_person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='BottomUpSTGestureDataset',
        ann_file=f'{data_root}/annotations/stgesture_person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}})
)
