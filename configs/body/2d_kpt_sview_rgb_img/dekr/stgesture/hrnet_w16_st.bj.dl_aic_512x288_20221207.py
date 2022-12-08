_base_ = [
    './repvgg_a1_st.bj.dl_aic_512x288_20221026.py',
]

channel_cfg = dict(
    _delete_=True,
    num_output_channels=13,
    dataset_joints=13,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

data_cfg = dict(
    _delete_=True,
    image_size=[512, 288],
    base_size=[256, 144],
    base_sigma=2,
    heatmap_size=[[128, 72]],
    # image_size=512,
    # base_size=256,
    # base_sigma=2,
    # heatmap_size=[128],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    num_scales=1,
    scale_aware_sigma=False,
)

model = dict(
    _delete_=True,
    type='DisentangledKeypointRegressor',
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(16,)),
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
        type='DEKRHeadV2',
        in_channels=[16, 32, 64, 128],
        in_index=[0, 1, 2, 3],
        upsample_scales=[1, 2, 4, 8],
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
            loss_weight=0.03,
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
        nms_joints_thr=9,
        keypoint_threshold=0.01,
        flip_test=False
    ))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=30,
        scale_factor=[0.8, 1.2],
        scale_type='long',
        trans_factor=20,
        clip=True),
    dict(type='BottomUpRandomFlip', flip_prob=0.0),
    # dict(
    #     type='PhotometricDistortion',
    #     brightness_delta=32,
    #     contrast_range=(0.7, 1.3),
    #     saturation_range=(0.7, 1.3),
    #     hue_delta=18),
    dict(type='ToTensor'),
    # butt as center
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
    dict(type='BottomUpGetImgSize', test_scale_factor=[1], base_length=32),
    dict(type='BottomUpResizeAlign', base_length=32, transforms=[
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
    train=[
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_beijing/rm_beijing.20221021.train.json',
            img_prefix=f'{data_root}/rails/rm_beijing/images/',
            data_cfg=data_cfg,
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        # dict(
        #     type='BottomUpSTGestureDataset',
        #     ann_file=
        #     f'{data_root}/rails/rm_dalian/rm_dalian.20221021.train.json',
        #     img_prefix=f'{data_root}/rails/rm_dalian/images/',
        #     data_cfg=data_cfg,
        #     pipeline=train_pipeline,
        #     dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_shuohuang/rm_shuohuang.11pts.20221021.train.json',
            img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
            data_cfg=data_cfg,
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_shuohuang/rm_shuohuang.9pts.20221021.train.json',
            img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
            data_cfg=data_cfg,
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_platform/20221122.train.json',
            img_prefix=f'{data_root}/rails/rm_platform/images/',
            data_cfg=data_cfg,
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        # dict(
        #     type='BottomUpSTGestureDataset',
        #     ann_file=f'{data_root}/aic/annotations/st_gesture_aic_train.json',
        #     img_prefix=f'{data_root}/aic/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902',
        #     data_cfg=data_cfg,
        #     pipeline=train_pipeline,
        #     dataset_info={{_base_.dataset_info}})
    ],
    val=dict(
        _delete_=True,
        type='ConcatDataset',
        separate_eval=False,
        data_cfg=data_cfg,
        dataset_info={{_base_.dataset_info}},
        datasets=[
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_shuohuang/rm_shuohuang.9pts.20221021.val.json',
                img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
                data_cfg=data_cfg,
                pipeline=val_pipeline,
                dataset_info={{_base_.dataset_info}}),
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_shuohuang/rm_shuohuang.11pts.20221021.val.json',
                img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
                data_cfg=data_cfg,
                pipeline=val_pipeline,
                dataset_info={{_base_.dataset_info}}),
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_platform/20221122.val.json',
                img_prefix=f'{data_root}/rails/rm_platform/images/',
                data_cfg=data_cfg,
                pipeline=val_pipeline,
                dataset_info={{_base_.dataset_info}}),
        ]
    ),
    test=dict(
        _delete_=True,
        type='ConcatDataset',
        separate_eval=False,
        data_cfg=data_cfg,
        dataset_info={{_base_.dataset_info}},
        datasets=[
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_shuohuang/rm_shuohuang.9pts.20221021.val.json',
                img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
                data_cfg=data_cfg,
                pipeline=test_pipeline,
                dataset_info={{_base_.dataset_info}}),
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_shuohuang/rm_shuohuang.11pts.20221021.val.json',
                img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
                data_cfg=data_cfg,
                pipeline=test_pipeline,
                dataset_info={{_base_.dataset_info}}),
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_platform/20221122.val.json',
                img_prefix=f'{data_root}/rails/rm_platform/images/',
                data_cfg=data_cfg,
                pipeline=test_pipeline,
                dataset_info={{_base_.dataset_info}}),
        ]
    )
)

optimizer = dict(
    type='Adam',
    lr=0.001,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[25, 40])
total_epochs = 50
