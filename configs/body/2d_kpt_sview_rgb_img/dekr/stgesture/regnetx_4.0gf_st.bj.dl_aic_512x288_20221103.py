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
    # regnetx_3.2gf: 96, 192, 432
    # regnetx_4.0gf: 80, 240, 560
    backbone=dict(type='RegNet', arch='regnetx_4.0gf', out_indices=[1, 2]),
    pretrained='https://download.openmmlab.com/pretrain/third_party/regnetx_4.0gf-a88f671e.pth',
    keypoint_head=dict(
        type='DEKRHeadV2',
        in_channels=[240, 560],
        in_index=[0, 1],
        upsample_scales=[2, 4],
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
        nms_joints_thr=5,
        keypoint_threshold=0.01,
        flip_test=False
    ))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=10,
        scale_factor=[0.85, 1.15],
        scale_type='long',
        trans_factor=10,
        clip=True),
    dict(type='BottomUpRandomFlip', flip_prob=0.5),
    dict(
        type='PhotometricDistortion',
        brightness_delta=32,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=18),
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
            f'{data_root}/rails/rm_platform/20221025.train.json',
            img_prefix=f'{data_root}/rails/rm_platform/images/',
            data_cfg=data_cfg,
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=f'{data_root}/aic/annotations/st_gesture_aic_train.json',
            img_prefix=f'{data_root}/aic/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902',
            data_cfg=data_cfg,
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}})
    ],
    val=dict(
        _delete_=True,
        type='ConcatDataset',
        separate_eval=False,
        data_cfg=data_cfg,
        dataset_info={{_base_.dataset_info}},
        datasets=[
            # dict(
            #     type='BottomUpSTGestureDataset',
            #     ann_file=
            #     f'{data_root}/rails/rm_beijing/rm_beijing.20221021.val.json',
            #     img_prefix=f'{data_root}/rails/rm_beijing/images/',
            #     data_cfg=data_cfg,
            #     pipeline=val_pipeline,
            #     dataset_info={{_base_.dataset_info}}),
            # dict(
            #     type='BottomUpSTGestureDataset',
            #     ann_file=
            #     f'{data_root}/rails/rm_dalian/rm_dalian.20221021.val.json',
            #     img_prefix=f'{data_root}/rails/rm_dalian/images/',
            #     data_cfg=data_cfg,
            #     pipeline=val_pipeline,
            #     dataset_info={{_base_.dataset_info}}),
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_shuohuang/rm_shuohuang.11pts.20221021.val.json',
                img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
                data_cfg=data_cfg,
                pipeline=val_pipeline,
                dataset_info={{_base_.dataset_info}}),
            # dict(
            #     type='BottomUpSTGestureDataset',
            #     ann_file=
            #     f'{data_root}/rails/rm_shuohuang/rm_shuohuang.9pts.20221021.val.json',
            #     img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
            #     data_cfg=data_cfg,
            #     pipeline=val_pipeline,
            #     dataset_info={{_base_.dataset_info}}),
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_platform/20221025.val.json',
                img_prefix=f'{data_root}/rails/rm_platform/images/',
                data_cfg=data_cfg,
                pipeline=val_pipeline,
                dataset_info={{_base_.dataset_info}}),
            # dict(
            #     type='BottomUpSTGestureDataset',
            #     ann_file=f'{data_root}/aic/annotations/st_gesture_aic_val.json',
            #     img_prefix=f'{data_root}/aic/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911',
            #     data_cfg=data_cfg,
            #     pipeline=val_pipeline,
            #     dataset_info={{_base_.dataset_info}}),
        ]
    ),
    test=dict(
        _delete_=True,
        type='ConcatDataset',
        separate_eval=False,
        data_cfg=data_cfg,
        dataset_info={{_base_.dataset_info}},
        datasets=[
            # dict(
            #     type='BottomUpSTGestureDataset',
            #     ann_file=
            #     f'{data_root}/rails/rm_beijing/rm_beijing.20221021.val.json',
            #     img_prefix=f'{data_root}/rails/rm_beijing/images/',
            #     data_cfg=data_cfg,
            #     pipeline=test_pipeline,
            #     dataset_info={{_base_.dataset_info}}),
            # dict(
            #     type='BottomUpSTGestureDataset',
            #     ann_file=
            #     f'{data_root}/rails/rm_dalian/rm_dalian.20221021.val.json',
            #     img_prefix=f'{data_root}/rails/rm_dalian/images/',
            #     data_cfg=data_cfg,
            #     pipeline=test_pipeline,
            #     dataset_info={{_base_.dataset_info}}),
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_shuohuang/rm_shuohuang.11pts.20221021.val.json',
                img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
                data_cfg=data_cfg,
                pipeline=test_pipeline,
                dataset_info={{_base_.dataset_info}}),
            # dict(
            #     type='BottomUpSTGestureDataset',
            #     ann_file=
            #     f'{data_root}/rails/rm_shuohuang/rm_shuohuang.9pts.20221021.val.json',
            #     img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
            #     data_cfg=data_cfg,
            #     pipeline=test_pipeline,
            #     dataset_info={{_base_.dataset_info}}),
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_platform/20221025.val.json',
                img_prefix=f'{data_root}/rails/rm_platform/images/',
                data_cfg=data_cfg,
                pipeline=test_pipeline,
                dataset_info={{_base_.dataset_info}}),
            # dict(
            #     type='BottomUpSTGestureDataset',
            #     ann_file=f'{data_root}/aic/annotations/st_gesture_aic_val.json',
            #     img_prefix=f'{data_root}/aic/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911',
            #     data_cfg=data_cfg,
            #     pipeline=test_pipeline,
            #     dataset_info={{_base_.dataset_info}}),
        ]
    )
)
