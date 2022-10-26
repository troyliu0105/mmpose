_base_ = [
    './repvgg_a1_st.bj.dl_aic_512x288_20221026.py',
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=10,
        scale_factor=[0.85, 1.15],
        scale_type='long',
        trans_factor=10,
        clip=True),
    dict(type='BottomUpRandomFlip', flip_prob=0.0),
    dict(
        type='PhotometricDistortion',
        brightness_delta=32,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=18),
    dict(type='ToTensor'),
    # butt as center
    dict(type='SelectKeypointAsCenterArea', center_ind=8),
    # dict(type='GetKeypointCenterArea'),
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

data_root = 'data'
data = dict(
    train=[
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_beijing/rm_beijing.20221021.train.json',
            img_prefix=f'{data_root}/rails/rm_beijing/images/',
            data_cfg={{_base_.data_cfg}},
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        # dict(
        #     type='BottomUpSTGestureDataset',
        #     ann_file=
        #     f'{data_root}/rails/rm_dalian/rm_dalian.20221021.train.json',
        #     img_prefix=f'{data_root}/rails/rm_dalian/images/',
        #     data_cfg={{_base_.data_cfg}},
        #     pipeline=train_pipeline,
        #     dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_shuohuang/rm_shuohuang.11pts.20221021.train.json',
            img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
            data_cfg={{_base_.data_cfg}},
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_shuohuang/rm_shuohuang.9pts.20221021.train.json',
            img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
            data_cfg={{_base_.data_cfg}},
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_platform/20221025.train.json',
            img_prefix=f'{data_root}/rails/rm_platform/images/',
            data_cfg={{_base_.data_cfg}},
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=f'{data_root}/aic/annotations/st_gesture_aic_train.json',
            img_prefix=f'{data_root}/aic/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902',
            data_cfg={{_base_.data_cfg}},
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}})
    ],
    val=dict(
        _delete_=True,
        type='ConcatDataset',
        separate_eval=False,
        datasets=[
            # dict(
            #     type='BottomUpSTGestureDataset',
            #     ann_file=
            #     f'{data_root}/rails/rm_beijing/rm_beijing.20221021.val.json',
            #     img_prefix=f'{data_root}/rails/rm_beijing/images/',
            #     data_cfg={{_base_.data_cfg}},
            #     pipeline={{_base_.val_pipeline}},
            #     dataset_info={{_base_.dataset_info}}),
            # dict(
            #     type='BottomUpSTGestureDataset',
            #     ann_file=
            #     f'{data_root}/rails/rm_dalian/rm_dalian.20221021.val.json',
            #     img_prefix=f'{data_root}/rails/rm_dalian/images/',
            #     data_cfg={{_base_.data_cfg}},
            #     pipeline={{_base_.val_pipeline}},
            #     dataset_info={{_base_.dataset_info}}),
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_shuohuang/rm_shuohuang.11pts.20221021.val.json',
                img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
                data_cfg={{_base_.data_cfg}},
                pipeline={{_base_.val_pipeline}},
                dataset_info={{_base_.dataset_info}}),
            # dict(
            #     type='BottomUpSTGestureDataset',
            #     ann_file=
            #     f'{data_root}/rails/rm_shuohuang/rm_shuohuang.9pts.20221021.val.json',
            #     img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
            #     data_cfg={{_base_.data_cfg}},
            #     pipeline={{_base_.val_pipeline}},
            #     dataset_info={{_base_.dataset_info}}),
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_platform/20221025.val.json',
                img_prefix=f'{data_root}/rails/rm_platform/images/',
                data_cfg={{_base_.data_cfg}},
                pipeline={{_base_.val_pipeline}},
                dataset_info={{_base_.dataset_info}}),
            # dict(
            #     type='BottomUpSTGestureDataset',
            #     ann_file=f'{data_root}/aic/annotations/st_gesture_aic_val.json',
            #     img_prefix=f'{data_root}/aic/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911',
            #     data_cfg={{_base_.data_cfg}},
            #     pipeline={{_base_.val_pipeline}},
            #     dataset_info={{_base_.dataset_info}}),
        ]
    ),
    test=dict(
        _delete_=True,
        type='ConcatDataset',
        separate_eval=False,
        datasets=[
            # dict(
            #     type='BottomUpSTGestureDataset',
            #     ann_file=
            #     f'{data_root}/rails/rm_beijing/rm_beijing.20221021.val.json',
            #     img_prefix=f'{data_root}/rails/rm_beijing/images/',
            #     data_cfg={{_base_.data_cfg}},
            #     pipeline={{_base_.test_pipeline}},
            #     dataset_info={{_base_.dataset_info}}),
            # dict(
            #     type='BottomUpSTGestureDataset',
            #     ann_file=
            #     f'{data_root}/rails/rm_dalian/rm_dalian.20221021.val.json',
            #     img_prefix=f'{data_root}/rails/rm_dalian/images/',
            #     data_cfg={{_base_.data_cfg}},
            #     pipeline={{_base_.test_pipeline}},
            #     dataset_info={{_base_.dataset_info}}),
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_shuohuang/rm_shuohuang.11pts.20221021.val.json',
                img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
                data_cfg={{_base_.data_cfg}},
                pipeline={{_base_.test_pipeline}},
                dataset_info={{_base_.dataset_info}}),
            # dict(
            #     type='BottomUpSTGestureDataset',
            #     ann_file=
            #     f'{data_root}/rails/rm_shuohuang/rm_shuohuang.9pts.20221021.val.json',
            #     img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
            #     data_cfg={{_base_.data_cfg}},
            #     pipeline={{_base_.test_pipeline}},
            #     dataset_info={{_base_.dataset_info}}),
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_platform/20221025.val.json',
                img_prefix=f'{data_root}/rails/rm_platform/images/',
                data_cfg={{_base_.data_cfg}},
                pipeline={{_base_.test_pipeline}},
                dataset_info={{_base_.dataset_info}}),
            # dict(
            #     type='BottomUpSTGestureDataset',
            #     ann_file=f'{data_root}/aic/annotations/st_gesture_aic_val.json',
            #     img_prefix=f'{data_root}/aic/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911',
            #     data_cfg={{_base_.data_cfg}},
            #     pipeline={{_base_.test_pipeline}},
            #     dataset_info={{_base_.dataset_info}}),
        ]
    )
)

