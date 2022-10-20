_base_ = [
    './repvgg_a1_st.bj.dl_aic_512x512_20221012.py',
]

channel_cfg = dict(
    num_output_channels=13,
    dataset_joints=13,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
)

data_cfg = dict(
    # image_size=[512, 288],
    # base_size=[256, 144],
    # base_sigma=2,
    # heatmap_size=[[128, 72]],
    image_size=[352, 288],
    base_size=[176, 144],
    base_sigma=2,
    heatmap_size=[[88, 72]],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    num_scales=1,
    scale_aware_sigma=False,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=5,
        scale_factor=[0.85, 1.15],
        scale_type='long',
        trans_factor=5),
    dict(type='BottomUpRandomFlip', flip_prob=0.0),
    dict(
        type='PhotometricDistortion',
        brightness_delta=32,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=18),
    dict(
        type='Albumentation',
        transforms=[
            dict(
                type='ImageCompression',
                quality_lower=40,
                quality_upper=80,
                p=0.9
            )
        ],
        keymap={'img': 'image', 'mask': 'old_mask'}),
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
            f'{data_root}/rails/rm_beijing/rm_beijing.20211123.train.json',
            img_prefix=f'{data_root}/rails/rm_beijing/images/',
            data_cfg=data_cfg,
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_shuohuang/11pts.train.json',
            img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
            data_cfg=data_cfg,
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_shuohuang/9pts.train.json',
            img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
            data_cfg=data_cfg,
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}),
    ],
    val=dict(
        type='ConcatDataset',
        separate_eval=False,
        datasets=[
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_beijing/rm_beijing.20211123.val.json',
                img_prefix=f'{data_root}/rails/rm_beijing/images/',
                data_cfg=data_cfg,
                pipeline=val_pipeline,
                dataset_info={{_base_.dataset_info}}),
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_shuohuang/11pts.val.json',
                img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
                data_cfg=data_cfg,
                pipeline=val_pipeline,
                dataset_info={{_base_.dataset_info}}),
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_shuohuang/9pts.val.json',
                img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
                data_cfg=data_cfg,
                pipeline=val_pipeline,
                dataset_info={{_base_.dataset_info}}),
        ]
    ),
    test=dict(
        type='ConcatDataset',
        separate_eval=False,
        datasets=[
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_beijing/rm_beijing.20211123.val.json',
                img_prefix=f'{data_root}/rails/rm_beijing/images/',
                data_cfg=data_cfg,
                pipeline=test_pipeline,
                dataset_info={{_base_.dataset_info}}),
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_shuohuang/11pts.val.json',
                img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
                data_cfg=data_cfg,
                pipeline=test_pipeline,
                dataset_info={{_base_.dataset_info}}),
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_shuohuang/9pts.val.json',
                img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
                data_cfg=data_cfg,
                pipeline=test_pipeline,
                dataset_info={{_base_.dataset_info}}),
        ]
    )
)
