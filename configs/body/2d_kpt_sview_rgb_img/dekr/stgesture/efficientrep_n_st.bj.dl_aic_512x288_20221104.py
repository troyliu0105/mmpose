_base_ = [
    './regnetx_3.2gf_st.bj.dl_aic_512x288_20221103.py',
]

model = dict(
    _delete_=False,
    type='DisentangledKeypointRegressor',
    # n: 32, 64, 128, 256
    # s: 64, 128, 256, 512
    # m: 96, 192, 384, 768
    # l: 128, 256, 512, 1024
    backbone=dict(type='EfficientRep', arch='n', out_indices=[0, 1, 2, 3]),
    pretrained=None,
    keypoint_head=dict(
        type='DEKRHeadV2',
        in_channels=[64, 128, 256],
        in_index=[1, 2, 3],
        upsample_scales=[2, 4, 8],
        use_sigmoid=True,
        heatmap_loss=dict(
            _delete_=True,
            type='FocalHeatmapLoss'
        ),
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=30,
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
        type='BottomUpGenerateHeatmapTargetV2',
        sigma=(2, 2),
        gen_center_heatmap=True,
        bg_weight=0.999,
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
)
