_base_ = [
    './repvgg_a1_st.bj.dl_aic_512x512_20221021.py',
]

optimizer = dict(
    _delete_=True,
    type='SGD',
    lr=0.0001,
    momentum=0.9,
    weight_decay=0.0001
)
# learning policy
lr_config = dict(
    _delete_=True,
    policy='fixed')
total_epochs = 4

data_root = 'data'
data = dict(
    train=[
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=f'{data_root}/aic/annotations/st_gesture_aic_train.json',
            img_prefix=f'{data_root}/aic/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902',
            data_cfg={{_base_.data_cfg}},
            pipeline={{_base_.train_pipeline}},
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_platform/train.json',
            img_prefix=f'{data_root}/rails/rm_platform/images/',
            data_cfg={{_base_.data_cfg}},
            pipeline={{_base_.train_pipeline}},
            dataset_info={{_base_.dataset_info}}),
    ],
    val=dict(
        type='ConcatDataset',
        separate_eval=False,
        datasets=[
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=f'{data_root}/aic/annotations/st_gesture_aic_val.json',
                img_prefix=f'{data_root}/aic/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911',
                data_cfg={{_base_.data_cfg}},
                pipeline={{_base_.val_pipeline}},
                dataset_info={{_base_.dataset_info}}),
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_platform/val.json',
                img_prefix=f'{data_root}/rails/rm_platform/images/',
                data_cfg={{_base_.data_cfg}},
                pipeline={{_base_.val_pipeline}},
                dataset_info={{_base_.dataset_info}}),

        ]
    ),
    test=dict(
        type='ConcatDataset',
        separate_eval=False,
        datasets=[
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=f'{data_root}/aic/annotations/st_gesture_aic_val.json',
                img_prefix=f'{data_root}/aic/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911',
                data_cfg={{_base_.data_cfg}},
                pipeline={{_base_.test_pipeline}},
                dataset_info={{_base_.dataset_info}}),
            dict(
                type='BottomUpSTGestureDataset',
                ann_file=
                f'{data_root}/rails/rm_platform/val.json',
                img_prefix=f'{data_root}/rails/rm_platform/images/',
                data_cfg={{_base_.data_cfg}},
                pipeline={{_base_.test_pipeline}},
                dataset_info={{_base_.dataset_info}}),
        ]
    )
)
