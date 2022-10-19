_base_ = [
    './repvgg_a1_st.bj.dl_aic_512x512_20221012.py'
]

# model settings
model = dict(
    backbone=dict(_delete_=True, type='RepVGG', arch="a0", out_indices=[1, 2, 3]),
    keypoint_head=dict(
        in_channels=[48, 96, 192],
        # in_channels=[128, 256],
        in_index=[0, 1, 2],
        # in_index=[1, 2],
    ))

data_root = 'data'
data = dict(
    train=[
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_beijing/rm_beijing.20211123.train.json',
            img_prefix=f'{data_root}/rails/rm_beijing/images/',
            data_cfg={{_base_.data_cfg}},
            pipeline={{_base_.train_pipeline}},
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_beijing/rm_beijing.20211123.val.json',
            img_prefix=f'{data_root}/rails/rm_beijing/images/',
            data_cfg={{_base_.data_cfg}},
            pipeline={{_base_.train_pipeline}},
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_dalian/rm_dalian.20211123.train.json',
            img_prefix=f'{data_root}/rails/rm_dalian/images/',
            data_cfg={{_base_.data_cfg}},
            pipeline={{_base_.train_pipeline}},
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_dalian/rm_dalian.20211123.val.json',
            img_prefix=f'{data_root}/rails/rm_dalian/images/',
            data_cfg={{_base_.data_cfg}},
            pipeline={{_base_.train_pipeline}},
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_shuohuang/11pts.train.json',
            img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
            data_cfg={{_base_.data_cfg}},
            pipeline={{_base_.train_pipeline}},
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_shuohuang/11pts.val.json',
            img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
            data_cfg={{_base_.data_cfg}},
            pipeline={{_base_.train_pipeline}},
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_shuohuang/9pts.train.json',
            img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
            data_cfg={{_base_.data_cfg}},
            pipeline={{_base_.train_pipeline}},
            dataset_info={{_base_.dataset_info}}),
        # dict(
        #     type='BottomUpSTGestureDataset',
        #     ann_file=f'{data_root}/aic/annotations/st_gesture_aic_train.json',
        #     img_prefix=f'{data_root}/aic/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902',
        #     data_cfg={{_base_.data_cfg}},
        #     pipeline={{_base_.train_pipeline}},
        #     dataset_info={{_base_.dataset_info}})
    ],
    val=dict(
        type='BottomUpSTGestureDataset',
        ann_file=
        f'{data_root}/rails/rm_shuohuang/9pts.val.json',
        img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
        data_cfg={{_base_.data_cfg}},
        pipeline={{_base_.val_pipeline}},
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='BottomUpSTGestureDataset',
        ann_file=
        f'{data_root}/rails/rm_shuohuang/9pts.val.json',
        img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
        data_cfg={{_base_.data_cfg}},
        pipeline={{_base_.test_pipeline}},
        dataset_info={{_base_.dataset_info}}),
    # val=dict(
    #     type='BottomUpSTGestureDataset',
    #     ann_file=f'{data_root}/aic/annotations/st_gesture_aic_val.json',
    #     img_prefix=f'{data_root}/aic/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911',
    #     data_cfg={{_base_.data_cfg}},
    #     pipeline={{_base_.val_pipeline}},
    #     dataset_info={{_base_.dataset_info}}),
    # test=dict(
    #     type='BottomUpSTGestureDataset',
    #     ann_file=f'{data_root}/aic/annotations/st_gesture_aic_val.json',
    #     img_prefix=f'{data_root}/aic/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911',
    #     data_cfg={{_base_.data_cfg}},
    #     pipeline={{_base_.test_pipeline}},
    #     dataset_info={{_base_.dataset_info}}),
)
