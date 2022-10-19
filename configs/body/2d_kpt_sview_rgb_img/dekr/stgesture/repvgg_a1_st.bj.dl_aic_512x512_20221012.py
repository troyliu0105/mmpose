_base_ = [
    './repvgg_a1_st.bj_576x384.py',
]

channel_cfg = dict(
    num_output_channels=13,
    dataset_joints=13,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        # [0, 1, 2, 3, 4, 8, 5, 6, 7]
    ],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # inference_channel=[0, 1, 2, 3, 4, 8, 5, 6, 7])
    # inference_channel=[0, 1, 2, 3, 4, 8, 5, 6, 7, 11, 9])

data_cfg = dict(
    # image_size=[512, 288],
    # base_size=[256, 144],
    # base_sigma=2,
    # heatmap_size=[[128, 72]],
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
    backbone=dict(_delete_=True, type='RepVGG', arch="a1", out_indices=[1, 2, 3]),
    keypoint_head=dict(
        _delete_=False,
        in_channels=(64, 128, 256),
        in_index=(0, 1, 2)),
)

data_root = 'data'
data = dict(
    train=[
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_beijing/rm_beijing.20211123.train.json',
            img_prefix=f'{data_root}/rails/rm_beijing/images/',
            data_cfg=data_cfg,
            pipeline={{_base_.train_pipeline}},
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_beijing/rm_beijing.20211123.val.json',
            img_prefix=f'{data_root}/rails/rm_beijing/images/',
            data_cfg=data_cfg,
            pipeline={{_base_.train_pipeline}},
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_dalian/rm_dalian.20211123.train.json',
            img_prefix=f'{data_root}/rails/rm_dalian/images/',
            data_cfg=data_cfg,
            pipeline={{_base_.train_pipeline}},
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_dalian/rm_dalian.20211123.val.json',
            img_prefix=f'{data_root}/rails/rm_dalian/images/',
            data_cfg=data_cfg,
            pipeline={{_base_.train_pipeline}},
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_shuohuang/11pts.train.json',
            img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
            data_cfg=data_cfg,
            pipeline={{_base_.train_pipeline}},
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_shuohuang/11pts.val.json',
            img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
            data_cfg=data_cfg,
            pipeline={{_base_.train_pipeline}},
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=
            f'{data_root}/rails/rm_shuohuang/9pts.train.json',
            img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
            data_cfg=data_cfg,
            pipeline={{_base_.train_pipeline}},
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='BottomUpSTGestureDataset',
            ann_file=f'{data_root}/aic/annotations/st_gesture_aic_train.json',
            img_prefix=f'{data_root}/aic/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902',
            data_cfg=data_cfg,
            pipeline={{_base_.train_pipeline}},
            dataset_info={{_base_.dataset_info}})
    ],
    val=dict(
        type='BottomUpSTGestureDataset',
        ann_file=
        f'{data_root}/rails/rm_shuohuang/9pts.val.json',
        img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
        data_cfg=data_cfg,
        pipeline={{_base_.val_pipeline}},
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='BottomUpSTGestureDataset',
        ann_file=
        f'{data_root}/rails/rm_shuohuang/9pts.val.json',
        img_prefix=f'{data_root}/rails/rm_shuohuang/images/',
        data_cfg=data_cfg,
        pipeline={{_base_.test_pipeline}},
        dataset_info={{_base_.dataset_info}}),
    # val=dict(
    #     type='BottomUpSTGestureDataset',
    #     ann_file=f'{data_root}/aic/annotations/st_gesture_aic_val.json',
    #     img_prefix=f'{data_root}/aic/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911',
    #     data_cfg=data_cfg,
    #     pipeline={{_base_.val_pipeline}},
    #     dataset_info={{_base_.dataset_info}}),
    # test=dict(
    #     type='BottomUpSTGestureDataset',
    #     ann_file=f'{data_root}/aic/annotations/st_gesture_aic_val.json',
    #     img_prefix=f'{data_root}/aic/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911',
    #     data_cfg=data_cfg,
    #     pipeline={{_base_.test_pipeline}},
    #     dataset_info={{_base_.dataset_info}}),
)
