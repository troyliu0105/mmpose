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
    keypoint_head=dict(
        type='DEKRHeadV2',
        in_channels=[64, 128, 256],
        in_index=[1, 2, 3],
        upsample_scales=[2, 4, 8],
    )
)

