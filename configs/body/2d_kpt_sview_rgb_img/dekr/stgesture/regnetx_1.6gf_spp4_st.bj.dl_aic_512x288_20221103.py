_base_ = [
    './regnetx_3.2gf_st.bj.dl_aic_512x288_20221103.py',
]

model = dict(
    _delete_=False,
    type='DisentangledKeypointRegressor',
    # regnetx_1.6gf: 72, 168, 408
    # regnetx_3.2gf: 96, 192, 432
    # regnetx_4.0gf: 80, 240, 560
    backbone=dict(type='RegNet', arch='regnetx_1.6gf', out_indices=[1, 2]),
    pretrained='https://download.openmmlab.com/pretrain/third_party/regnetx_3.2gf-c2599b0f.pth',
    keypoint_head=dict(
        type='DEKRHeadV2',
        in_channels=[168, 408],
        in_index=[0, 1],
        upsample_scales=[2, 4],
        spp_branch=4,
    )
)

