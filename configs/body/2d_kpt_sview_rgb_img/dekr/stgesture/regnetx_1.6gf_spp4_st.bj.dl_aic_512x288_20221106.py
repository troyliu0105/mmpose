_base_ = [
    './regnetx_1.6gf_spp4_st.bj.dl_aic_512x288_20221103.py',
]

model = dict(
    _delete_=False,
    type='DisentangledKeypointRegressor',
    backbone=dict(type='RegNet', arch='regnetx_1.6gf', out_indices=[0, 1, 2]),
    pretrained='https://download.openmmlab.com/pretrain/third_party/regnetx_1.6gf-5791c176.pth',
    keypoint_head=dict(
        type='DEKRHeadV2',
        in_channels=[72, 168, 408],
        in_index=[0, 1, 2],
        num_heatmap_filters=64,
        upsample_scales=[1, 2, 4],
        upsample_use_deconv=True,
        spp_branch=3,
        spp_channels=128
    )
)
