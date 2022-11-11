_base_ = [
    './repvgg_a1_st.bj.dl_aic_512x288_20221107.py',
]

model = dict(
    _delete_=False,
    type='DisentangledKeypointRegressor',
    keypoint_head=dict(
        upsample_use_deconv=True,
        upsample_freeze_deconv=True,
        last_spp_branch=3,
        last_spp_channels=64,
        num_heatmap_filters=32,
        use_sigmoid=True,
        num_offset_filters_per_joint=16,
        num_offset_filters_layers=1,
    )
)
