_base_ = [
    './regnetx_1.6gf_spp3.deconv_st.bj.dl_aic_512x288_20221106.py',
]

model = dict(
    _delete_=False,
    keypoint_head=dict(
        heatmap_loss=dict(
            _delete_=True,
            type='JointsMSELoss',
            use_target_weight=True,
            supervise_empty=False,
            loss_weight=1.0,
        )
    )
)
