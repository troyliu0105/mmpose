_base_ = [
    './efficientrep_s_st.bj.dl_aic_512x288_20221104.py',
]

model = dict(
    keypoint_head=dict(
        upsample_use_deconv=True,
    )
)
