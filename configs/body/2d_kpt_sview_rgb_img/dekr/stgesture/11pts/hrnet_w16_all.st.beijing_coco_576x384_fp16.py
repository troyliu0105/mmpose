_base_ = ['./hrnet_w16_all.st.beijing_coco_576x384.py']

# fp16 settings
fp16 = dict(loss_scale='dynamic')
