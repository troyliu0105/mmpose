import argparse
import json

import numpy as np
import tqdm
from xtcocotools.coco import COCO

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True, help="Input annotation directory")
    parser.add_argument("--out", type=str, required=True, help="Output json annotation file")
    opts = parser.parse_args()

    # 可以直接映射 10 个 坐标点，但是头、脖子、屁股需要计算
    mapper = {0: 10, 1: 9, 4: 11, 5: 12, 6: 8, 8: 4, 9: 3,
              10: 7, 11: 6, 12: 5, 13: 2, 14: 1, 15: 0}

    # 关键点标签名
    kps_labels = [
        # 0
        "left_wrist", "left_elbow", "left_shoulder",
        # 3
        "head", "neck",
        # 5
        "right_shoulder", "right_elbow", "right_wrist",
        # 8
        "butt",
        # 9
        "right_knee", "right_ankle",
        # 11
        "left_knee", "left_ankle",
    ]
    root_categories = [{
        'id': 1, 'name': 'person',
        'keypoints': kps_labels,
        'skeleton': [[0, 1], [1, 2], [2, 4], [3, 4], [4, 5], [5, 6], [6, 7], [4, 8], [8, 9], [9, 10], [8, 11], [11, 12]]
    }]
    coco = COCO(opts.annotation_file)

    root_images = []
    root_annotations = []
    for img_id in tqdm.tqdm(coco.imgToAnns, desc="Processing..."):
        shapes = [s for s in coco.imgToAnns[img_id] if s['iscrowd'] == 0]
        for shape in shapes:
            kps = np.array(shape["keypoints"]).reshape(-1, 3)
            dest_kps = np.zeros((len(kps_labels), 3), dtype=np.int32)
            for src, dst in mapper.items():
                dest_kps[dst] = kps[src]
            valid_idx = dest_kps[:, -1] != 0
            dest_kps[~valid_idx, :] = 0
            kp_anno = {
                "keypoints": dest_kps.flatten().tolist(),
                "num_keypoints": int(valid_idx.sum()),
                "category_id": 1,
                "image_id": img_id,
                "iscrowd": 0,
                "bbox": shape['bbox'],
                "area": shape['area'],
                "id": shape['id']
            }
            if 'segmentation' in shape:
                kp_anno['segmentation'] = shape['segmentation']
            root_annotations.append(kp_anno)
        if len(shapes) > 0:
            root_images += coco.loadImgs(img_id)
    annotations = {
        "images": root_images,
        "annotations": root_annotations,
        "categories": root_categories
    }
    if len(root_annotations) > 0:
        with open(opts.out, "w") as fp:
            json.dump(annotations, fp)
