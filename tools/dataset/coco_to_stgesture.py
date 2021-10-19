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
    mapper = {5: 2, 6: 5, 7: 1, 8: 6, 9: 0, 10: 7, 13: 11, 14: 9, 15: 12, 16: 10}

    # 关键点标签名
    kps_labels = [
        "left_wrist", "left_elbow", "left_shoulder",
        "head", "neck",
        "right_shoulder", "right_elbow", "right_wrist",
        "butt",
        "right_knee", "right_ankle",
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
            # 生成头、颈、屁股的关键点
            head = kps[:5, :]
            head = head[head[:, 2] > 0]
            if len(head) > 0:
                dest_kps[3, :2] = np.mean(head[:, :2], axis=0).astype(dest_kps.dtype)
                dest_kps[3, 2] = np.round(np.mean(head[:, 2])).astype(dest_kps.dtype)
            shoulders = kps[5:7]
            if np.all(shoulders[:, 2] > 0):
                dest_kps[4, :2] = np.mean(shoulders[:, :2], axis=0).astype(dest_kps.dtype)
                dest_kps[4, 2] = np.ceil(np.mean(shoulders[:, 2])).astype(dest_kps.dtype)
            hips = kps[11:13]
            if np.all(hips[:, 2] > 0):
                dest_kps[8, :2] = np.mean(hips[:, :2], axis=0).astype(dest_kps.dtype)
                dest_kps[8, 2] = np.ceil(np.mean(hips[:, 2])).astype(dest_kps.dtype)

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
