"""
9pts:   0:0,1:1,2:2,3:3,4:4,5:8,6:5,7:6,8:7
11pts:  0:11,1:0,2:1,3:2,4:3,5:4,6:8,7:5,8:6,9:7,10:9
beijing:    "left_wrist", "left_elbow", "left_shoulder",
            "head", "neck", "butt",
            "right_shoulder", "right_elbow", "right_wrist"
dalian:     "left_wrist", "left_elbow", "left_shoulder",
            "head",
            "right_shoulder", "right_elbow", "right_wrist",
"""

import argparse
import glob
import json
import random
from os import path

import numpy as np
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-folder", type=str, required=True, help="Input image directory")
    parser.add_argument("--annotation-folder", type=str, required=True, help="Input annotation directory")
    parser.add_argument("--out", type=str, required=True, help="Output json annotation file")
    parser.add_argument("--map", type=str, default="0:0,1:1,2:2", help="map current kps to generic kps")
    parser.add_argument("--ratio", type=float, default=1.0, help="Train set ratio")
    opts = parser.parse_args()

    # 标签类别 idx 映射
    mapper = opts.map
    mapper = {m[0]: m[1] for m in [[int(i) for i in x.split(":")] for x in mapper.split(",")]}

    # 关键点标签名
    kps_labels = [
        # 0         , 1           , 2
        "left_wrist", "left_elbow", "left_shoulder",
        # 3   , 4
        "head", "neck",
        # 5             , 6            , 7
        "right_shoulder", "right_elbow", "right_wrist",
        # 8
        "butt",
        # 9         , 10
        "right_knee", "right_ankle",
        # 11       , 12
        "left_knee", "left_ankle",
    ]
    root_categories = [{
        'id': 1, 'name': 'person',
        'keypoints': kps_labels,
        'skeleton': [[0, 1], [1, 2], [2, 4], [3, 4], [4, 5], [5, 6], [6, 7], [4, 8], [8, 9], [9, 10], [8, 11], [11, 12]]
    }]

    imgs = glob.glob(f"{opts.img_folder}/*.jpg")
    random.shuffle(imgs)
    imgs_split = [imgs[:int(len(imgs) * opts.ratio)], imgs[int(len(imgs) * opts.ratio):]]
    # anns = glob.glob(f"{opts.annotation_folder}/*.json")
    img_id = 1
    ann_id = 1
    for imgs, suffix in zip(imgs_split, ["train", "val"]):
        root_images = []
        root_annotations = []
        for img in tqdm.tqdm(imgs, desc="Processing..."):
            img_name = path.split(img)[1]
            anno_name = path.splitext(img_name)[0] + ".json"
            anno_file = path.join(opts.annotation_folder, anno_name)
            if path.exists(anno_file):
                try:
                    with open(anno_file, encoding="utf-8") as fp:
                        anno = json.load(fp)
                except:
                    try:
                        with open(anno_file, encoding="utf-8") as fp:
                            anno = json.load(fp)
                    except:
                        continue
                try:
                    shapes = anno['shapes']
                    record_annos = []
                    for shape in shapes:
                        points = np.array(shape["points"])
                        points_type = np.array(shape["point_type"])
                        kps = np.concatenate((points, points_type[:, None]), axis=-1)
                        dest_kps = np.zeros((len(kps_labels), 3), dtype=np.int32)
                        for src, dst in mapper.items():
                            if src < len(kps):
                                dest_kps[dst] = kps[src]
                        valid_idx = dest_kps[:, -1] != 0
                        dest_kps[~valid_idx, :] = 0
                        min_x = np.maximum(0, np.min(dest_kps[valid_idx, 0]) - 1)
                        min_y = np.maximum(0, np.min(dest_kps[valid_idx, 1]) - 1)
                        max_x = np.minimum(anno["imageWidth"], np.max(dest_kps[valid_idx, 0]) - 1)
                        max_y = np.minimum(anno["imageHeight"], np.max(dest_kps[valid_idx, 1]) - 1)
                        # scale = 1.2
                        # w = max_x - min_x
                        # h = max_y - min_y
                        # bbox = np.array([min_x - (scale - 1) * w / 2, min_y - (scale * 1) * h / 2,
                        #                  w * scale, h * scale])
                        # bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, anno["imageWidth"])
                        # bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, anno["imageHeight"])
                        # area = bbox[2] * bbox[3]
                        kp_anno = {
                            "keypoints": dest_kps.flatten().tolist(),
                            "num_keypoints": int(valid_idx.sum()),
                            "category_id": 1,
                            "image_id": img_id,
                            "iscrowd": 0,
                            # "area": int(area),
                            # "bbox": bbox.astype(np.int32).tolist(),
                            "bbox": [0, 0, anno["imageWidth"], anno["imageHeight"]],
                            "id": ann_id
                        }
                        root_annotations.append(kp_anno)
                        ann_id += 1
                except:
                    continue
                record_image = {
                    "file_name": img_name,
                    "height": anno["imageHeight"],
                    "width": anno["imageWidth"],
                    "id": img_id
                }
                root_images.append(record_image)
                img_id += 1
        annotations = {
            "images": root_images,
            "annotations": root_annotations,
            "categories": root_categories
        }
        if len(root_annotations) > 0:
            p, s = path.splitext(opts.out)
            out = p + f".{suffix}" + s
            with open(out, "w") as fp:
                json.dump(annotations, fp)
