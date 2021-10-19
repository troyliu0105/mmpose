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
                            dest_kps[dst] = kps[src]
                        valid_idx = dest_kps[:, -1] != 0
                        dest_kps[~valid_idx, :] = 0
                        min_x = np.maximum(0, np.min(dest_kps[valid_idx, 0]) - 1)
                        min_y = np.maximum(0, np.min(dest_kps[valid_idx, 1]) - 1)
                        max_x = np.minimum(anno["imageWidth"], np.max(dest_kps[valid_idx, 0]) - 1)
                        max_y = np.minimum(anno["imageHeight"], np.max(dest_kps[valid_idx, 1]) - 1)
                        area = (max_x - min_x) * (max_y - min_y)
                        kp_anno = {
                            "keypoints": dest_kps.flatten().tolist(),
                            "num_keypoints": int(valid_idx.sum()),
                            "category_id": 1,
                            "image_id": img_id,
                            "iscrowd": 0,
                            "area": int(area),
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
