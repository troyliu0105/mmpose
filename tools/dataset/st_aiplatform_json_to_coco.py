# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import json
import os
import random
import shutil
from datetime import datetime

import numpy as np
import tqdm

# 关键点标签名
kps_labels = [
    # 0         , 1             , 2
    'left_wrist', 'left_elbow', 'left_shoulder',
    # 3    , 4
    'head', 'neck',
    # 5              , 6             , 7
    'right_shoulder', 'right_elbow', 'right_wrist',
    # 8
    'butt',
    # 9           , 10
    'right_knee', 'right_ankle',
    # 11        , 12
    'left_knee', 'left_ankle',
]
kps_labels_cn = [
    '左手腕', '左手肘', '左肩', '头顶', '脖子', '右肩', '右手肘', '右手腕', '尾椎骨', '右膝盖', '右脚踝',
    '左膝盖', '左脚踝'
]
cn_vis_to_int = {'正常': 2, '遮挡': 1, '不存在': 0}
# 平台的标注顺序和 STGesture 定义顺序不一样，需要转换
# source: https://troyliu0105.notion.site/6e21ddbeb54f46a493546b70b1500b2f
# platform_label_seq_trans = [12, 11, 0, 1, 2, 3, 4, 8, 5, 6, 7, 9, 10]
platform_label_seq_trans = [2, 3, 4,
                            5, 6,
                            8, 9, 10,
                            7,
                            11, 12,
                            1, 0]
root_categories = [{
    'id':
        1,
    'name':
        'person',
    'keypoints':
        kps_labels,
    'skeleton': [[0, 1], [1, 2], [2, 4], [3, 4], [4, 5], [5, 6], [6, 7],
                 [4, 8], [8, 9], [9, 10], [8, 11], [11, 12]]
}]


def extract_pose(json_data, current_image_idx, current_anno_idx):
    annos = []
    shapes = json_data['shapes']
    polylines = [ann for ann in shapes if ann['type'] == 'polyline']
    for poly in polylines:
        if poly['label'] != '司机关键点':
            continue
        try:
            kpts = np.array(poly['points']).reshape((13, 2))
            kpts = kpts[platform_label_seq_trans]
        except:
            raise ValueError('标签错误！！！')
        vis_label = []
        for label_cn in kps_labels_cn:
            curr_attr = None
            for attr in poly['attributes']:
                curr_attr = attr
                if attr['name'] == label_cn:
                    break
            assert curr_attr is not None
            vis_label.append(cn_vis_to_int[curr_attr['value']])
        vis_label = np.array(vis_label)[:, None]
        kpts = np.concatenate((kpts, vis_label), axis=-1)
        kpts[kpts[:, -1] == 0, :] = 0

        vis_kpts = kpts[kpts[:, -1] != 0, :2].copy()
        xmin, ymin = np.min(vis_kpts[:, 0]), np.min(vis_kpts[:, 1])
        xmax, ymax = np.max(vis_kpts[:, 0]), np.max(vis_kpts[:, 1])
        bbox = np.array([xmin, ymin, xmax - xmin,
                         ymax - ymin]).astype(np.int32).tolist()

        kp_anno = {
            'keypoints': kpts.flatten().tolist(),
            'num_keypoints': int((kpts[:, -1] != 0).sum()),
            "avail_keypoints": list(range(13)),
            'category_id': 1,
            'image_id': current_image_idx,
            'iscrowd': 0,
            'area': bbox[2] * bbox[3],
            'bbox': bbox,
            'id': current_anno_idx
        }
        annos.append(kp_anno)
        current_anno_idx += 1
    return annos, current_anno_idx


def main(args):
    root_dir = args.root
    output_dir = args.output
    ratio = args.ratio
    out = os.path.join(output_dir, f'{datetime.now().strftime("%Y%m%d")}.json')
    if not os.path.exists(
            output_image_dir := os.path.join(output_dir, 'images')):
        os.makedirs(output_image_dir)
    base_dirs = [f for f in glob.glob(f'{root_dir}/*') if os.path.isdir(f)]

    current_image_idx = 0
    current_anno_idx = 0

    images = []
    jsons = []

    for base_dir in base_dirs:
        base_name = os.path.split(base_dir)[1]
        images_dir = os.path.join(base_dir, 'images')
        json_v0_dir = os.path.join(base_dir, 'json_v0')

        sub_images = glob.glob(f'{images_dir}/**/*.jpg', recursive=True)
        sub_jsons = glob.glob(f'{json_v0_dir}/*.json')
        images_names = set(
            [os.path.split(f)[1].replace('.jpg', '') for f in sub_images])
        jsons_names = set(
            [os.path.split(f)[1].replace('.json', '') for f in sub_jsons])
        common_names = images_names & jsons_names
        sub_images = [
            f for f in sub_images
            if os.path.split(f)[1].replace('.jpg', '') in common_names
        ]
        sub_jsons = [
            f'{json_v0_dir}/{os.path.split(f)[1].replace(".jpg", ".json")}'
            for f in sub_images
        ]

        images += sub_images
        jsons += sub_jsons

    pairs = list(zip(images, jsons))
    random.shuffle(pairs)

    imgs_split = [
        pairs[:int(len(pairs) * ratio)], pairs[int(len(pairs) * ratio):]
    ]

    for sub_pairs, suffix in zip(imgs_split, ['train', 'val']):
        root_images = []
        root_annotations = []
        for img_path, json_path in tqdm.tqdm(sub_pairs):
            base_name = os.path.normpath(json_path).split(os.path.sep)[-3]
            img_name = os.path.split(img_path)[1]
            img_dst_path = os.path.join(output_image_dir,
                                        f'{base_name}_{img_name}')
            img_dst_name = os.path.split(img_dst_path)[1]
            with open(json_path) as fp:
                json_data = json.load(fp)
            try:
                annos, current_anno_idx = extract_pose(json_data,
                                                       current_image_idx,
                                                       current_anno_idx)
            except ValueError:
                continue
            root_annotations += annos
            root_images.append({
                'file_name': img_dst_name,
                'height': json_data['height'],
                'width': json_data['width'],
                'id': current_image_idx
            })
            shutil.copy(img_path, img_dst_path)
            current_image_idx += 1
        annotations = {
            'images': root_images,
            'annotations': root_annotations,
            'categories': root_categories
        }
        if len(root_annotations) > 0:
            p, s = os.path.splitext(out)
            sub_out = p + f'.{suffix}' + s
            with open(sub_out, 'w') as fp:
                json.dump(annotations, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', help='root folder')
    parser.add_argument('output', help='output folder')
    parser.add_argument('ratio', default=0.9, type=float)
    opts = parser.parse_args()
    main(opts)
