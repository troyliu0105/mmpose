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
import xmltodict


def convert_to_voc(img_path, json_data):
    path_parts = os.path.normpath(json_data['img_name']).split(os.path.sep)
    fname = path_parts[-1]
    clip_name = None
    for p in reversed(path_parts):
        if p.startswith('clip'):
            clip_name = p
            break
    assert clip_name is not None, f'img_path: {img_path}'
    dst_name = f'{clip_name}_{fname}'
    shapes = json_data['shapes']
    if shapes == 0:
        print("Detect empty labels")
        raise ValueError("Detect empty labels")
    accepted_labels = ['司机头部', '玩手机']
    heads_map = {'左扭头': 'left', '直视': 'straight', '右扭头': 'right'}
    objs = []
    for shape in shapes:
        if (label := shape['label']) not in accepted_labels:
            continue
        if label == accepted_labels[0]:
            cls_type = f'head_{heads_map[shape["attributes"][0]["value"]]}'
        elif label == accepted_labels[1]:
            cls_type = 'phone'
        objs.append({
            'name': cls_type,
            'bndbox': {
                'xmin': shape['points'][0],
                'ymin': shape['points'][1],
                'xmax': shape['points'][2],
                'ymax': shape['points'][3],
            },
            'truncated': 0,
            'difficult': 0
        })
    data = {
        'annotation': {
            '@verified': 'no',
            'filename': dst_name,
            'size': {
                'width': json_data['width'],
                'height': json_data['height'],
                'depth': 3
            },
            'segmented': 0,
        }
    }
    if len(objs) > 0:
        if len(objs) == 1:
            objs = objs[0]
        data['annotation']['object'] = objs
    return dst_name, data


def main(args):
    root_dir = args.root
    output_dir = args.output
    if not os.path.exists(output_image_dir := os.path.join(output_dir, 'JPEGImages')):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_anno_dir := os.path.join(output_dir, 'Annotations')):
        os.makedirs(output_anno_dir)
    base_dirs = [f for f in glob.glob(f'{root_dir}/*') if os.path.isdir(f)]

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

    for img_file, json_file in tqdm.tqdm(pairs):
        with open(json_file) as fp:
            json_data = json.load(fp)
        try:
            dst_image_path, voc_anno_dict, = convert_to_voc(img_file, json_data)
            dst_anno_path = os.path.join(output_anno_dir, dst_image_path.replace('.jpg', '.xml'))
            dst_image_path = os.path.join(output_image_dir, dst_image_path)
            shutil.copy(img_file, dst_image_path)
            with open(dst_anno_path, 'w', encoding='utf-8') as fp:
                xml_str = xmltodict.unparse(voc_anno_dict, pretty=True, encoding='utf-8')
                fp.write(xml_str)
        except:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', help='root folder')
    parser.add_argument('output', help='output folder')
    opts = parser.parse_args()
    main(opts)
