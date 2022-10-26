import argparse
import os.path
from os.path import join as path_join

import cv2
import numpy as np
import tqdm
from xtcocotools.coco import COCO


def plot_keypoints(img, kps):
    for i, kp in enumerate(kps):
        if kp[2] > 0:
            center = (int(kp[0]), int(kp[1]))
            cv2.circle(img, center, 4, (255, 0, 0))
            cv2.putText(img, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2, color=(255, 0, 0))
    return img


def main(img_root, json_file, out_root):
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    coco = COCO(json_file)
    img_infos = coco.imgs.values()
    for img_info in tqdm.tqdm(img_infos):
        anno = coco.loadAnns(coco.getAnnIds(imgIds=img_info['id']))
        src_path = path_join(img_root, img_info['file_name'])
        dst_path = path_join(out_root, img_info['file_name'])
        img = cv2.imread(src_path)
        for a in anno:
            img = plot_keypoints(img, np.array(a['keypoints']).reshape(-1, 3))
        cv2.imwrite(dst_path, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default='',
        help='Json file containing image info.')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='Root of the output img file. '
             'Default not saving the visualization images.')
    opts = parser.parse_args()
    main(opts.img_root, opts.json_file, opts.out_img_root)
