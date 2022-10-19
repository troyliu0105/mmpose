import argparse
import json
import os.path
import shutil
import tqdm
from xtcocotools.coco import COCO


def main(args):
    output_root = args.output_root
    output_json = args.output_json
    folders = args.folders
    jsons = args.jsons
    assert len(folders) == len(jsons)

    current_image_id = 0
    current_anno_id = 0
    categories = None
    images = []
    annotations = []
    for folder, jf in tqdm.tqdm(zip(folders, jsons)):
        coco = COCO(jf)
        if categories is None:
            categories = list(coco.cats.values())
        for img_id in tqdm.tqdm(coco.getImgIds()):
            img_info = coco.imgs[img_id]
            img_info["id"] = current_image_id

            src_img_file = os.path.join(folder, img_info["file_name"])
            dst_img_file = os.path.join(output_root, img_info["file_name"])
            img_annos = coco.imgToAnns[img_id]
            for anno in img_annos:
                anno["image_id"] = current_image_id
                anno["id"] = current_anno_id
                current_anno_id += 1
                annotations.append(anno)
            current_image_id += 1
            images.append(img_info)
            shutil.copy(src_img_file, dst_img_file)
    data = {"images": images, "annotations": annotations, "categories": categories}
    with open(output_json, "w") as fp:
        json.dump(data, fp, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=str, help="output validate folder")
    parser.add_argument("output_json", type=str, help="output validate json")
    parser.add_argument("--folders", type=str, nargs='+', help="image folders")
    parser.add_argument("--jsons", type=str, nargs='+', help="json annotations")
    opts = parser.parse_args()
    main(opts)
