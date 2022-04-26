import argparse
import json
import copy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Coco annotation json files")
    parser.add_argument("--output", required=True, type=str, help="output file")
    return parser.parse_args()


def combile(files, output):
    jsons = []
    for f in files:
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        jsons.append(data)
    # check all annotations are same type
    # assert len(set([d["categories"] for d in jsons])) == 1
    categories = jsons[0]["categories"]
    out_images = []
    out_annotations = []
    map_id_images = {}
    map_id_annotations = {}
    id_image = 0
    id_anno = 0
    for data in jsons:
        images = copy.deepcopy(data["images"])
        annotations = copy.deepcopy(data["annotations"])
        for anno in annotations:
            anno_id = anno["id"]
            image_id = anno["image_id"]
            if image_id not in map_id_images:
                image_item = None
                for i, image in enumerate(images):
                    if image["id"] == image_id:
                        image_item = image
                        break
                assert image_item is not None
                images.pop(i)
                map_id_images[image_id] = id_image
                image_item["id"] = id_image
                out_images.append(image_item)
                id_image += 1
            anno["id"] = id_anno
            anno["image_id"] = map_id_images[image_id]
            out_annotations.append(anno)
            id_anno += 1
    output_json = {
        "images": out_images,
        "annotations": out_annotations,
        "categories": categories
    }
    with open(output, "w", encoding="utf-8") as fp:
        json.dump(output_json, fp, ensure_ascii=False, allow_nan=False)


if __name__ == '__main__':
    args = parse_args()
    combile(args.files, args.output)
