import os

import numpy as np
from xtcocotools.cocoeval import COCOeval

from . import TopDownCocoDataset
from ...builder import DATASETS


@DATASETS.register_module()
class TopDownSTGestureDataset(TopDownCocoDataset):
    """ST rail gestures dataset for top-down pose estimation.

        The dataset loads raw features and apply specified transforms
        to return a dict containing the image tensors and other information.

        STGesture keypoint indexes::

            0: 'left_wrist',
            1: 'left_elbow',
            2: 'left_shoulder',
            3: 'head',
            4: 'neck',
            5: 'right_shoulder',
            6: 'right_elbow',
            7: 'right_wrist',
            8: 'butt',
            9: 'right_knee',
            10: 'right_ankle',
            11: 'left_knee',
            12: 'left_ankle'

        Args:
            ann_file (str): Path to the annotation file.
            img_prefix (str): Path to a directory where images are held.
                Default: None.
            data_cfg (dict): config
            pipeline (list[dict | callable]): A sequence of data transforms.
            dataset_info (DatasetInfo): A class containing all dataset info.
            test_mode (bool): Store True when building test or
                validation dataset. Default: False.
        """
    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):
        data_cfg['use_gt_bbox'] = True
        data_cfg['bbox_file'] = None
        data_cfg['det_bbox_thr'] = 0.0
        data_cfg['use_nms'] = False
        data_cfg['soft_nms'] = False
        data_cfg['nms_thr'] = 1.0
        super(TopDownSTGestureDataset, self).__init__(ann_file, img_prefix, data_cfg, pipeline,
                                                      dataset_info=dataset_info, test_mode=test_mode)

    def _load_coco_keypoint_annotation_kernel(self, img_id):
        """load annotation from COCOAPI.

        Note:
            bbox:[x1, y1, w, h]
        Args:
            img_id: coco image id
        Returns:
            dict: db entry
        """
        img_ann = self.coco.loadImgs(img_id)[0]
        width = img_ann['width']
        height = img_ann['height']
        num_joints = self.ann_info['num_joints']

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            if 'bbox' not in obj:
                obj['bbox'] = [0, 0, width, height]
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        bbox_id = 0
        rec = []
        for obj in objs:
            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

            center, scale = self._xywh2cs(*obj['clean_bbox'][:4])

            image_file = os.path.join(self.img_prefix, self.id2name[img_id])
            rec.append({
                'image_file': image_file,
                'center': center,
                'scale': scale,
                'bbox': obj['clean_bbox'][:4],
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1

        return rec

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_det, 'keypoints', self.sigmas, use_area=False)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str
