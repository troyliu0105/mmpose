import json

import numpy as np
from xtcocotools.cocoeval import COCOeval

from mmpose.datasets.builder import DATASETS
from .bottom_up_coco import BottomUpCocoDataset


@DATASETS.register_module()
class BottomUpSTGestureDataset(BottomUpCocoDataset):
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
    """
    def __init__(self, *args, **kwargs):
        super(BottomUpSTGestureDataset, self).__init__(*args, **kwargs)
        channels = self.ann_info['dataset_channel'][0]
        idx_relocate_map = {d: i for (i, d) in enumerate(channels)}
        self.ann_info['num_joints'] = len(channels)
        # flip_pairs
        new_flip_pairs = []
        for pair in self.ann_info['flip_pairs']:
            a, b = pair
            if a in channels and b in channels:
                new_flip_pairs.append([idx_relocate_map[a], idx_relocate_map[b]])
        self.ann_info['flip_pairs'] = new_flip_pairs
        # flip_index
        flip_index = list(range(len(channels)))
        for pair in self.ann_info['flip_pairs']:
            a, b = pair
            flip_index[a], flip_index[b] = flip_index[b], flip_index[a]
        self.ann_info['flip_index'] = flip_index
        # upper_body_ids
        upper_body_ids = []
        for i in self.ann_info['upper_body_ids']:
            if i in channels:
                upper_body_ids.append(idx_relocate_map[i])
        self.ann_info['upper_body_ids'] = upper_body_ids
        # lower_body_ids
        lower_body_ids = []
        for i in self.ann_info['lower_body_ids']:
            if i in channels:
                lower_body_ids.append(idx_relocate_map[i])
        self.ann_info['lower_body_ids'] = lower_body_ids
        # joint_weights
        self.ann_info['joint_weights'] = self.ann_info['joint_weights'][channels]
        # skeleton
        skeleton = []
        for sk in self.ann_info['skeleton']:
            a, b = sk
            if a in channels and b in channels:
                skeleton.append([idx_relocate_map[a], idx_relocate_map[b]])
        self.ann_info['skeleton'] = skeleton
        # coco anns
        for ann_key in self.coco.anns:
            ann = self.coco.anns[ann_key]
            kps = np.array(ann['keypoints'])
            kps = kps.reshape(-1, 3)[channels]
            ann['keypoints'] = kps.flatten().tolist()
            ann['num_keypoints'] = np.count_nonzero(kps[:, -1])
        self.sigmas = self.sigmas[channels]

    def _get_single(self, idx):
        return super()._get_single(idx)

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        with open(res_file, 'r') as file:
            res_json = json.load(file)
            if not res_json:
                info_str = list(zip(stats_names, [
                    0,
                ] * len(stats_names)))
                return info_str

        coco_det = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_det, 'keypoints', self.sigmas, use_area=False)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str

    def _get_joints(self, anno):
        results = super(BottomUpSTGestureDataset, self)._get_joints(anno)
        # results[:, [0, 7], 2] = (results[:, [0, 7], 2] == 2) * 2
        return results
