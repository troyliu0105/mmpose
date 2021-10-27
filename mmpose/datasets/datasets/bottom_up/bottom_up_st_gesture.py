import json

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
