from mmpose.datasets.builder import DATASETS
from .bottom_up_coco import BottomUpCocoDataset


@DATASETS.register_module()
class BottomUpSTGestureDataset(BottomUpCocoDataset):
    def _get_single(self, idx):
        return super()._get_single(idx)
