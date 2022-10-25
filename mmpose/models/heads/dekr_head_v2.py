# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import HEADS
from .dekr_head import DEKRHead

from mmpose.models.utils.ops import resize

try:
    from mmcv.ops import DeformConv2d

    has_mmcv_full = True
except (ImportError, ModuleNotFoundError):
    has_mmcv_full = False


@HEADS.register_module()
class DEKRHeadV2(DEKRHead):
    """DisEntangled Keypoint Regression head. "Bottom-up human pose estimation
    via disentangled keypoint regression", CVPR'2021.

    Args:
        in_channels (int): Number of input channels.
        num_joints (int): Number of joints.
        num_heatmap_filters (int): Number of filters for heatmap branch.
        num_offset_filters_per_joint (int): Number of filters for each joint.
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            Default: None.

            - 'resize_concat': Multiple feature maps will be resized to the
                same size as the first one and then concat together.
                Usually used in FCN head of HRNet.
            - 'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            - None: Only one select feature map is allowed.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        heatmap_loss (dict): Config for heatmap loss. Default: None.
        offset_loss (dict): Config for offset loss. Default: None.
    """

    def __init__(self,
                 upsample_scales=(2, 4),
                 **kwargs):
        super().__init__(**kwargs)
        self.upsample_scales = upsample_scales

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, (list, tuple)):
            return inputs

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    scale_factor=scale,
                    mode='nearest') if scale != 1 else x for x, scale in zip(inputs, self.upsample_scales)
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs
