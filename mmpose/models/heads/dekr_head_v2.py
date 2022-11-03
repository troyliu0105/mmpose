# Copyright (c) OpenMMLab. All rights reserved.
import torch

from torch import nn
from ..builder import HEADS
from .dekr_head import DEKRHead, AdaptiveActivationBlock
from ..backbones.resnet import BasicBlock, Bottleneck

from mmcv.cnn import ConvModule, build_conv_layer, constant_init, normal_init
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
                 num_offset_filters_layers=2,
                 offset_layer_type="AdaptiveBlock",
                 **kwargs):
        super().__init__(**kwargs)
        self.upsample_scales = upsample_scales
        all_offset_layer_types = {"AdaptiveBlock": AdaptiveActivationBlock, "BasicBlock": BasicBlock,
                                  "Bottleneck": Bottleneck}
        offset_layer_clz = all_offset_layer_types[offset_layer_type]
        self.num_offset_filters_per_joint = kwargs.get('num_offset_filters_per_joint', 15)
        self.num_joints = kwargs.get('num_joints')

        num_offset_filters = self.num_joints * self.num_offset_filters_per_joint

        self.offset_conv_transition_layer = nn.Sequential(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=num_offset_filters,
                kernel_size=1,
                norm_cfg=dict(type='BN'))
        )
        self.offset_conv_output_layers = nn.ModuleList(
            [
                nn.Sequential(
                    *[offset_layer_clz(self.num_offset_filters_per_joint, self.num_offset_filters_per_joint) for _ in
                      range(num_offset_filters_layers)],
                    build_conv_layer(
                        dict(type='Conv2d'),
                        in_channels=self.num_offset_filters_per_joint,
                        out_channels=2,
                        kernel_size=1,
                        groups=1
                    )
                )
                for _ in range(self.num_joints)
            ]
        )
        self.offset_conv_layers = nn.Sequential()

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

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        heatmap = self.heatmap_conv_layers(x)
        offset = self.offset_conv_transition_layer(x)
        final_offset = []
        offset_feature = torch.split(offset, self.num_offset_filters_per_joint, dim=1)
        for j in range(self.num_joints):
            o = self.offset_conv_output_layers[j](offset_feature[j])
            final_offset.append(o)
        offset = torch.cat(final_offset, dim=1)
        return [[heatmap, offset]]

    def init_weights(self):
        super().init_weights()
        for name, m in self.heatmap_conv_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for name, m in self.offset_conv_transition_layer.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'transform_matrix_conv' in name:
                    normal_init(m, std=1e-8, bias=0)
                else:
                    normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for name, m in self.offset_conv_output_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
