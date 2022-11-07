# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, List

import torch

from torch import nn, Tensor
from torch.nn import functional as F
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


class SPPBlock(nn.Module):
    def __init__(self, branch=3):
        super(SPPBlock, self).__init__()
        self.branch = branch

    def forward(self, x):
        y = [x]
        for i in range(self.branch - 1):
            x = F.max_pool2d(x, 3, 1, 1)
            x = F.max_pool2d(x, 3, 1, 1)
            y.append(x)
        y = torch.cat(y, dim=1)
        return y


class BilinearConvTranspose2d(nn.ConvTranspose2d):
    """A conv transpose initialized to bilinear interpolation."""

    def __init__(self, channels, stride):
        """Set up the layer.
        Parameters
        ----------
        channels: int
            The number of input and output channels
        stride: int or tuple
            The amount of upsampling to do
        """
        if isinstance(stride, int):
            stride = (stride, stride)

        kernel_size = (2 * stride[0] - stride[0] % 2, 2 * stride[1] - stride[1] % 2)
        self.center_loc = (stride[0] - 1 if kernel_size[0] % 2 == 1 else stride[0] - 0.5,
                           stride[1] - 1 if kernel_size[1] % 2 == 1 else stride[1] - 0.5)
        padding = (
            int((kernel_size[0] - stride[0]) / 2),
            int((kernel_size[1] - stride[1]) / 2)
        )
        super().__init__(
            channels, channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=channels)

    def reset_parameters(self):
        """Reset the weight and bias."""
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.weight, 0)
        bilinear_kernel = self.bilinear_kernel(self.stride)
        for i in range(self.in_channels):
            if self.groups == 1:
                j = i
            else:
                j = 0
            self.weight.data[i, j] = bilinear_kernel

    def bilinear_kernel(self, stride):
        """Generate a bilinear upsampling kernel."""
        ksize = self.kernel_size
        kernel = torch.zeros(ksize)
        for y in range(ksize[0]):
            for x in range(ksize[1]):
                value = (1 - abs((y - self.center_loc[0]) / stride[0])) * \
                        (1 - abs((x - self.center_loc[1]) / stride[1]))
                kernel[y, x] = value
        return kernel


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
                 upsample_use_deconv=False,
                 num_offset_filters_layers=2,
                 offset_layer_type="AdaptiveBlock",
                 spp_channels=128,
                 spp_branch=0,
                 use_sigmoid=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.upsample_use_deconv = upsample_use_deconv
        self.offset_pre_spp_channels = spp_channels
        self.offset_pre_spp = spp_branch
        self.upsample_scales = upsample_scales
        self.use_sigmoid = use_sigmoid
        all_offset_layer_types = {"AdaptiveBlock": AdaptiveActivationBlock, "BasicBlock": BasicBlock,
                                  "Bottleneck": Bottleneck}
        offset_layer_clz = all_offset_layer_types[offset_layer_type]
        self.num_offset_filters_per_joint = kwargs.get('num_offset_filters_per_joint', 15)
        self.num_joints = kwargs.get('num_joints')
        self.num_heatmap_filters = kwargs.get('num_heatmap_filters')

        num_offset_filters = self.num_joints * self.num_offset_filters_per_joint

        if spp_branch > 0:
            self.final_layer = nn.Sequential(
                ConvModule(in_channels=self.in_channels,
                           out_channels=spp_channels,
                           kernel_size=1,
                           norm_cfg=dict(type='BN')),
                SPPBlock(branch=spp_branch)
            )
            self.in_channels = spp_channels * spp_branch

        if upsample_use_deconv:
            upsamples = []
            for s, c in zip(upsample_scales, kwargs.get('in_channels')):
                if s == 1:
                    upsamples.append(nn.Identity())
                else:
                    up = []
                    while s != 1:
                        up.append(BilinearConvTranspose2d(c, 2))
                        s //= 2
                    upsamples.append(nn.Sequential(*up))
            self.upsample_deconvs = nn.ModuleList(upsamples)

        self.heatmap_conv_layers = nn.Sequential(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=self.num_heatmap_filters,
                kernel_size=1,
                norm_cfg=dict(type='BN')),
            BasicBlock(self.num_heatmap_filters, self.num_heatmap_filters),
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=self.num_heatmap_filters,
                out_channels=1 + self.num_joints,
                kernel_size=1))
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
            if self.upsample_use_deconv:
                upsampled_inputs = [up(x) for up, x in zip(self.upsample_deconvs, inputs)]
            else:
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
        if self.use_sigmoid:
            heatmap = torch.sigmoid(heatmap)
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
