# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch.nn as nn
from mmcv.cnn import constant_init, ConvModule, normal_init
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.utils import get_root_logger
from .base_backbone import BaseBackbone
from .utils import load_checkpoint
from ..builder import BACKBONES


class HourglassModuleLite(nn.Module):
    """Hourglass Module for HourglassNet backbone.

    Generate module recursively and use BasicBlock as the base unit.

    Args:
        depth (int): Depth of current HourglassModule.
        stage_channels (list[int]): Feature channels of sub-modules in current
            and follow-up HourglassModule.
        stage_blocks (list[int]): Number of sub-modules stacked in current and
            follow-up HourglassModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(self,
                 depth,
                 stage_channel,
                 increase=128,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()

        self.depth = depth

        cur_channel = stage_channel
        next_channel = stage_channel + increase

        self.up1 = ConvModule(cur_channel, cur_channel, kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg, )
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.low1 = ConvModule(cur_channel, next_channel, kernel_size=3, stride=2, padding=1, norm_cfg=norm_cfg)

        if self.depth > 1:
            self.low2 = HourglassModuleLite(depth - 1, next_channel, increase, norm_cfg)
        else:
            self.low2 = ConvModule(next_channel, next_channel, kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg)

        self.low3 = ConvModule(next_channel, cur_channel, kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg)

        self.up2 = ConvModule(cur_channel, cur_channel, kernel_size=2, stride=2, padding=0,
                              norm_cfg=norm_cfg, conv_cfg=dict(type="deconv"))

    def forward(self, x):
        """Model forward function."""
        up1 = self.up1(x)
        low1 = self.low1(x)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


@BACKBONES.register_module()
class HourglassNetLite(BaseBackbone):
    def __init__(self,
                 downsample_times=2,
                 num_stacks=4,
                 stack_pre_channels=(32, 32, 64, 128),
                 channel_increase=128,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()

        self.num_stacks = num_stacks
        assert self.num_stacks >= 1
        assert len(stack_pre_channels) == num_stacks

        self.stem = nn.Sequential(
            ConvModule(3, 32, 3, padding=1, stride=2, norm_cfg=norm_cfg),
            ConvModule(32, 32, 3, padding=1, stride=2, norm_cfg=norm_cfg),
        )

        self.hourglass_modules = nn.ModuleList([
            nn.Sequential(
                HourglassModuleLite(downsample_times, stack_pre_channels[i], channel_increase),
                ConvModule(stack_pre_channels[i], stack_pre_channels[i + 1], kernel_size=3, stride=1, padding=1,
                           norm_cfg=norm_cfg) if i != num_stacks - 1 else nn.Identity()
            )
            for i in range(num_stacks)
        ])

    """HourglassNet backbone.

    Stacked Hourglass Networks for Human Pose Estimation.
    More details can be found in the `paper
    <https://arxiv.org/abs/1603.06937>`__ .

    Args:
        downsample_times (int): Downsample times in a HourglassModule.
        num_stacks (int): Number of HourglassModule modules stacked,
            1 for Hourglass-52, 2 for Hourglass-104.
        stage_channels (list[int]): Feature channel of each sub-module in a
            HourglassModule.
        stage_blocks (list[int]): Number of sub-modules stacked in a
            HourglassModule.
        feat_channel (int): Feature channel of conv after a HourglassModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.

    Example:
        >>> from mmpose.models import HourglassNet
        >>> import torch
        >>> self = HourglassNet()
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 511, 511)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     print(tuple(level_output.shape))
        (1, 256, 128, 128)
        (1, 256, 128, 128)
    """

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Model forward function."""
        inter_feat = self.stem(x)
        out_feats = [inter_feat]

        for ind in range(self.num_stacks):
            single_hourglass = self.hourglass_modules[ind]

            inter_feat = single_hourglass(inter_feat)
            out_feats.append(inter_feat)

        return out_feats
