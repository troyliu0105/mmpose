import math
import torch

from torch import nn

from mmcv.cnn import ConvModule
from .base_backbone import BaseBackbone
from .repvgg import RepVGGBlock
from ..builder import BACKBONES


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


class SimSPPF(nn.Module):
    """Simplified SPPF with ReLU activation"""

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = ConvModule(in_channels, c_, 1, 1)
        self.cv2 = ConvModule(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y = torch.cat([x, y1, y2, self.m(y2)], 1)
        y = self.cv2(y)
        return y


class BottleRep(nn.Module):

    def __init__(self, in_channels, out_channels, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = basic_block(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs


class RepBlock(nn.Module):
    """
    RepBlock is a stage block with rep-style basic block
    """

    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
        super().__init__()

        self.conv1 = block(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.block = nn.Sequential(*(block(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                                     for _ in range(n - 1))) if n > 1 else None
        if block == BottleRep:
            self.conv1 = BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = nn.Sequential(
                *(BottleRep(out_channels, out_channels, basic_block=basic_block, weight=True) for _ in
                  range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


@BACKBONES.register_module()
class EfficientRep(BaseBackbone):
    """EfficientRep Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    """
    arch_list = {
        'n': {'depth_multiple': 0.33, 'width_multiple': 0.25},
        't': {'depth_multiple': 0.33, 'width_multiple': 0.375},
        's': {'depth_multiple': 0.33, 'width_multiple': 0.50},
        'm': {'depth_multiple': 0.60, 'width_multiple': 0.75},
        'l': {'depth_multiple': 1.9, 'width_multiple': 1.0},
    }
    num_repeats = [1, 6, 12, 18, 6]
    out_channels = [64, 128, 256, 512, 1024]

    def __init__(
            self,
            arch=None,
            in_channels=3,
            channels_list=None,
            num_repeats=None,
            block='RepVGGBlock',
            out_indices=None,
    ):
        super().__init__()
        block = {'RepVGGBlock': RepVGGBlock, 'BottleRep': BottleRep}[block]

        assert bool(channels_list is not None) ^ (arch is not None)
        assert bool(num_repeats is not None) ^ (arch is not None)

        if arch is not None:
            cfg = self.__class__.arch_list[arch]
            backbone_num_repeats = self.__class__.num_repeats
            backbone_out_channels = self.__class__.out_channels
            depth_multiple = cfg['depth_multiple']
            width_multiple = cfg['width_multiple']
            channels_list = [make_divisible(i * width_multiple, 8) for i in (backbone_out_channels + [])]
            num_repeats = [(max(round(i * depth_multiple), 1) if i > 1 else i) for i in
                           (backbone_num_repeats + [])]
        if out_indices is None:
            out_indices = [0, 1, 2, 3]
        self.out_indices = out_indices

        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            RepBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                block=block,
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                block=block,
            )
        )

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                block=block,
            )
        )

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                block=block,
            ),
            SimSPPF(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5
            )
        )

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for i, block in enumerate([self.ERBlock_2, self.ERBlock_3, self.ERBlock_4, self.ERBlock_5]):
            x = block(x)
            if i in self.out_indices:
                outputs.append(x)

        return tuple(outputs)
