import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(input_channels, internal_neurons, kernel_size=1, stride=1, padding=0, bias=True)
        self.up = nn.Conv2d(internal_neurons, input_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size()[2:])
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        x = x * inputs
        return x


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros', use_se=False):
        super(RepVGGBlock, self).__init__()
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2
        self.nonlinearity = nn.ReLU()
        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()
        self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation, groups=groups, bias=True,
                                     padding_mode=padding_mode)
        self.rbr_identity = nn.BatchNorm2d(
            num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, groups=groups)
        self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                               stride=stride, padding=padding_11, groups=groups)

    def train(self: T, mode: bool = True) -> T:
        if not mode:
            self.switch_to_deploy()
        return super().train(mode)

    def forward(self, inputs):
        if self.training:
            if self.rbr_identity is None:
                id_out = 0
            else:
                id_out = self.rbr_identity(inputs)
            return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
        else:
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor.to(device=branch.weight.device, dtype=branch.weight.dtype)
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias


@BACKBONES.register_module()
class RepVGG(BaseBackbone):
    """RepVGG backbone

    """

    optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    g2_map = {l: 2 for l in optional_groupwise_layers}
    g4_map = {l: 4 for l in optional_groupwise_layers}
    arch_settings = {
        "a0": dict(num_blocks=[2, 4, 14, 1],
                   width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None),
        "a1": dict(num_blocks=[2, 4, 14, 1],
                   width_multiplier=[1, 1, 1, 2.5], override_groups_map=None),
        "a2": dict(num_blocks=[2, 4, 14, 1],
                   width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None),
        "b0": dict(num_blocks=[4, 6, 16, 1],
                   width_multiplier=[1, 1, 1, 2.5], override_groups_map=None),
        "b1": dict(num_blocks=[4, 6, 16, 1],
                   width_multiplier=[2, 2, 2, 4], override_groups_map=None),
        "b1g2": dict(num_blocks=[4, 6, 16, 1],
                     width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map),
        "b1g4": dict(num_blocks=[4, 6, 16, 1],
                     width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map),
        "b2": dict(num_blocks=[4, 6, 16, 1],
                   width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None),
        "b2g2": dict(num_blocks=[4, 6, 16, 1],
                     width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map),
        "b2g4": dict(num_blocks=[4, 6, 16, 1],
                     width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map),
        "b3": dict(num_blocks=[4, 6, 16, 1],
                   width_multiplier=[3, 3, 3, 5], override_groups_map=None),
        "b3g2": dict(num_blocks=[4, 6, 16, 1],
                     width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map),
        "b3g4": dict(num_blocks=[4, 6, 16, 1],
                     width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map),
        "d2se": dict(num_blocks=[8, 14, 24, 1],
                     width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, use_se=True)
    }

    def __init__(self,
                 arch,
                 num_classes=-1,
                 out_indices=None,
                 frozen_stages=-1
                 ):
        super().__init__()
        assert arch in self.arch_settings
        settings = self.arch_settings[arch]
        num_blocks = settings.get("num_blocks")
        width_multiplier = settings.get("width_multiplier")
        override_groups_map = settings.get("override_groups_map")
        use_se = settings.get("use_se", False)
        assert len(width_multiplier) == 4

        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se

        assert 0 not in self.override_groups_map

        self.frozen_stages = frozen_stages
        self.num_classes = num_classes
        self.in_channels = 3
        if out_indices is None:
            out_indices = (5,) if num_classes > 0 else (4,)
        self.out_indices = out_indices
        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=self.in_channels,
                                  out_channels=self.in_planes,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1,
                                  use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(),
                nn.Linear(int(512 * width_multiplier[3]), num_classes)
            )

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups,
                                      use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        outs = []
        requested_stage = list(self.out_indices)
        for stage_id in range(5):
            if len(requested_stage) == 0:
                break
            stage = getattr(self, "stage{}".format(stage_id))
            x = stage(x)
            if stage_id in requested_stage:
                outs.append(x)
                requested_stage.remove(stage_id)
        if self.num_classes > 0:
            x = self.classifier(x)
            outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            m = getattr(self, "stage{}".format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        self._freeze_stages()
        return super().train(mode)
