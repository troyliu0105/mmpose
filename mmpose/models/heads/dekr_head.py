import torch
import torch.nn as nn
from mmcv import ops
from mmcv.cnn import (ConvModule, make_res_layer)
from mmcv.cnn.resnet import BasicBlock

from mmpose.models.builder import build_loss, HEADS
from mmpose.models.utils.ops import resize


class BasicBlockRepeat2(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BasicBlockRepeat2, self).__init__()
        self.block = nn.Sequential(*[BasicBlock(*args, **kwargs) for _ in range(2)])

    def forward(self, x):
        return self.block(x)


class BasicBlockRepeat3(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BasicBlockRepeat3, self).__init__()
        self.block = nn.Sequential(*[BasicBlock(*args, **kwargs) for _ in range(3)])

    def forward(self, x):
        return self.block(x)


@HEADS.register_module()
class DEKRHead(nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_joints=17,
                 transition_head_channels=32,
                 offset_pre_kpt=15,
                 offset_pre_blocks=1,
                 offset_feature_type="AdaptBlock",
                 extra=None,
                 in_index=0,
                 input_transform=None,
                 align_corners=False,
                 loss_keypoint=None):
        super().__init__()
        offset_feature_types = {"AdaptBlock": AdaptBlock, "BasicBlock": BasicBlock,
                                "BasicBlockRepeat2": BasicBlockRepeat2,
                                "BasicBlockRepeat3": BasicBlockRepeat3}
        assert offset_feature_type in offset_feature_types.keys()
        self.in_channels = in_channels
        self.loss = build_loss(loss_keypoint)
        self.num_joints = num_joints

        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners
        self.offset_pre_kpt = offset_pre_kpt

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        self.transition_heatmap = ConvModule(self.in_channels, transition_head_channels,
                                             1, 1, 0, norm_cfg=dict(type="BN"))
        self.transition_offset = ConvModule(self.in_channels, num_joints * offset_pre_kpt,
                                            1, 1, 0, norm_cfg=dict(type="BN"))
        self.heatmap_head = nn.Sequential(
            make_res_layer(BasicBlock, transition_head_channels, transition_head_channels, 1, 1),
            ConvModule(transition_head_channels, num_joints + 1, 1, 1, 0,
                       act_cfg=None)
        )
        offset_feature_layers = [make_res_layer(
            offset_feature_types[offset_feature_type],
            offset_pre_kpt,
            offset_pre_kpt,
            offset_pre_blocks,
            1, 1)
            for _ in range(num_joints)]

        offset_final_layers = [
            ConvModule(offset_pre_kpt, 2, 1, 1, 0, act_cfg=None)
            for _ in range(num_joints)]
        self.offset_feature_layers = nn.ModuleList(offset_feature_layers)
        self.offset_final_layers = nn.ModuleList(offset_final_layers)

    """Simple deconv head.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resized to the
                same size as the first one and then concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_keypoint (dict): Config for loss. Default: None.
    """

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform is not None, in_channels and in_index must be
        list or tuple, with the same length.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

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
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def get_loss(self, outputs, target, mask, offset, offset_w):
        """Calculate bottom-up masked mse loss.

        Note:
            batch_size: N
            num_channels: C
            heatmaps height: H
            heatmaps weight: W

        Args:
            outputs (List(torch.Tensor[NxCxHxW])): Multi-scale outputs.
            target (List(torch.Tensor[NxCxHxW])): Multi-scale targets.
            mask (List(torch.Tensor[NxHxW])): Masks of multi-scale targets.
        """

        losses = dict()
        pred_heatmap, pred_offset = outputs
        heatmap_loss, offset_loss = self.loss(pred_heatmap, pred_offset,
                                              target[0], mask[0], offset[0], offset_w[0])
        losses['heatmap_loss'] = heatmap_loss
        losses['offset_loss'] = offset_loss

        return losses

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        heatmap_feature = self.transition_heatmap(x)
        offset_feature = self.transition_offset(x)

        heatmap = self.heatmap_head(heatmap_feature)
        final_offset = []
        for j in range(self.num_joints):
            offset = offset_feature[:, j * self.offset_pre_kpt:(j + 1) * self.offset_pre_kpt]
            offset = self.offset_feature_layers[j](offset)
            offset = self.offset_final_layers[j](offset)
            final_offset.append(offset)
        offset = torch.cat(final_offset, dim=1)
        return heatmap, offset

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if hasattr(m, 'transform_matrix_conv'):
                nn.init.constant_(m.transform_matrix_conv.weight, 0)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.transform_matrix_conv.bias, 0)
            if hasattr(m, 'translation_conv'):
                nn.init.constant_(m.translation_conv.weight, 0)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.translation_conv.bias, 0)


class AdaptBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 dilation=1, downsample=None, deformable_groups=1, *args, **kwargs):
        super(AdaptBlock, self).__init__()
        regular_matrix = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1],
                                       [-1, 0, 1, -1, 0, 1, -1, 0, 1]])
        self.register_buffer('regular_matrix', regular_matrix.float())
        self.downsample = downsample
        self.transform_matrix_conv = nn.Conv2d(inplanes, 4, 3, 1, 1, bias=True)
        self.translation_conv = nn.Conv2d(inplanes, 2, 3, 1, 1, bias=True)
        self.adapt_conv = ops.DeformConv2d(inplanes, planes, kernel_size=3, stride=stride,
                                           padding=dilation, dilation=dilation, bias=False, groups=deformable_groups)
        self.bn = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        N, _, H, W = x.shape
        transform_matrix = self.transform_matrix_conv(x)
        transform_matrix = transform_matrix.permute(0, 2, 3, 1).reshape((N * H * W, 2, 2))
        offset = torch.matmul(transform_matrix, self.regular_matrix)
        offset = offset - self.regular_matrix
        offset = offset.transpose(1, 2).reshape((N, H, W, 18)).permute(0, 3, 1, 2)

        translation = self.translation_conv(x)
        # x 方向的位移
        offset[:, 0::2, :, :] += translation[:, 0:1, :, :]
        # y 方向的位移
        offset[:, 1::2, :, :] += translation[:, 1:2, :, :]

        out = self.adapt_conv(x, offset)
        out = self.bn(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
