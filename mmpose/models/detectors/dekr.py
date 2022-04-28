import warnings
from typing import List

import cv2
import mmcv
import numpy as np
import torch
from mmcv.image import imwrite
from mmcv.parallel import DataContainer
from mmcv.visualization.image import imshow

from mmpose.core.evaluation import (aggregate_scale, aggregate_stage_flip,
                                    flip_feature_maps)
from mmpose.core.visualization import imshow_keypoints, imshow_bboxes
from mmpose.core.post_processing.group import DEKRParser
from mmpose.core.post_processing.post_transforms import transform_preds
from .base import BasePose
from .. import builder
from ..builder import POSENETS

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class DEKR(BasePose):
    def __init__(self,
                 backbone,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None):
        super(DEKR, self).__init__()
        self.fp16_enabled = False
        self.backbone = builder.build_backbone(backbone)

        if keypoint_head is not None:
            if 'loss_keypoint' not in keypoint_head and loss_pose is not None:
                warnings.warn(
                    '`loss_pose` for BottomUp is deprecated, '
                    'use `loss_keypoint` for heads instead. See '
                    'https://github.com/open-mmlab/mmpose/pull/382'
                    ' for more information.', DeprecationWarning)
                keypoint_head['loss_keypoint'] = loss_pose

            self.keypoint_head = builder.build_head(keypoint_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_udp = test_cfg.get('use_udp', False)
        self.parser = DEKRParser(test_cfg)
        self.pretrained = pretrained
        self.init_weights()

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        if pretrained is not None:
            self.pretrained = pretrained
        self.backbone.init_weights(self.pretrained)
        if self.with_keypoint:
            self.keypoint_head.init_weights()

    @auto_fp16(apply_to=('img',))
    def forward(self,
                img=None,
                target=None,
                mask=None,
                offset=None,
                offset_w=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss is True.
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            heatmaps weight: W
            heatmaps height: H
            max_num_people: M
        Args:
            img (torch.Tensor[NxCximgHximgW]): Input image.
            targets (List(torch.Tensor[NxKxHxW])): Multi-scale target heatmaps.
            masks (List(torch.Tensor[NxHxW])): Masks of multi-scale target
                                              heatmaps
            offset (List[torch.Tensor[Nx2KxHxW]]): Multi-scale offset targets
            offset_w (List[torch.Tensor[Nx2KxHxW]]): Multi-scale weight for offset loss
            img_metas(dict):Information about val&test
                By default this includes:
                - "image_file": image path
                - "aug_data": input
                - "test_scale_factor": test scale factor
                - "base_size": base size of input
                - "center": center of image
                - "scale": scale of image
                - "flip_index": flip index of keypoints

            return loss(bool): Option to 'return_loss'. 'return_loss=True' for
                training, 'return_loss=False' for validation & test
            return_heatmap (bool) : Option to return heatmap.

        Returns:
            dict|tuple: if 'return_loss' is true, then return losses.
              Otherwise, return predicted poses, scores, image
              paths and heatmaps.
        """

        if return_loss:
            return self.forward_train(img, target, mask, offset, offset_w, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)

    # noinspection PyMethodOverriding
    def forward_train(self, img, target, mask, offset, offset_w, img_metas,
                      **kwargs):
        output = self.backbone(img)
        if self.with_keypoint:
            output = self.keypoint_head(output)
        losses = {}
        if self.with_keypoint:
            keypoint_losses = self.keypoint_head.get_loss(
                output, target, mask, offset, offset_w
            )
            losses.update(keypoint_losses)
        return losses

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            Tensor: Outputs.
        """
        output = self.backbone(img)
        if self.with_keypoint:
            output = self.keypoint_head(output)
        return output

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        assert img.size(0) == 1
        assert len(img_metas) == 1

        if isinstance(img_metas, DataContainer):
            img_metas = img_metas.data[0][0]
        else:
            img_metas = img_metas[0]
        # 添加虚拟中心点的 flip_index（不翻转中心点）
        img_metas['offset_flip_index'] = img_metas['flip_index'].copy()
        img_metas['flip_index'].append(len(img_metas['flip_index']))

        aug_data = img_metas['aug_data']

        test_scale_factor = img_metas['test_scale_factor']
        base_size = img_metas['base_size']
        center = img_metas['center']
        scale = img_metas['scale']
        result = {}

        scale_heatmaps_list = []
        scale_posemaps_list = []
        reversed_scale_list = []

        for idx, s in enumerate(sorted(test_scale_factor, reverse=True)):
            image_resized = aug_data[idx].to(img.device)

            features = self.backbone(image_resized)
            if self.with_keypoint:
                outputs = self.keypoint_head(features)

            heatmaps, offsets = outputs
            posemaps = offset_to_pose(offsets, flip=False)
            heatmaps, posemaps = to_list(heatmaps, posemaps)

            if self.test_cfg.get('flip_test', True):
                # use flip test
                features_flipped = self.backbone(
                    torch.flip(image_resized, [3]))
                if self.with_keypoint:
                    outputs_flipped = self.keypoint_head(features_flipped)

                heatmaps_flipped, offsets_flipped = outputs_flipped
                posemaps_flipped = offset_to_pose(offsets_flipped, flip=True, flip_index=img_metas['offset_flip_index'])
                heatmaps_flipped, posemaps_flipped = to_list(heatmaps_flipped, posemaps_flipped)

                heatmaps_flipped = flip_feature_maps(heatmaps_flipped, flip_index=img_metas['flip_index'])
            else:
                heatmaps_flipped = None
                posemaps_flipped = None

            aggregated_heatmaps = aggregate_stage_flip(
                heatmaps,
                heatmaps_flipped,
                index=-1,
                project2image=False,
                size_projected=base_size,
                align_corners=self.test_cfg.get('align_corners', True),
                aggregate_stage='average',
                aggregate_flip='average')

            aggregated_posemaps = aggregate_stage_flip(
                posemaps,
                posemaps_flipped,
                index=-1,
                project2image=False,
                size_projected=base_size,
                align_corners=self.test_cfg.get('align_corners', True),
                aggregate_stage='average',
                aggregate_flip='average')

            if isinstance(aggregated_heatmaps, list):
                scale_heatmaps_list.extend(aggregated_heatmaps)
            else:
                scale_heatmaps_list.append(aggregated_heatmaps)
            if isinstance(aggregated_posemaps, list):
                scale_posemaps_list.extend(aggregated_posemaps)
            else:
                scale_posemaps_list.append(aggregated_posemaps)
            reversed_scale = image_resized.size(2) / aggregated_heatmaps[0].size(2) / s
            reversed_scale_list.append(reversed_scale)

        poses, scores = self.parser.parse(scale_heatmaps_list, scale_posemaps_list, reversed_scale_list,
                                          test_scale_factor)
        preds = []
        for persons in poses:
            for p in persons:
                p[:, :2] = transform_preds(p[:, :2], center, scale, base_size)
                preds.append(p)

        image_paths = [img_metas['image_file']]

        if return_heatmap:
            output_heatmap = scale_heatmaps_list[0].detach().cpu().numpy()
        else:
            output_heatmap = None

        result['preds'] = preds
        result['scores'] = scores
        result['image_paths'] = image_paths
        result['output_heatmap'] = output_heatmap

        return result

    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color=None,
                    pose_kpt_color=None,
                    pose_link_color=None,
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized image only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        img_h, img_w, _ = img.shape

        pose_result = []
        bbox_result = []
        for res in result:
            pose_result.append(res['keypoints'])
            if 'bbox' in res:
                bbox_result.append(res['bbox'])

        imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                         pose_kpt_color, pose_link_color, radius, thickness, show_keypoint_weight=show_keypoint_weight)
        if len(bbox_result) and bbox_color is not None:
            bbox_result = np.stack(bbox_result, axis=0)
            imshow_bboxes(img, bbox_result, colors=bbox_color, thickness=thickness, show=False)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img


def to_list(*args):
    if len(args) > 1:
        return [[arg] if not isinstance(arg, (list, tuple)) else arg for arg in args]
    else:
        args = args[0]
        return [args] if not isinstance(args, (list, tuple)) else args


def offset_to_pose(offset, flip=True, flip_index=None):
    num_offset, h, w = offset.shape[1:]
    num_joints = int(num_offset / 2)
    reg_poses = get_reg_poses(offset[0], num_joints)

    if flip:
        reg_poses = reg_poses[:, flip_index, :]
        reg_poses[:, :, 0] = w - reg_poses[:, :, 0] - 1

    reg_poses = reg_poses.contiguous().view(h * w, 2 * num_joints).permute(1, 0)
    reg_poses = reg_poses.contiguous().view(1, -1, h, w).contiguous()

    return reg_poses


def get_reg_poses(offset, num_joints):
    _, h, w = offset.shape
    offset = offset.permute(1, 2, 0).reshape(h * w, num_joints, 2)
    locations = get_locations(h, w, offset.device)
    locations = locations[:, None, :].expand(-1, num_joints, -1)
    poses = locations - offset

    return poses


def get_locations(output_h, output_w, device):
    shifts_x = torch.arange(
        0, output_w, step=1,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, output_h, step=1,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1)

    return locations
