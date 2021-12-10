import warnings

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
        self.init_weights(pretrained=pretrained)

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.backbone.init_weights(pretrained)
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
        scale_poses_list = []
        poses = []
        heatmap_sum = 0

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

            reversed_scale = image_resized.size(2) / aggregated_heatmaps[0].size(2) / s

            h, w = aggregated_heatmaps[0].shape[2:]
            heatmap_sum += up_interpolate(aggregated_heatmaps[0],
                                          size=(int(reversed_scale * h), int(reversed_scale * w)))
            center_heatmap = aggregated_heatmaps[0][0, -1:]
            pose_ind, ctr_score = get_maximum_from_heatmap(center_heatmap,
                                                           self.test_cfg.get("detection_threshold", 0.1))
            posemap = aggregated_posemaps[0][0] \
                .permute(1, 2, 0) \
                .view(center_heatmap.shape[1] * center_heatmap.shape[2], -1, 2)
            pose = reversed_scale * posemap[pose_ind]
            ctr_score = ctr_score[:, None].expand(-1, pose.shape[-2])[:, :, None]
            poses.append(torch.cat([pose, ctr_score], dim=2))

            if isinstance(aggregated_heatmaps, list):
                scale_heatmaps_list.extend(aggregated_heatmaps)
            else:
                scale_heatmaps_list.append(aggregated_heatmaps)

        heatmap_avg = heatmap_sum / len(test_scale_factor)
        poses, scores = pose_nms(heatmap_avg, poses,
                                 test_scale_factor,
                                 self.test_cfg.get('max_num_people', 30))
        preds = get_final_preds(poses, center, scale, base_size)

        image_paths = []
        image_paths.append(img_metas['image_file'])

        if return_heatmap:
            output_heatmap = aggregated_heatmaps.detach().cpu().numpy()
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


def up_interpolate(x, size, mode='bilinear'):
    H = x.size()[2]
    W = x.size()[3]
    scale_h = int(size[0] / H)
    scale_w = int(size[1] / W)
    inter_x = torch.nn.functional.interpolate(x, size=[size[0] - scale_h + 1, size[1] - scale_w + 1],
                                              align_corners=True, mode=mode)
    padd = torch.nn.ReplicationPad2d((0, scale_w - 1, 0, scale_h - 1))
    return padd(inter_x)


def get_final_preds(grouped_joints, center, scale, heatmap_size):
    final_results = []
    if len(grouped_joints) == 0:
        return []
    for person in grouped_joints[0]:
        joints = np.zeros((person.shape[0], 3))
        joints = transform_preds(person, center, scale, heatmap_size)
        final_results.append(joints)

    return final_results


def transform_preds(coords, center, scale, output_size):
    def get_affine_transform(center,
                             scale,
                             rot,
                             output_size,
                             shift=np.array([0, 0], dtype=np.float32),
                             inv=0):
        def get_dir(src_point, rot_rad):
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)

            src_result = [0, 0]
            src_result[0] = src_point[0] * cs - src_point[1] * sn
            src_result[1] = src_point[0] * sn + src_point[1] * cs

            return src_result

        def get_3rd_point(a, b):
            direct = a - b
            return b + np.array([-direct[1], direct[0]], dtype=np.float32)

        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            print(scale)
            scale = np.array([scale, scale])

        scale_tmp = scale * 200.0
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def affine_transform(pt, t):
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]

    # target_coords = np.zeros(coords.shape)
    target_coords = coords.copy()
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def pose_nms(heatmap_avg, poses, scale_factor, max_people):
    """
    NMS for the regressed poses results.

    Args:
        heatmap_avg (Tensor): Avg of the heatmaps at all scales (1, 1+num_joints, w, h)
        poses (List): Gather of the pose proposals [(num_people, num_joints, 3)]
    """
    DECREASE = 0.8
    scale1_index = sorted(scale_factor, reverse=True).index(1.0)
    pose_norm = poses[scale1_index]
    max_score = pose_norm[:, :, 2].max() if pose_norm.shape[0] else 1

    for i, pose in enumerate(poses):
        if i != scale1_index:
            max_score_scale = pose[:, :, 2].max() if pose.shape[0] else 1
            pose[:, :, 2] = pose[:, :, 2] / max_score_scale * max_score * DECREASE

    pose_score = torch.cat([pose[:, :, 2:] for pose in poses], dim=0)
    pose_coord = torch.cat([pose[:, :, :2] for pose in poses], dim=0)

    if pose_coord.shape[0] == 0:
        return [], []

    num_people, num_joints, _ = pose_coord.shape
    heatval = get_heat_value(pose_coord, heatmap_avg[0])
    # [topk], 相当于每个人关键点的平均得分作为 heat_score
    heat_score = (torch.sum(heatval, dim=1) / num_joints)[:, 0]

    pose_score = pose_score * heatval
    poses = torch.cat([pose_coord.cpu(), pose_score.cpu()], dim=2)

    keep_pose_inds = nms_core(pose_coord, heat_score, 0.05, 7)
    poses = poses[keep_pose_inds]
    heat_score = heat_score[keep_pose_inds]

    if len(keep_pose_inds) > max_people:
        heat_score, topk_inds = torch.topk(heat_score, max_people)
        poses = poses[topk_inds]

    poses = [poses.numpy()]
    scores = [i[:, 2].mean() for i in poses[0]]

    return poses, scores


def get_heat_value(pose_coord, heatmap):
    kpt_now, num_joints, _ = pose_coord.shape
    heatval = torch.zeros((kpt_now, num_joints, 1), device=pose_coord.device)
    for i in range(kpt_now):
        for j in range(num_joints):
            k1, k2 = int(pose_coord[i, j, 0]), int(pose_coord[i, j, 0]) + 1
            k3, k4 = int(pose_coord[i, j, 1]), int(pose_coord[i, j, 1]) + 1
            u = pose_coord[i, j, 0] - int(pose_coord[i, j, 0])
            v = pose_coord[i, j, 1] - int(pose_coord[i, j, 1])
            if k2 < heatmap.shape[2] and k1 >= 0 \
                    and k4 < heatmap.shape[1] and k3 >= 0:
                heatval[i, j, 0] = \
                    heatmap[j, k3, k1] * (1 - v) * (1 - u) + heatmap[j, k4, k1] * (1 - u) * v + \
                    heatmap[j, k3, k2] * u * (1 - v) + heatmap[j, k4, k2] * u * v
    return heatval


def nms_core(pose_coord, heat_score, nms_thre, nms_num_thre):
    def cal_area_2_torch(v):
        w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
        h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
        return w * w + h * h

    num_people, num_joints, _ = pose_coord.shape
    pose_area = cal_area_2_torch(pose_coord)[:, None].repeat(1, num_people * num_joints)
    pose_area = pose_area.reshape(num_people, num_people, num_joints)

    pose_diff = pose_coord[:, None, :, :] - pose_coord
    pose_diff.pow_(2)
    pose_dist = pose_diff.sum(3)
    pose_dist.sqrt_()
    pose_thre = nms_thre * torch.sqrt(pose_area)
    pose_dist = (pose_dist < pose_thre).sum(2)
    nms_pose = pose_dist > nms_num_thre

    ignored_pose_inds = []
    keep_pose_inds = []
    for i in range(nms_pose.shape[0]):
        if i in ignored_pose_inds:
            continue
        keep_inds = nms_pose[i].nonzero().cpu().numpy()
        keep_inds = [list(kind)[0] for kind in keep_inds]
        keep_scores = heat_score[keep_inds]
        ind = torch.argmax(keep_scores)
        keep_ind = keep_inds[ind]
        if keep_ind in ignored_pose_inds:
            continue
        keep_pose_inds += [keep_ind]
        ignored_pose_inds += list(set(keep_inds) - set(ignored_pose_inds))

    return keep_pose_inds


def get_maximum_from_heatmap(heatmap, detection_threshold):
    def hierarchical_pool(heatmap):
        pool1 = torch.nn.MaxPool2d(3, 1, 1)
        pool2 = torch.nn.MaxPool2d(5, 1, 2)
        pool3 = torch.nn.MaxPool2d(7, 1, 3)
        map_size = (heatmap.shape[1] + heatmap.shape[2]) / 2.0
        if map_size > 300:
            maxm = pool3(heatmap[None, :, :, :])
        elif map_size > 200:
            maxm = pool2(heatmap[None, :, :, :])
        else:
            maxm = pool1(heatmap[None, :, :, :])

        return maxm

    maxm = hierarchical_pool(heatmap)
    maxm = torch.eq(maxm, heatmap).float()
    heatmap = heatmap * maxm
    scores = heatmap.view(-1)
    scores, pos_ind = scores.topk(30)

    # cfg.TEST.KEYPOINT_THRESHOLD
    select_ind = (scores > detection_threshold).nonzero()
    scores = scores[select_ind][:, 0]
    pos_ind = pos_ind[select_ind][:, 0]

    return pos_ind, scores


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
