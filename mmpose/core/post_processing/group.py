# ------------------------------------------------------------------------------
# Adapted from https://github.com/princeton-vl/pose-ae-train/
# Original licence: Copyright (c) 2017, umich-vl, under BSD 3-Clause License.
# ------------------------------------------------------------------------------
from typing import List

import numpy as np
import torch
from munkres import Munkres

from mmpose.core.evaluation import post_dark_udp


def _py_max_match(scores):
    """Apply munkres algorithm to get the best match.

    Args:
        scores(np.ndarray): cost matrix.

    Returns:
        np.ndarray: best match.
    """
    m = Munkres()
    tmp = m.compute(scores)
    tmp = np.array(tmp).astype(int)
    return tmp


def _match_by_tag(inp, params):
    """Match joints by tags. Use Munkres algorithm to calculate the best match
    for keypoints grouping.

    Note:
        number of keypoints: K
        max number of people in an image: M (M=30 by default)
        dim of tags: L
            If use flip testing, L=2; else L=1.

    Args:
        inp(tuple):
            tag_k (np.ndarray[KxMxL]): tag corresponding to the
                top k values of feature map per keypoint.
            loc_k (np.ndarray[KxMx2]): top k locations of the
                feature maps for keypoint.
            val_k (np.ndarray[KxM]): top k value of the
                feature maps per keypoint.
        params(Params): class Params().

    Returns:
        np.ndarray: result of pose groups.
    """
    assert isinstance(params, _Params), 'params should be class _Params()'

    tag_k, loc_k, val_k = inp

    default_ = np.zeros((params.num_joints, 3 + tag_k.shape[2]),
                        dtype=np.float32)

    joint_dict = {}
    tag_dict = {}
    for i in range(params.num_joints):
        idx = params.joint_order[i]

        tags = tag_k[idx]
        joints = np.concatenate((loc_k[idx], val_k[idx, :, None], tags), 1)
        mask = joints[:, 2] > params.detection_threshold
        tags = tags[mask]  # shape: [M, L]
        joints = joints[mask]  # shape: [M, 3 + L], 3: x, y, val

        if joints.shape[0] == 0:
            continue

        if i == 0 or len(joint_dict) == 0:
            for tag, joint in zip(tags, joints):
                key = tag[0]
                joint_dict.setdefault(key, np.copy(default_))[idx] = joint
                tag_dict[key] = [tag]
        else:
            # shape: [M]
            grouped_keys = list(joint_dict.keys())
            if params.ignore_too_much:
                grouped_keys = grouped_keys[:params.max_num_people]
            # shape: [M, L]
            grouped_tags = [np.mean(tag_dict[i], axis=0) for i in grouped_keys]

            # shape: [M, M, L]
            diff = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]
            # shape: [M, M]
            diff_normed = np.linalg.norm(diff, ord=2, axis=2)
            diff_saved = np.copy(diff_normed)

            if params.use_detection_val:
                diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]

            num_added = diff.shape[0]
            num_grouped = diff.shape[1]

            if num_added > num_grouped:
                diff_normed = np.concatenate(
                    (diff_normed,
                     np.zeros((num_added, num_added - num_grouped),
                              dtype=np.float32) + 1e10),
                    axis=1)

            pairs = _py_max_match(diff_normed)
            for row, col in pairs:
                if (row < num_added and col < num_grouped
                        and diff_saved[row][col] < params.tag_threshold):
                    key = grouped_keys[col]
                    joint_dict[key][idx] = joints[row]
                    tag_dict[key].append(tags[row])
                else:
                    key = tags[row][0]
                    joint_dict.setdefault(key, np.copy(default_))[idx] = \
                        joints[row]
                    tag_dict[key] = [tags[row]]

    joint_dict_keys = list(joint_dict.keys())
    if params.ignore_too_much:
        # The new person joints beyond the params.max_num_people will be
        # ignored, for the dict is in ordered when python > 3.6 version.
        joint_dict_keys = joint_dict_keys[:params.max_num_people]
    results = np.array([joint_dict[i]
                        for i in joint_dict_keys]).astype(np.float32)
    return results


class _Params:
    """A class of parameter.

    Args:
        cfg(Config): config.
    """

    def __init__(self, cfg):
        self.num_joints = cfg['num_joints']
        self.max_num_people = cfg['max_num_people']

        self.detection_threshold = cfg['detection_threshold']
        self.tag_threshold = cfg['tag_threshold']
        self.use_detection_val = cfg['use_detection_val']
        self.ignore_too_much = cfg['ignore_too_much']

        if self.num_joints == 17:
            self.joint_order = [
                i - 1 for i in
                [1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17]
            ]
        else:
            self.joint_order = list(np.arange(self.num_joints))


class HeatmapParser:
    """The heatmap parser for post processing."""

    def __init__(self, cfg):
        self.params = _Params(cfg)
        self.tag_per_joint = cfg['tag_per_joint']
        self.pool = torch.nn.MaxPool2d(cfg['nms_kernel'], 1,
                                       cfg['nms_padding'])
        self.use_udp = cfg.get('use_udp', False)
        self.score_per_joint = cfg.get('score_per_joint', False)

    def nms(self, heatmaps):
        """Non-Maximum Suppression for heatmaps.

        Args:
            heatmap(torch.Tensor): Heatmaps before nms.

        Returns:
            torch.Tensor: Heatmaps after nms.
        """

        maxm = self.pool(heatmaps)
        maxm = torch.eq(maxm, heatmaps).float()
        heatmaps = heatmaps * maxm

        return heatmaps

    def match(self, tag_k, loc_k, val_k):
        """Group keypoints to human poses in a batch.

        Args:
            tag_k (np.ndarray[NxKxMxL]): tag corresponding to the
                top k values of feature map per keypoint.
            loc_k (np.ndarray[NxKxMx2]): top k locations of the
                feature maps for keypoint.
            val_k (np.ndarray[NxKxM]): top k value of the
                feature maps per keypoint.

        Returns:
            list
        """

        def _match(x):
            return _match_by_tag(x, self.params)

        return list(map(_match, zip(tag_k, loc_k, val_k)))

    def top_k(self, heatmaps, tags):
        """Find top_k values in an image.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            max number of people: M
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmaps (torch.Tensor[NxKxHxW])
            tags (torch.Tensor[NxKxHxWxL])

        Returns:
            dict: A dict containing top_k values.

            - tag_k (np.ndarray[NxKxMxL]):
                tag corresponding to the top k values of
                feature map per keypoint.
            - loc_k (np.ndarray[NxKxMx2]):
                top k location of feature map per keypoint.
            - val_k (np.ndarray[NxKxM]):
                top k value of feature map per keypoint.
        """
        heatmaps = self.nms(heatmaps)
        N, K, H, W = heatmaps.size()
        heatmaps = heatmaps.view(N, K, -1)
        val_k, ind = heatmaps.topk(self.params.max_num_people, dim=2)

        tags = tags.view(tags.size(0), tags.size(1), W * H, -1)
        if not self.tag_per_joint:
            tags = tags.expand(-1, self.params.num_joints, -1, -1)

        tag_k = torch.stack(
            [torch.gather(tags[..., i], 2, ind) for i in range(tags.size(3))],
            dim=3)

        x = ind % W
        y = ind // W

        ind_k = torch.stack((x, y), dim=3)

        results = {
            'tag_k': tag_k.cpu().numpy(),
            'loc_k': ind_k.cpu().numpy(),
            'val_k': val_k.cpu().numpy()
        }

        return results

    @staticmethod
    def adjust(results, heatmaps):
        """Adjust the coordinates for better accuracy.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            results (list(np.ndarray)): Keypoint predictions.
            heatmaps (torch.Tensor[NxKxHxW]): Heatmaps.
        """
        _, _, H, W = heatmaps.shape
        for batch_id, people in enumerate(results):
            for people_id, people_i in enumerate(people):
                for joint_id, joint in enumerate(people_i):
                    if joint[2] > 0:
                        x, y = joint[0:2]
                        xx, yy = int(x), int(y)
                        tmp = heatmaps[batch_id][joint_id]
                        if tmp[min(H - 1, yy + 1), xx] > tmp[max(0, yy - 1),
                                                             xx]:
                            y += 0.25
                        else:
                            y -= 0.25

                        if tmp[yy, min(W - 1, xx + 1)] > tmp[yy,
                                                             max(0, xx - 1)]:
                            x += 0.25
                        else:
                            x -= 0.25
                        results[batch_id][people_id, joint_id,
                                          0:2] = (x + 0.5, y + 0.5)
        return results

    @staticmethod
    def refine(heatmap, tag, keypoints, use_udp=False):
        """Given initial keypoint predictions, we identify missing joints.

        Note:
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmap: np.ndarray(K, H, W).
            tag: np.ndarray(K, H, W) |  np.ndarray(K, H, W, L)
            keypoints: np.ndarray of size (K, 3 + L)
                        last dim is (x, y, score, tag).
            use_udp: bool-unbiased data processing

        Returns:
            np.ndarray: The refined keypoints.
        """

        K, H, W = heatmap.shape
        if len(tag.shape) == 3:
            tag = tag[..., None]

        tags = []
        for i in range(K):
            if keypoints[i, 2] > 0:
                # save tag value of detected keypoint
                x, y = keypoints[i][:2].astype(int)
                x = np.clip(x, 0, W - 1)
                y = np.clip(y, 0, H - 1)
                tags.append(tag[i, y, x])

        # mean tag of current detected people
        prev_tag = np.mean(tags, axis=0)
        results = []

        for _heatmap, _tag in zip(heatmap, tag):
            # distance of all tag values with mean tag of
            # current detected people
            distance_tag = (((_tag -
                              prev_tag[None, None, :])**2).sum(axis=2)**0.5)
            norm_heatmap = _heatmap - np.round(distance_tag)

            # find maximum position
            y, x = np.unravel_index(np.argmax(norm_heatmap), _heatmap.shape)
            xx = x.copy()
            yy = y.copy()
            # detection score at maximum position
            val = _heatmap[y, x]
            if not use_udp:
                # offset by 0.5
                x += 0.5
                y += 0.5

            # add a quarter offset
            if _heatmap[yy, min(W - 1, xx + 1)] > _heatmap[yy, max(0, xx - 1)]:
                x += 0.25
            else:
                x -= 0.25

            if _heatmap[min(H - 1, yy + 1), xx] > _heatmap[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25

            results.append((x, y, val))
        results = np.array(results)

        if results is not None:
            for i in range(K):
                # add keypoint if it is not detected
                if results[i, 2] > 0 and keypoints[i, 2] == 0:
                    keypoints[i, :3] = results[i, :3]

        return keypoints

    def parse(self, heatmaps, tags, adjust=True, refine=True):
        """Group keypoints into poses given heatmap and tag.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmaps (torch.Tensor[NxKxHxW]): model output heatmaps.
            tags (torch.Tensor[NxKxHxWxL]): model output tagmaps.

        Returns:
            tuple: A tuple containing keypoint grouping results.

            - results (list(np.ndarray)): Pose results.
            - scores (list/list(np.ndarray)): Score of people.
        """
        results = self.match(**self.top_k(heatmaps, tags))

        if adjust:
            if self.use_udp:
                for i in range(len(results)):
                    if results[i].shape[0] > 0:
                        results[i][..., :2] = post_dark_udp(
                            results[i][..., :2].copy(), heatmaps[i:i + 1, :])
            else:
                results = self.adjust(results, heatmaps)

        if self.score_per_joint:
            scores = [i[:, 2] for i in results[0]]
        else:
            scores = [i[:, 2].mean() for i in results[0]]

        if refine:
            results = results[0]
            # for every detected person
            for i in range(len(results)):
                heatmap_numpy = heatmaps[0].cpu().numpy()
                tag_numpy = tags[0].cpu().numpy()
                if not self.tag_per_joint:
                    tag_numpy = np.tile(tag_numpy,
                                        (self.params.num_joints, 1, 1, 1))
                results[i] = self.refine(
                    heatmap_numpy, tag_numpy, results[i], use_udp=self.use_udp)
            results = [results]

        return results, scores


class DEKRParser(object):
    def __init__(self, cfg):
        self.detection_threshold = cfg.get("detection_threshold", 0.1)
        self.max_num_people = cfg.get("max_num_people", 30)
        self.decrease = cfg.get("decrease", 0.8)
        self.nms_thre = cfg.get("nms_thre", 0.05)
        self.nms_num_thre = cfg.get("nms_num_thre", 7)

    def parse(self, heatmaps: List[torch.Tensor], posemaps: List[torch.Tensor],
              revsersed_scale_list: List[float], test_scale_factor: List[float]):
        poses = []
        heatmap_sum = 0
        for heatmap, posemap, reversed_scale in zip(heatmaps, posemaps, revsersed_scale_list):
            h, w = heatmap.shape[2:]
            heatmap_sum += self.up_interpolate(heatmap,
                                               size=(int(reversed_scale * h), int(reversed_scale * w)))
            # [1, H, W]
            center_heatmap = heatmap[0, -1:]
            pose_ind, ctr_score = self.get_maximum_from_heatmap(center_heatmap)
            posemap = posemap[0] \
                .permute(1, 2, 0) \
                .view(center_heatmap.shape[1] * center_heatmap.shape[2], -1, 2)
            pose = reversed_scale * posemap[pose_ind]
            ctr_score = ctr_score[:, None].expand(-1, pose.shape[-2])[:, :, None]
            poses.append(torch.cat([pose, ctr_score], dim=2))
        heatmap_avg = heatmap_sum / len(heatmaps)
        poses, scores = self.pose_nms(heatmap_avg, poses,
                                      test_scale_factor)

        return poses, scores

    @staticmethod
    def up_interpolate(x, size, mode='bilinear'):
        H = x.size()[2]
        W = x.size()[3]
        scale_h = int(size[0] / H)
        scale_w = int(size[1] / W)
        inter_x = torch.nn.functional.interpolate(x, size=[size[0] - scale_h + 1, size[1] - scale_w + 1],
                                                  align_corners=True, mode=mode)
        padd = torch.nn.ReplicationPad2d((0, scale_w - 1, 0, scale_h - 1))
        return padd(inter_x)

    @staticmethod
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

    @staticmethod
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

    def get_maximum_from_heatmap(self, heatmap):
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
        scores, pos_ind = scores.topk(self.max_num_people)

        # cfg.TEST.KEYPOINT_THRESHOLD
        select_ind = (scores > self.detection_threshold).nonzero()
        scores = scores[select_ind][:, 0]
        pos_ind = pos_ind[select_ind][:, 0]

        return pos_ind, scores

    def pose_nms(self, heatmap_avg, poses, scale_factor):
        """
        NMS for the regressed poses results.

        Args:
            heatmap_avg (Tensor): Avg of the heatmaps at all scales (1, 1+num_joints, w, h)
            poses (List): Gather of the pose proposals [(num_people, num_joints, 3)]
        """
        scale1_index = sorted(scale_factor, reverse=True).index(1.0)
        pose_norm = poses[scale1_index]
        max_score = pose_norm[:, :, 2].max() if pose_norm.shape[0] else 1

        for i, pose in enumerate(poses):
            if i != scale1_index:
                max_score_scale = pose[:, :, 2].max() if pose.shape[0] else 1
                pose[:, :, 2] = pose[:, :, 2] / max_score_scale * max_score * self.decrease

        pose_score = torch.cat([pose[:, :, 2:] for pose in poses], dim=0)
        pose_coord = torch.cat([pose[:, :, :2] for pose in poses], dim=0)

        if pose_coord.shape[0] == 0:
            return [], []

        num_people, num_joints, _ = pose_coord.shape
        heatval = self.get_heat_value(pose_coord, heatmap_avg[0])
        # [topk], 相当于每个人关键点的平均得分作为 heat_score
        heat_score = (torch.sum(heatval, dim=1) / num_joints)[:, 0]

        pose_score = pose_score * heatval
        poses = torch.cat([pose_coord.cpu(), pose_score.cpu()], dim=2)

        keep_pose_inds = self.nms_core(pose_coord, heat_score, self.nms_thre, self.nms_num_thre)
        poses = poses[keep_pose_inds]
        heat_score = heat_score[keep_pose_inds]

        if len(keep_pose_inds) > self.max_num_people:
            heat_score, topk_inds = torch.topk(heat_score, self.max_num_people)
            poses = poses[topk_inds]

        poses = [poses.numpy()]
        scores = [i[:, 2].mean() for i in poses[0]]

        return poses, scores
