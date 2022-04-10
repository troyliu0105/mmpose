from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

logger = logging.getLogger(__name__)


class HeatmapLoss(nn.Module):
    """Accumulate the heatmap loss for each image in the batch.

    Args:
        supervise_empty (bool): Whether to supervise empty channels.
    """

    def __init__(self, supervise_empty=True):
        super(HeatmapLoss, self).__init__()
        self.supervise_empty = supervise_empty

    def forward(self, pred, gt, mask):
        """
        Note:
            batch_size: N
            heatmaps weight: W
            heatmaps height: H
            max_num_people: M
            num_keypoints: K
        Args:
            pred (torch.Tensor[NxKxHxW]):heatmap of output.
            gt (torch.Tensor[NxKxHxW]): target heatmap.
            mask (torch.Tensor[NxHxW]): mask of target.
        """
        assert pred.size() == gt.size(
        ), f'pred.size() is {pred.size()}, gt.size() is {gt.size()}'

        if not self.supervise_empty:
            empty_mask = (gt.sum(dim=[2, 3], keepdim=True) > 0).float()
            loss = ((pred - gt) ** 2) * empty_mask.expand_as(pred) * mask
        else:
            loss = ((pred - gt) ** 2) * mask
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
        return loss


class OffsetsLoss(nn.Module):
    def __init__(self):
        super(OffsetsLoss, self).__init__()

    def forward(self, pred, gt, weights):
        assert pred.size() == gt.size()
        num_pos = torch.nonzero(weights > 0).size()[0]
        loss = F.smooth_l1_loss(pred, gt, reduction='none', beta=1. / 9) * weights
        if num_pos == 0:
            num_pos = 1.
        loss = loss.sum() / num_pos
        return loss


@LOSSES.register_module()
class DEKRMultiLossFactory(nn.Module):
    def __init__(self,
                 num_joints,
                 num_stages,
                 heatmaps_loss_factor,
                 offset_loss_factor,
                 bg_weight,
                 supervise_empty=True):
        super().__init__()
        assert isinstance(heatmaps_loss_factor, float), \
            'heatmaps_loss_factor should be a float'
        assert isinstance(heatmaps_loss_factor, float), \
            'heatmaps_loss_factor should be a float'

        self.num_joints = num_joints
        self.num_stages = num_stages
        self.heatmaps_loss_factor = heatmaps_loss_factor
        self.offset_loss_factor = offset_loss_factor

        self.bg_weight = bg_weight

        self.heatmap_loss = HeatmapLoss(supervise_empty=supervise_empty)
        self.offset_loss = OffsetsLoss()

    def forward(self, output, poffset, heatmap, mask, offset, offset_w):
        if self.heatmap_loss:
            heatmap_loss = self.heatmap_loss(output, heatmap, mask)
            heatmap_loss = heatmap_loss * self.heatmaps_loss_factor
        else:
            heatmap_loss = None

        if self.offset_loss:
            offset_loss = self.offset_loss(poffset, offset, offset_w)
            offset_loss = offset_loss * self.offset_loss_factor
        else:
            offset_loss = None

        return heatmap_loss, offset_loss
