import numpy as np
import torch
import random

from ..builder import BBOX_SAMPLERS
from .random_sampler import RandomSampler
from .sampling_result import SamplingResult


@BBOX_SAMPLERS.register_module()
class ClassBalancedPosSampler(RandomSampler):
    """Instance balanced sampler that samples equal number of positive samples
    for each instance."""

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 labels=None,
                 **kwargs):
        from mmdet.core.bbox import demodata
        super(RandomSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                            add_gt_as_proposals)
        self.rng = demodata.ensure_rng(kwargs.get('rng', None))
        self.labels = labels

    # def _sample_pos(self, assign_result, num_expected, **kwargs):
    #     """Randomly sample some positive samples."""
    #     # print(assign_result.gt_inds[assign_result.gt_inds > 0])
    #     pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
    #     # if assign_result.labels is not None:
    #     #     print(assign_result.gt_inds.shape, assign_result.labels.shape)
    #     if pos_inds.numel() != 0:
    #         pos_inds = pos_inds.squeeze(1)
    #     if pos_inds.numel() <= num_expected:
    #         return pos_inds
    #     else:
    #         return self.random_choice(pos_inds, num_expected)
    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.

        Example:
            >>> from mmdet.core.bbox import RandomSampler
            >>> from mmdet.core.bbox import AssignResult
            >>> from mmdet.core.bbox.demodata import ensure_rng, random_boxes
            >>> rng = ensure_rng(None)
            >>> assign_result = AssignResult.random(rng=rng)
            >>> bboxes = random_boxes(assign_result.num_preds, rng=rng)
            >>> gt_bboxes = random_boxes(assign_result.num_gts, rng=rng)
            >>> gt_labels = None
            >>> self = RandomSampler(num=32, pos_fraction=0.5, neg_pos_ub=-1,
            >>>                      add_gt_as_proposals=False)
            >>> self = self.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        """
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        bboxes = bboxes[:, :4]

        gt_flags = bboxes.new_zeros((bboxes.shape[0],), dtype=torch.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, labels=self.labels, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()

        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result

    def _sample_pos(self, assign_result, num_expected, labels, **kwargs):
        """Sample positive boxes.

        Args:
            assign_result (:obj:`AssignResult`): The assigned results of boxes.
            num_expected (int): The number of expected positive samples

        Returns:
            Tensor or ndarray: sampled indices.
        """
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        elif assign_result.labels is None:
            # print(pos_inds)
            return pos_inds
        else:
            sampled_inds = []
            for label in labels:
                sampled_inds.append(torch.nonzero(assign_result.labels == label, as_tuple=False).squeeze(1))
            sampled_inds = torch.cat(sampled_inds)
            if len(sampled_inds) == 0:
                return sampled_inds
            # if len(sampled_inds) < num_expected:
            #     num_extra = (num_expected - len(sampled_inds)) // len(sampled_inds) + 2
            #     sampled_inds = sampled_inds.repeat([num_extra])
            #     sampled_inds = self.random_choice(sampled_inds, num_expected)
            if len(sampled_inds) < num_expected:
                num_extra = num_expected - len(sampled_inds)
                # print(pos_inds, sampled_inds)
                extra_inds = np.array(
                    list(set(pos_inds.cpu()) - set(sampled_inds.cpu())))
                if len(extra_inds) > num_extra:
                    extra_inds = self.random_choice(extra_inds, num_extra)
                extra_inds = torch.from_numpy(extra_inds).to(
                    assign_result.gt_inds.device).long()
                # print(sampled_inds.shape, extra_inds.shape)
                sampled_inds = torch.cat([sampled_inds, extra_inds])
            if len(sampled_inds) > num_expected:
                sampled_inds = self.random_choice(sampled_inds, num_expected)
            return sampled_inds
