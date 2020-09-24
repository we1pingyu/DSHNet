import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

from mmdet.core import (force_fp32, multiclass_nms)
from .tail_bbox_head import TailBBoxHead
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy


@HEADS.register_module
class TailGSBBoxHead(TailBBoxHead):

    def __init__(self, gs_config=None, *args, **kwargs):
        super(TailGSBBoxHead, self).__init__(*args, **kwargs)
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes + 1 + gs_config.num_bins)
            # self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes + 1)
        self.loss_bins = []
        for i in range(gs_config.num_bins):
            self.loss_bins.append(build_loss(gs_config.loss_bin))

        self.label2binlabel = torch.Tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                            [0, 1, 2, 0, 3, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 1, 0, 2, 3, 4, 5, 6, 7], ]).cuda()
        self.pred_slice = np.array([[0, 2],
                                    [2, 4],
                                    [6, 8]])
        self.fg_splits = []
        self.fg_splits.append(torch.Tensor([1, 2, 4]).cuda())
        self.fg_splits.append(torch.Tensor([3, 5, 6, 7, 8, 9, 10]).cuda())
        self.others_sample_ratio = gs_config.others_sample_ratio

    def _sample_others(self, label):

        fg = torch.where(label > 0, torch.ones_like(label),
                         torch.zeros_like(label))
        fg_idx = fg.nonzero(as_tuple=True)[0]
        fg_num = fg_idx.shape[0]
        if fg_num == 0:
            return torch.zeros_like(label)

        bg = 1 - fg
        bg_idx = bg.nonzero(as_tuple=True)[0]
        bg_num = bg_idx.shape[0]

        bg_sample_num = int(fg_num * self.others_sample_ratio)

        if bg_sample_num >= bg_num:
            weight = torch.ones_like(label)
        else:
            sample_idx = np.random.choice(bg_idx.cpu().numpy(),
                                          (bg_sample_num,), replace=False)
            sample_idx = torch.from_numpy(sample_idx).cuda()
            fg[sample_idx] = 1
            weight = fg

        return weight

    def _remap_labels(self, labels):

        num_bins = self.label2binlabel.shape[0]
        new_labels = []
        new_weights = []
        new_avg = []
        for i in range(num_bins):
            mapping = self.label2binlabel[i]
            new_bin_label = mapping[labels.type(torch.int64)]

            if i < 1:
                weight = torch.ones_like(new_bin_label)
                # weight = torch.zeros_like(new_bin_label)
            else:
                weight = self._sample_others(new_bin_label)
            new_labels.append(new_bin_label)
            new_weights.append(weight)

            avg_factor = max(torch.sum(weight).float().item(), 1.)
            new_avg.append(avg_factor)

        return new_labels, new_weights, new_avg

    def _remap_labels1(self, labels):

        num_bins = self.label2binlabel.shape[0]
        new_labels = []
        new_weights = []
        new_avg = []
        for i in range(num_bins):
            mapping = self.label2binlabel[i]
            new_bin_label = mapping[labels]

            weight = torch.ones_like(new_bin_label)

            new_labels.append(new_bin_label)
            new_weights.append(weight)

            avg_factor = max(torch.sum(weight).float().item(), 1.)
            new_avg.append(avg_factor)

        return new_labels, new_weights, new_avg

    def _slice_preds(self, cls_score):

        new_preds = []

        num_bins = self.pred_slice.shape[0]
        for i in range(num_bins):
            start = self.pred_slice[i, 0]
            length = self.pred_slice[i, 1]
            sliced_pred = cls_score.narrow(1, int(start), int(length))
            new_preds.append(sliced_pred)

        return new_preds

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()

        if cls_score is not None:
            # avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            # if cls_score.numel() > 0:
            #     losses['loss_cls'] = self.loss_cls(
            #         cls_score,
            #         labels,
            #         label_weights,
            #         avg_factor=avg_factor,
            #         reduction_override=reduction_override)
            #     losses['acc'] = accuracy(cls_score, labels)
            # Original label_weights is 1 for each roi.
            new_labels, new_weights, new_avgfactors = self._remap_labels(labels)
            new_preds = self._slice_preds(cls_score)

            num_bins = len(new_labels)
            for i in range(num_bins):
                losses['loss_cls_bin{}'.format(i)] = self.loss_bins[i](
                    new_preds[i],
                    new_labels[i].long(),
                    new_weights[i],
                    avg_factor=new_avgfactors[i],
                    reduction_override=reduction_override
                )
                losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
        return losses

    # @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    # def loss(self,
    #          cls_score,
    #          bbox_pred,
    #          rois,
    #          labels,
    #          label_weights,
    #          bbox_targets,
    #          bbox_weights,
    #          reduction_override=None):
    #     losses = dict()
    #     if cls_score is not None:
    #         avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
    #         if cls_score.numel() > 0:
    #             losses['loss_cls'] = self.loss_cls(
    #                 cls_score,
    #                 labels,
    #                 label_weights,
    #                 avg_factor=avg_factor,
    #                 reduction_override=reduction_override)
    #             losses['acc'] = accuracy(cls_score, labels)
    #     if bbox_pred is not None:
    #         bg_class_ind = self.num_classes
    #         # 0~self.num_classes-1 are FG, self.num_classes is BG
    #         pos_inds = (labels >= 0) & (labels < bg_class_ind)
    #         # do not perform bounding box regression for BG anymore.
    #         if pos_inds.any():
    #             if self.reg_decoded_bbox:
    #                 bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
    #             if self.reg_class_agnostic:
    #                 pos_bbox_pred = bbox_pred.view(
    #                     bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
    #             else:
    #                 pos_bbox_pred = bbox_pred.view(
    #                     bbox_pred.size(0), -1,
    #                     4)[pos_inds.type(torch.bool),
    #                        labels[pos_inds.type(torch.bool)]]
    #             losses['loss_bbox'] = self.loss_bbox(
    #                 pos_bbox_pred,
    #                 bbox_targets[pos_inds.type(torch.bool)],
    #                 bbox_weights[pos_inds.type(torch.bool)],
    #                 avg_factor=bbox_targets.size(0),
    #                 reduction_override=reduction_override)
    #         else:
    #             losses['loss_bbox'] = bbox_pred.sum() * 0
    #     return losses

    @force_fp32(apply_to=('cls_score'))
    def _merge_score1(self, cls_score):
        '''
        Do softmax in each bin. Merge the scores directly.
        '''
        num_proposals = cls_score.shape[0]

        new_preds = self._slice_preds(cls_score)
        new_scores = [F.softmax(pred, dim=1) for pred in new_preds]

        bg_score = new_scores[0]
        fg_score = new_scores[1:]

        fg_merge = torch.zeros((num_proposals, 1231)).cuda()
        merge = torch.zeros((num_proposals, 1231)).cuda()

        for i, split in enumerate(self.fg_splits):
            fg_merge[:, split] = fg_score[i]

        merge[:, 0] = bg_score[:, 0]
        fg_idx = (bg_score[:, 1] > 0.5).nonzero(as_tuple=True)[0]
        merge[fg_idx] = fg_merge[fg_idx]

        return merge

    @force_fp32(apply_to=('cls_score'))
    def _merge_score2(self, cls_score):
        '''
        Do softmax in each bin. Softmax again after merge.
        '''
        num_proposals = cls_score.shape[0]

        new_preds = self._slice_preds(cls_score)
        new_scores = [F.softmax(pred, dim=1) for pred in new_preds]

        bg_score = new_scores[0]
        fg_score = new_scores[1:]

        fg_merge = torch.zeros((num_proposals, 1231)).cuda()
        merge = torch.zeros((num_proposals, 1231)).cuda()

        for i, split in enumerate(self.fg_splits):
            fg_merge[:, split] = fg_score[i]

        merge[:, 0] = bg_score[:, 0]
        fg_idx = (bg_score[:, 1] > 0.5).nonzero(as_tuple=True)[0]
        merge[fg_idx] = fg_merge[fg_idx]
        merge = F.softmax(merge)

        return merge

    @force_fp32(apply_to=('cls_score'))
    def _merge_score(self, cls_score):
        '''
        Do softmax in each bin. Decay the score of normal classes
        with the score of fg.
        From v1.
        '''

        num_proposals = cls_score.shape[0]

        new_preds = self._slice_preds(cls_score)
        new_scores = [F.softmax(pred, dim=1) for pred in new_preds]

        bg_score = new_scores[0]
        fg_score = new_scores[1:]

        fg_merge = torch.zeros((num_proposals, self.num_classes + 1)).cuda()
        merge = torch.zeros((num_proposals, self.num_classes + 1)).cuda()

        # import pdb
        # pdb.set_trace()
        for i, split in enumerate(self.fg_splits):
            fg_merge[:, split.type(torch.int64)] = fg_score[i][:, 1:]

        weight = bg_score.narrow(1, 1, 1)

        # Whether we should add this? Test
        fg_merge = weight * fg_merge

        merge[:, 0] = bg_score[:, 0]
        merge[:, 1:] = fg_merge[:, 1:]
        # fg_idx = (bg_score[:, 1] > 0.5).nonzero(as_tuple=True)[0]
        # erge[fg_idx] = fg_merge[fg_idx]

        return merge

    @force_fp32(apply_to=('cls_score'))
    def _merge_score4(self, cls_score):
        '''
        Do softmax in each bin.
        Do softmax on merged fg classes.
        Decay the score of normal classes with the score of fg.
        From v2 and v3
        '''
        num_proposals = cls_score.shape[0]

        new_preds = self._slice_preds(cls_score)
        new_scores = [F.softmax(pred, dim=1) for pred in new_preds]

        bg_score = new_scores[0]
        fg_score = new_scores[1:]

        fg_merge = torch.zeros((num_proposals, 1231)).cuda()
        merge = torch.zeros((num_proposals, 1231)).cuda()

        for i, split in enumerate(self.fg_splits):
            fg_merge[:, split] = fg_score[i]

        fg_merge = F.softmax(fg_merge, dim=1)
        weight = bg_score.narrow(1, 1, 1)
        fg_merge = weight * fg_merge

        merge[:, 0] = bg_score[:, 0]
        merge[:, 1:] = fg_merge[:, 1:]
        # fg_idx = (bg_score[:, 1] > 0.5).nonzero(as_tuple=True)[0]
        # erge[fg_idx] = fg_merge[fg_idx]

        return merge

    @force_fp32(apply_to=('cls_score'))
    def _merge_score5(self, cls_score):
        '''
        Do softmax in each bin.
        Pick the bin with the max score for each box.
        '''
        num_proposals = cls_score.shape[0]

        new_preds = self._slice_preds(cls_score)
        new_scores = [F.softmax(pred, dim=1) for pred in new_preds]

        bg_score = new_scores[0]
        fg_score = new_scores[1:]
        max_scores = [s.max(dim=1, keepdim=True)[0] for s in fg_score]
        max_scores = torch.cat(max_scores, 1)
        max_idx = max_scores.argmax(dim=1)

        fg_merge = torch.zeros((num_proposals, 1231)).cuda()
        merge = torch.zeros((num_proposals, 1231)).cuda()

        for i, split in enumerate(self.fg_splits):
            tmp_merge = torch.zeros((num_proposals, 1231)).cuda()
            tmp_merge[:, split] = fg_score[i]
            roi_idx = torch.where(max_idx == i,
                                  torch.ones_like(max_idx),
                                  torch.zeros_like(max_idx)).nonzero(
                as_tuple=True)[0]
            fg_merge[roi_idx] = tmp_merge[roi_idx]

        merge[:, 0] = bg_score[:, 0]
        fg_idx = (bg_score[:, 1] > 0.5).nonzero(as_tuple=True)[0]
        merge[fg_idx] = fg_merge[fg_idx]

        return merge

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        scores = self._merge_score(cls_score)
        # scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels
