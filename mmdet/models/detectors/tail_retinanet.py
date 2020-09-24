from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from mmdet.core import bbox2result
import torch


@DETECTORS.register_module()
class TailRetinaNet(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 labels=None,
                 labels_tail=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TailRetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                            test_cfg, pretrained)
        self.labels = labels
        self.labels_tail = labels_tail

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            np.ndarray: proposals
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list, bbox_list_tail = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        for bboxes, labels in bbox_list:
            det_bboxes = bboxes
            det_labels = labels
        if self.labels is not None:
            inds = []
            for label in self.labels:
                inds.append(torch.nonzero(det_labels == label, as_tuple=False).squeeze(1))
            inds = torch.cat(inds)
            det_bboxes = det_bboxes[inds]
            det_labels = det_labels[inds]
        for bboxes, labels in bbox_list_tail:
            det_bboxes_tail = bboxes
            det_labels_tail = labels
        # print(torch.max(det_labels_tail))
        if self.labels_tail is not None:
            inds = []
            for label in self.labels_tail:
                inds.append(torch.nonzero(det_labels_tail == label, as_tuple=False).squeeze(1))
            inds = torch.cat(inds)
            det_bboxes_tail = det_bboxes_tail[inds]
            det_labels_tail = det_labels_tail[inds]
        det_bboxes = torch.cat((det_bboxes, det_bboxes_tail))
        det_labels = torch.cat((det_labels, det_labels_tail))
        return bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
