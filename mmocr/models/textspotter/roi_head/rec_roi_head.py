# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models import StandardRoIHead
from mmdet.models.builder import HEADS, build_roi_extractor

from mmocr.core.bbox.mixer import Mixer
from mmocr.models.builder import build_assigner, build_recognizer
from .text_mixins import RecTestMixin


@HEADS.register_module()
class RecRoIHead(StandardRoIHead, RecTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 recog_roi_extractor=None,
                 recog_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        # super().__init__(
        #     bbox_roi_extractor=bbox_roi_extractor,
        #     bbox_head=bbox_head,
        #     train_cfg=train_cfg,
        #     test_cfg=test_cfg,
        #     pretrained=pretrained,
        #     init_cfg=init_cfg)
        self.init_assigner()
        self.init_mixer()
        self.train_cfg = train_cfg
        self.train_cfg = test_cfg
        self.data_mixer = Mixer()
        if recog_roi_extractor is not None:
            self.init_recognition_roi_extractor_head(recog_roi_extractor,
                                                     recog_head)

    @property
    def with_recog(self):
        """bool: whether the RoI head contains a `recognition head`."""
        return hasattr(self,
                       'recognition_head') and self.recog_head is not None

    def init_assigner(self):
        """Initialize assigner."""
        if self.train_cfg and self.train_cfg.recog_assigner:
            self.recog_assigner = build_assigner(self.train_cfg.recog_assigner)

    def init_mixer(self):
        """Initialize mixer."""
        mixer_cfg = {}
        if self.train_cfg:
            mixer_cfg = self.train_cfg.get('recog_mixer', {})
        self.recog_mixer = Mixer(**mixer_cfg)

    def init_recog_head(self, recog_roi_extractor, recog_head):
        """Initialize ``recognition_head``"""
        if recog_roi_extractor is not None:
            self.recog_roi_extractor = build_roi_extractor(recog_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.recog_roi_extractor = self.bbox_roi_extractor
        self.recog_head = build_recognizer(recog_head)

    # def forward_dummy(self, x, proposals):
    #     """Dummy forward function."""
    #     # bbox head
    #     outs = ()
    #     rois = bbox2roi([proposals])
    #     if self.with_bbox:
    #         bbox_results = self._bbox_forward(x, rois)
    #         outs = outs + (bbox_results['cls_score'],
    #                        bbox_results['bbox_pred'])
    #     # mask head
    #     if self.with_mask:
    #         mask_rois = rois[:100]
    #         mask_results = self._mask_forward(x, mask_rois)
    #         outs = outs + (mask_results['mask_pred'], )
    #     # recognition head
    #     if self.with_recognition:
    #         recognition_rois = rois[:100]
    #         recognition_results = self._recognition_forward(
    #             x, recognition_rois)
    #         outs = outs + (recognition_results['recognition_pred'], )
    #     return outs

    def _recog_forward_train(self, x, img_metas, det_results):
        """Run forward function and calculate loss for box head in training."""
        # pos_inds = []

        # pos_rois, pos_inds = None, None
        # if not self.share_roi_extractor:
        # pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        # else:
        img_metas_list = []
        for i in range(len(det_results)):
            gt_inds = None
            if self.recog_assigner is not None:
                gt_inds = self.recog_assigner(det_results[i], img_metas[i])
            img_metas_list.append(
                self.recog_mixer.mix_gt_pred(img_metas[i], det_results[i],
                                             gt_inds))

        new_img_metas, idx_mapping = self.recog_mixer.cat_img_metas(
            img_metas_list)
        new_x = self.recog_roi_extractor(x, new_img_metas, idx_mapping)

        loss = self.recog_head.forward_train(new_x, new_img_metas)
        return loss

    # def _recognition_forward(self):
    #     pass

    def simple_test(self, x, img_metas, det_results):
        assert self.with_recog, 'Recognition head must be implemented.'
        recognition_results = self.simple_test_rec(x, img_metas, det_results)

        return recognition_results

    def simple_test_rec(self, x, img_metas, det_results):

        img_metas_list = []
        for i in range(len(det_results)):
            img_metas_list.append(
                self.recog_mixer.mix_gt_pred(
                    img_metas[i], det_results[i], testing=True))

        new_img_metas, idx_mapping = self.recog_mixer.cat_img_metas(
            img_metas_list)
        new_x = self.recog_roi_extractor(x, new_img_metas, idx_mapping)

        recog_results = self.recog_head.simple_test(new_x, img_metas)
        # return dict(recog_results=recog_results)
        return recog_results

    def forward_train(
        self,
        x,
        img_metas,
        det_results,
    ):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            det_results (list[dict]): list of detection results.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        losses = dict()
        if self.with_recognition:
            # if locals().get('bbox_results', None):
            #     bbox_feats = bbox_results['bbox_feats']
            # else:
            #     bbox_feats = None
            recog_loss = self._recog_forward_train(x, img_metas, det_results)

            losses.update(recog_loss)

        return losses
