# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models import BaseDetector

from mmocr.models.builder import build_backbone, build_head, build_neck


class TwoStageTextSpotter(BaseDetector):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 det_head=None,
                 rec_roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(TwoStageTextSpotter, self).__init__(init_cfg)

        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if det_head is not None:
            det_train_cfg = None if train_cfg is None else \
                train_cfg.get('det', None)
            det_test_cfg = None if test_cfg is None else \
                test_cfg.get('det', None)
            det_head_ = det_head.copy()
            det_head_.update(train_cfg=det_train_cfg, test_cfg=det_test_cfg)
            self.det_head = build_head(det_head_)

        if rec_roi_head is not None:
            rec_roi_train_cfg = None if train_cfg is None else \
                train_cfg.get('rec_roi', None)
            rec_roi_test_cfg = None if test_cfg is None else \
                test_cfg.get('rec_roi', None)
            rec_roi_head_ = rec_roi_head.copy()
            rec_roi_head_.update(
                train_cfg=rec_roi_train_cfg, test_cfg=rec_roi_test_cfg)
            self.rec_roi_head = build_head(rec_roi_head_)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_det_head(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'det_head') and self.det_head is not None

    @property
    def with_rec_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'rec_roi_head') and self.rec_roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, img_metas, **kwargs):
        x = self.extract_feat(img)

        losses = dict()

        outs = self.det_head(x)
        det_losses = self.det_head.loss(outs, **kwargs)
        losses.update(det_losses)

        # if use_gt:
        #     proposal_list = [None for _ in range(len(img_metas))]
        # else:
        #     proposal_list = self.bbox_head.get_boundary(preds)
        det_results = self.det_head.get_boundaries(outs, img_metas)

        roi_losses = self.rec_roi_head.forward_train(x, img_metas, det_results)
        losses.update(roi_losses)
        return losses

    def simple_test(self, img, img_metas, rescale=False):

        x = self.extract_feat(img)
        outs = self.det_head(x)
        det_results = self.det_head.get_boundaries(outs, img_metas)

        return self.rec_roi_head.simple_test(x, img_metas, det_results)
