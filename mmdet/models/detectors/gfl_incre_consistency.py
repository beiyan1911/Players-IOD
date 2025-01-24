# coding=utf-8
from ..builder import DETECTORS
from .single_stage import SingleStageDetector

import os
import torch
import warnings
import mmcv
import numpy as np
from collections import OrderedDict
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint, load_state_dict
from mmcv.parallel import MMDistributedDataParallel
from mmdet.core import distance2bbox
import torch.nn.functional as F

try:
    import sklearn.mixture as skm
except ImportError:
    skm = None


@DETECTORS.register_module()
class GFLIncreConsistency(SingleStageDetector):
    """Incremental object detector based on GFL.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ori_config_file=None,
                 ori_checkpoint_file=None,
                 ori_num_classes=40,
                 top_k=100,
                 dist_loss_weight=1):
        super().__init__(backbone, neck, bbox_head, train_cfg,
                         test_cfg, pretrained)
        self.ori_checkpoint_file = ori_checkpoint_file
        self.ori_num_classes = ori_num_classes
        self.top_k = top_k
        self.dist_loss_weight = dist_loss_weight
        self.init_detector(ori_config_file, ori_checkpoint_file)

    def _load_checkpoint_for_new_model(self, checkpoint_file, map_location=None, strict=False, logger=None):
        # load ckpt
        checkpoint = torch.load(checkpoint_file, map_location=map_location)
        # get state_dict from checkpoint
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise RuntimeError(
                'No state_dict found in checkpoint file {}'.format(checkpoint_file))
        # strip prefix of state_dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k,
            v in checkpoint['state_dict'].items()}
        # modify cls head size of state_dict
        added_branch_weight = self.bbox_head.gfl_cls.weight[self.ori_num_classes:, ...]
        added_branch_bias = self.bbox_head.gfl_cls.bias[self.ori_num_classes:, ...]
        state_dict['bbox_head.gfl_cls.weight'] = torch.cat(
            (state_dict['bbox_head.gfl_cls.weight'], added_branch_weight), dim=0)
        state_dict['bbox_head.gfl_cls.bias'] = torch.cat(
            (state_dict['bbox_head.gfl_cls.bias'], added_branch_bias), dim=0)
        # load state_dict
        if hasattr(self, 'module'):
            load_state_dict(self.module, state_dict, strict, logger)
        else:
            load_state_dict(self, state_dict, strict, logger)

    def init_detector(self, config, checkpoint_file):
        """Initialize detector from config file.

        Args:
            config (str): Config file path or the config
                object.
            checkpoint_file (str): Checkpoint path. If left as None, the model
                will not load any weights.

        Returns:
            nn.Module: The constructed detector.
        """
        assert os.path.isfile(checkpoint_file), '{} is not a valid file'.format(checkpoint_file)
        ##### init original model & frozen it #####
        # build model
        cfg = mmcv.Config.fromfile(config)
        cfg.model.pretrained = None
        cfg.model.bbox_head.num_classes = self.ori_num_classes
        ori_model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))  # test_cfg=cfg.test_cfg
        # load checkpoint
        load_checkpoint(ori_model, checkpoint_file)
        # set to eval mode
        ori_model.eval()
        ori_model.forward = ori_model.forward_dummy
        # set requires_grad of all parameters to False
        for param in ori_model.parameters():
            param.requires_grad = False

        ##### init original branchs of new model #####
        self._load_checkpoint_for_new_model(checkpoint_file)

        self.ori_model = ori_model

    def model_forward(self, img):
        """Inference image(s) with the detector.

        Args:
            model (nn.Module): The loaded detector.
            img (Tensor): Input to the model.

        Returns:
            outs (Tuple(List[Tensor])): Three model outputs.
                # cls_scores (List[Tensor]): Classification scores for each FPN level.
                # bbox_preds (List[Tensor]): BBox predictions for each FPN level.
                # centernesses (List[Tensor]): Centernesses predictions for each FPN level.
        """
        # forward the model without gradients
        with torch.no_grad():
            outs = self.ori_model(img)

        return outs

    def gmm_separation_scheme(self, gmm_assignment, scores, gmm_inds):
        """A general separation scheme for gmm model.

        It separates a GMM distribution of candidate samples into three
        parts, 0 1 and uncertain areas, and you can implement other
        separation schemes by rewriting this function.

        Args:
            gmm_assignment (Tensor): The prediction of GMM which is of shape
                (num_samples,). The 0/1 value indicates the distribution
                that each sample comes from.
            scores (Tensor): The probability of sample coming from the
                fit GMM distribution. The tensor is of shape (num_samples,).
            gmm_inds (Tensor): All the indexes of samples which are used
                to fit GMM model. The tensor is of shape (num_samples,)

        Returns:
            tuple[Tensor]: The indices of positive and ignored samples.

                - pos_inds_temp (Tensor): Indices of positive samples.
                - ignore_inds_temp (Tensor): Indices of ignore samples.
        """
        fgs = gmm_assignment == 1
        other_fgs = gmm_assignment == 0
        pos_inds_temp = fgs.new_tensor([], dtype=torch.long)
        ignore_inds_temp = fgs.new_tensor([], dtype=torch.long)
        if fgs.nonzero().numel():
            pos_inds_temp = gmm_inds[fgs]
            ignore_inds_temp = gmm_inds[other_fgs]
        return pos_inds_temp, ignore_inds_temp

    def gmm_operators(self, gmm_input, gmm_input_idx, show_ignore=False):
        device = gmm_input.device
        if len(gmm_input) < 2:  # add valid for empty
            # print('****************************')
            empty_res = torch.zeros(0, dtype=torch.long, device=gmm_input.device)
            if len(gmm_input) == 0:
                if show_ignore:
                    return empty_res, empty_res
                else:
                    return empty_res
            else:
                if show_ignore:
                    return gmm_input_idx, empty_res
                else:
                    return gmm_input_idx

        sorted_gmm, sort_inds = gmm_input.sort()
        sorted_gmm_inds = gmm_input_idx[sort_inds]
        sorted_gmm = sorted_gmm.view(-1, 1).cpu().numpy()
        min_value, max_value = sorted_gmm.min(), sorted_gmm.max()
        means_init = np.array([min_value, max_value]).reshape(2, 1)
        weights_init = np.array([0.5, 0.5])
        precisions_init = np.array([1.0, 1.0]).reshape(2, 1, 1)  # full
        # if covariance_type == 'spherical':
        #     precisions_init = precisions_init.reshape(2)
        covariance_type = 'diag'
        precisions_init = precisions_init.reshape(2, 1)
        # elif covariance_type == 'tied':
        #     precisions_init = np.array([[1.0]])

        if skm is None:
            raise ImportError('Please run "pip install sklearn" '
                              'to install sklearn first.')
        gmm = skm.GaussianMixture(
            2,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            covariance_type=covariance_type)
        gmm.fit(sorted_gmm)
        gmm_assignment = gmm.predict(sorted_gmm)
        scores = gmm.score_samples(sorted_gmm)
        gmm_assignment = torch.from_numpy(gmm_assignment).to(device)
        scores = torch.from_numpy(scores).to(device)
        sel_inds, ignore_inds = self.gmm_separation_scheme(
            gmm_assignment, scores, sorted_gmm_inds)
        if show_ignore:
            return sel_inds, ignore_inds
        else:
            return sel_inds

    def sel_pos(self, cls_scores, bbox_preds):
        """Select positive predictions based on classification scores.

        Args:
            model (nn.Module): The loaded detector.
            cls_scores (List[Tensor]): Classification scores for each FPN level.
            bbox_preds (List[Tensor]): BBox predictions for each FPN level.
            #centernesses (List[Tensor]): Centernesses predictions for each FPN level.

        Returns:
            cat_cls_scores (Tensor): FPN concatenated classification scores.
            #cat_centernesses (Tensor): FPN concatenated centernesses.
            topk_bbox_preds (Tensor): Selected top-k bbox predictions.
            topk_inds (Tensor): Selected top-k indices.
        """
        # assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)
        cat_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.ori_model.bbox_head.cls_out_channels)
            for cls_score in cls_scores
        ]
        cat_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 68)  # ori:4
            for bbox_pred in bbox_preds
        ]

        cat_cls_scores = torch.cat(cat_cls_scores, dim=1)
        cat_bbox_preds = torch.cat(cat_bbox_preds, dim=1)

        cat_conf = cat_cls_scores.sigmoid()


        min_thr = 0.05
        max_scores, _ = cat_conf.max(dim=-1)
        valid_mask = max_scores > min_thr

        # ===>  cls and box metric
        # max_bbox, _ = cat_bbox_preds.max(dim=-1)
        cons_cls = max_scores

        pre_shape = cat_bbox_preds.shape[:2]
        cat_bbox_preds_softmax = cat_bbox_preds.reshape(pre_shape + (4, -1)).softmax(-1)
        cons_box, _ = (-cat_bbox_preds_softmax * cat_bbox_preds_softmax.log2()).sum(-1).max(-1)
        cons_box = -torch.log((1 - max_scores).clamp(min=1e-2)) * cons_box


        # ===>  first images
        # cls
        cand_area_inds_0 = valid_mask[0].nonzero(as_tuple=False).squeeze(1)
        sel_cls_inds_0_first, sel_cls_inds_0_second = self.gmm_operators(cons_cls[0][cand_area_inds_0],
                                                                         cand_area_inds_0, show_ignore=True)
        sel_cls_0_first = cat_cls_scores[0].gather(  # shape:(N,dim)
            0, sel_cls_inds_0_first.unsqueeze(-1).expand(-1, cat_cls_scores[0].size(-1)))
        sel_cls_0_second = cat_cls_scores[0].gather(  # shape:(N,dim)
            0, sel_cls_inds_0_second.unsqueeze(-1).expand(-1, cat_cls_scores[0].size(-1)))

        # bbox
        sel_bbox_inds_0_first, sel_bbox_inds_0_second = self.gmm_operators(cons_box[0][cand_area_inds_0],
                                                                           cand_area_inds_0, show_ignore=True)

        # ===>  second images
        cand_area_inds_1 = valid_mask[1].nonzero(as_tuple=False).squeeze(1)
        sel_cls_inds_1_first, sel_cls_inds_1_second = self.gmm_operators(cons_cls[1][cand_area_inds_1],
                                                                         cand_area_inds_1, show_ignore=True)
        sel_cls_1_first = cat_cls_scores[1].gather(  # shape:(N,dim)
            0, sel_cls_inds_1_first.unsqueeze(-1).expand(-1, cat_cls_scores[1].size(-1)))
        sel_cls_1_second = cat_cls_scores[1].gather(  # shape:(N,dim)
            0, sel_cls_inds_1_second.unsqueeze(-1).expand(-1, cat_cls_scores[1].size(-1)))

        # bbox
        sel_bbox_inds_1_first, sel_bbox_inds_1_second = self.gmm_operators(cons_box[1][cand_area_inds_1],
                                                                           cand_area_inds_1, show_ignore=True)

        sel_cls_first = torch.cat((sel_cls_0_first, sel_cls_1_first), 0)
        sel_cls_second = torch.cat((sel_cls_0_second, sel_cls_1_second), 0)



        out = (sel_cls_first, sel_cls_second, sel_cls_inds_0_first, sel_cls_inds_0_second, sel_cls_inds_1_first,
               sel_cls_inds_1_second, sel_bbox_inds_0_first, sel_bbox_inds_0_second, sel_bbox_inds_1_first,
               sel_bbox_inds_1_second)
        return out

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels):
        # get original model outputs
        ori_outs = self.model_forward(img)
        ori_outs_head_tower = ori_outs[2:]
        ori_outs = ori_outs[:2]

        # select positive predictions from original model
        sel_cls_first, sel_cls_second, sel_cls_inds_0_first, sel_cls_inds_0_second, sel_cls_inds_1_first, sel_cls_inds_1_second, sel_bbox_inds_0_first, sel_bbox_inds_0_second, sel_bbox_inds_1_first, sel_bbox_inds_1_second = self.sel_pos(
            *ori_outs)

        # get new model outputs
        x = self.extract_feat(img)

        # outs = self.bbox_head(x)
        # outs_head_tower = outs[1]
        # outs = outs[0]
        outs = self.bbox_head(x)
        outs_head_tower = self.bbox_head.forward_for_tower_feature(x)

        # get original model neck outputs
        ori_outs_neck = self.ori_model.extract_feat(img)

        # get new model backbone outputs
        new_outs_backbone = self.backbone(img)
        # get new model neck outputs
        new_outs_neck = self.neck(new_outs_backbone)

        # calculate losses including general losses of new model and distillation losses of original model
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas) + \
                      (sel_cls_first, sel_cls_second, sel_cls_inds_0_first, sel_cls_inds_0_second, sel_cls_inds_1_first,
                       sel_cls_inds_1_second, sel_bbox_inds_0_first, sel_bbox_inds_0_second, sel_bbox_inds_1_first,
                       sel_bbox_inds_1_second,
                       self.ori_num_classes, self.dist_loss_weight, self, ori_outs_head_tower, outs_head_tower,
                       ori_outs_neck, new_outs_neck, ori_outs)

        losses = self.bbox_head.loss(*loss_inputs)
        return losses
