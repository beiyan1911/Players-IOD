# coding=utf-8
import torch

from ..builder import HEADS, build_loss
from .gfl_head import GFLHead, Integral
from mmcv.runner import force_fp32
from mmdet.core import distance2bbox, bbox_overlaps, bbox2distance, reduce_mean, multi_apply
import torch.nn.functional as F
from mmcv.ops import batched_nms

INF = 1e8


@HEADS.register_module()
class GFLHeadIncreIncons(GFLHead):
    """
    Incremental GFL head for incremental object detection.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
                 loss_ld=dict(
                     type='LocalizationDistillationLoss',
                     loss_weight=0.25,
                     T=10),
                 loss_kd=None,
                 reg_max=16,
                 **kwargs):

        # reuse father's constructor
        super().__init__(num_classes, in_channels, stacked_convs=stacked_convs, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                         loss_dfl=loss_dfl, reg_max=reg_max, **kwargs)

        self.loss_dkd = build_loss(dict(type='DKDLoss', alpha=1.0, beta=1.0, temperature=2.0))
        self.loss_ld_first = build_loss(dict(type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=10))
        self.loss_ld_second = build_loss(dict(type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=5))

        self.loss_tckd = build_loss(dict(type='TCKDLoss', alpha=1.0, beta=8.0, temperature=2.0))
        if loss_kd:
            self.loss_kd = build_loss(loss_kd)
        else:
            self.loss_kd = None

    def loss_single(self, anchors, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, stride, num_total_samples, ori_num_classes):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            stride (tuple): Stride in this scale level.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.
            ori_num_classes (int): Number of class that original detector can detect.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        # cls_score = cls_score[:, ori_num_classes:, :, :].permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) # old version 2021/05/31
        cls_score = cls_score[:, ori_num_classes:, :, :].permute(0, 2, 3, 1).reshape(-1,
                                                                                     self.cls_out_channels - ori_num_classes)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes - ori_num_classes  # minus ori_num_classes
        labels[labels == self.num_classes] = bg_class_ind  # revise labels
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = distance2bbox(pos_anchor_centers,
                                                 pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = bbox2distance(pos_anchor_centers,
                                           pos_decode_bbox_targets,
                                           self.reg_max).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)

        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_dfl, weight_targets.sum()

    @staticmethod
    def l2_loss(pred, target, weights=None, reduction='mean'):
        r"""Function that takes the mean element-wise square value difference.
        """
        assert target.size() == pred.size()
        loss = (pred - target).abs().float()
        if weights is not None:
            loss *= weights.unsqueeze(-1).expand(-1, loss.shape[-1])
        if reduction != 'none':
            loss = torch.mean(loss) if reduction == 'mean' else torch.sum(loss)
        return loss

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'ori_topk_cls_scores', 'ori_topk_bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,

             ori_sel_cls_first,
             ori_sel_cls_second,
             ori_sel_cls_inds_0_first,
             ori_sel_cls_inds_0_second,
             ori_sel_cls_inds_1_first,
             ori_sel_cls_inds_1_second,
             ori_sel_bbox_inds_0_first,
             ori_sel_bbox_inds_0_second,
             ori_sel_bbox_inds_1_first,
             ori_sel_bbox_inds_1_second,

             ori_num_classes,
             dist_loss_weight,
             new_model,
             ori_outs_head_tower,
             outs_head_tower,
             ori_outs_neck,
             new_outs_neck,
             ori_outs,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
        #                                   bbox_preds[0].device)
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        ###   原模型输出
        ori_cls_scores = ori_outs[0]
        ori_bbox_preds = ori_outs[1]
        ###

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        # ==============================>  常规的 detection loss  ====================================#
        losses_cls, losses_bbox, losses_dfl, \
            avg_factor = multi_apply(
            self.loss_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            self.anchor_generator.strides,
            num_total_samples=num_total_samples,
            ori_num_classes=ori_num_classes, )

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))

        # ==============================>  分类分支蒸馏loss  ====================================#
        num_imgs = cls_scores[0].size(0)
        new_cls_scores = [
            cls_score[:, :ori_num_classes, :, :].permute(0, 2, 3, 1).reshape(
                num_imgs, -1, ori_num_classes) for cls_score in cls_scores]
        new_cls_scores = torch.cat(new_cls_scores, dim=1)

        ### distillation classification

        new_cls_scores_0_first = new_cls_scores[0].gather(
            0, ori_sel_cls_inds_0_first.unsqueeze(-1).expand(-1, new_cls_scores[0].size(-1)))
        new_cls_scores_0_second = new_cls_scores[0].gather(
            0, ori_sel_cls_inds_0_second.unsqueeze(-1).expand(-1, new_cls_scores[0].size(-1)))

        new_cls_scores_1_first = new_cls_scores[1].gather(
            0, ori_sel_cls_inds_1_first.unsqueeze(-1).expand(-1, new_cls_scores[1].size(-1)))
        new_cls_scores_1_second = new_cls_scores[1].gather(
            0, ori_sel_cls_inds_1_second.unsqueeze(-1).expand(-1, new_cls_scores[1].size(-1)))


        ori_0_first_len = len(ori_sel_cls_inds_0_first)
        ori_0_second_len = len(ori_sel_cls_inds_0_second)
        ori_1_first_len = len(ori_sel_cls_inds_1_first)
        ori_1_second_len = len(ori_sel_cls_inds_1_second)

        if ori_0_first_len == 0 or ori_0_second_len == 0:
            loss_0_dist_cls_first = 0.0 if ori_0_first_len == 0 else dist_loss_weight * self.loss_dkd(
                new_cls_scores_0_first, ori_sel_cls_first[:ori_0_first_len])
            loss_0_dist_cls_second = 0.0 if ori_0_second_len == 0 else dist_loss_weight * self.l2_loss(
                new_cls_scores_0_second, ori_sel_cls_second[:ori_0_second_len])
            loss_0_dist_cls = loss_0_dist_cls_first + loss_0_dist_cls_second
        else:
            loss_0_dist_cls_first = dist_loss_weight * self.loss_dkd(new_cls_scores_0_first,
                                                                     ori_sel_cls_first[:ori_0_first_len])
            loss_0_dist_cls_second = dist_loss_weight * self.l2_loss(new_cls_scores_0_second,
                                                                     ori_sel_cls_second[:ori_0_second_len])
            loss_0_dist_cls = ori_0_first_len / (ori_0_first_len + ori_0_second_len) * loss_0_dist_cls_first + \
                              ori_0_second_len / (ori_0_first_len + ori_0_second_len) * loss_0_dist_cls_second

        if ori_1_first_len == 0 or ori_1_second_len == 0:
            loss_1_dist_cls_first = 0.0 if ori_1_first_len == 0 else dist_loss_weight * self.loss_dkd(
                new_cls_scores_1_first, ori_sel_cls_first[ori_0_first_len:ori_0_first_len + ori_1_first_len])
            loss_1_dist_cls_second = 0.0 if ori_1_second_len == 0 else dist_loss_weight * self.l2_loss(
                new_cls_scores_1_second, ori_sel_cls_second[ori_0_second_len:ori_0_second_len + ori_1_second_len])
            loss_1_dist_cls = loss_1_dist_cls_first + loss_1_dist_cls_second
        else:
            loss_1_dist_cls_first = dist_loss_weight * self.loss_dkd(
                new_cls_scores_1_first, ori_sel_cls_first[ori_0_first_len:ori_0_first_len + ori_1_first_len])
            loss_1_dist_cls_second = dist_loss_weight * self.l2_loss(
                new_cls_scores_1_second, ori_sel_cls_second[ori_0_second_len:ori_0_second_len + ori_1_second_len])
            loss_1_dist_cls = ori_1_first_len / (ori_1_first_len + ori_1_second_len) * loss_1_dist_cls_first + \
                              ori_1_second_len / (ori_1_first_len + ori_1_second_len) * loss_1_dist_cls_second

        loss_dist_cls = 0.5 * (loss_0_dist_cls + loss_1_dist_cls)
        # ==============================>  回归分支蒸馏loss  ====================================#

        new_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4 * (self.reg_max + 1))
            for bbox_pred in bbox_preds]
        new_bbox_preds = torch.cat(new_bbox_preds, dim=1)

        ori_bbox_preds = [
            ori_bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4 * (self.reg_max + 1))
            for ori_bbox_pred in ori_bbox_preds]
        ori_bbox_preds = torch.cat(ori_bbox_preds, dim=1)

        # ************************* 公共 *************************
        anchors = [anchor.permute(0, 1, 2).reshape(num_imgs, -1, 4) for anchor in anchor_list]
        anchors = torch.cat(anchors, dim=1)

        ori_cls_scores = [
            ori_cls_score[:, :ori_num_classes, :, :].permute(0, 2, 3, 1).reshape(
                num_imgs, -1, ori_num_classes) for ori_cls_score in ori_cls_scores]
        ori_cls_scores = torch.cat(ori_cls_scores, dim=1)
        nms_cfg = dict(iou_threshold=0.005)  # 0.005

        # ************************* image 0 *************************
        ori_bbox_preds_0 = self.integral(ori_bbox_preds[0])
        anchor_centers_0 = self.anchor_center(anchors[0])
        # ori decode bbox, shape (Num,4)
        decode_ori_bbox_pred_0 = distance2bbox(anchor_centers_0, ori_bbox_preds_0)

        ori_cls_conf_0 = ori_cls_scores[0].sigmoid()
        cls_conf_0, ids_0 = ori_cls_conf_0.max(dim=-1)

        # ============> first part
        if len(ori_sel_bbox_inds_0_first) > 0:
            thr_bboxes_0_first, thr_scores_0_first, thr_id_0_first = decode_ori_bbox_pred_0[ori_sel_bbox_inds_0_first], \
                cls_conf_0[ori_sel_bbox_inds_0_first], ids_0[ori_sel_bbox_inds_0_first]
            _, keep_0_first = batched_nms(thr_bboxes_0_first, thr_scores_0_first, thr_id_0_first, nms_cfg)
            nms_bbox_preds_0_first = new_bbox_preds[0].gather(
                0, ori_sel_bbox_inds_0_first.unsqueeze(-1).expand(-1, new_bbox_preds[0].size(-1)))
            new_topk_bbox_preds_0_first = nms_bbox_preds_0_first.gather(
                0, keep_0_first.unsqueeze(-1).expand(-1, nms_bbox_preds_0_first.size(-1)))
            nms_ori_topk_bbox_preds_0_first = ori_bbox_preds[0].gather(
                0, ori_sel_bbox_inds_0_first.unsqueeze(-1).expand(-1, ori_bbox_preds[0].size(-1)))
            ori_topk_bbox_preds_0_first = nms_ori_topk_bbox_preds_0_first.gather(
                0, keep_0_first.unsqueeze(-1).expand(-1, nms_ori_topk_bbox_preds_0_first.size(-1)))
            new_topk_bbox_corners_0_first = new_topk_bbox_preds_0_first.reshape(-1, self.reg_max + 1)
            ori_topk_pred_corners_0_first = ori_topk_bbox_preds_0_first.reshape(-1, self.reg_max + 1)
            weight_targets_0_first = ori_cls_scores[0].reshape(-1, ori_num_classes)[
                ori_sel_bbox_inds_0_first].detach().sigmoid()
            weight_targets_0_first = weight_targets_0_first.max(dim=1)[0][keep_0_first.reshape(-1)]
            loss_dist_bbox_0_first = dist_loss_weight * self.loss_ld_first(new_topk_bbox_corners_0_first,
                                                                           ori_topk_pred_corners_0_first,
                                                                           weight=weight_targets_0_first[:,
                                                                                  None].expand(
                                                                               -1, 4).reshape(-1), avg_factor=4.0)
        else:
            loss_dist_bbox_0_first = 0.0

        # ============> second part
        if len(ori_sel_bbox_inds_0_second) > 0:
            thr_bboxes_0_second, thr_scores_0_second, thr_id_0_second = decode_ori_bbox_pred_0[
                ori_sel_bbox_inds_0_second], \
                cls_conf_0[ori_sel_bbox_inds_0_second], ids_0[ori_sel_bbox_inds_0_second]
            _, keep_0_second = batched_nms(thr_bboxes_0_second, thr_scores_0_second, thr_id_0_second, nms_cfg)
            nms_bbox_preds_0_second = new_bbox_preds[0].gather(
                0, ori_sel_bbox_inds_0_second.unsqueeze(-1).expand(-1, new_bbox_preds[0].size(-1)))
            new_topk_bbox_preds_0_second = nms_bbox_preds_0_second.gather(
                0, keep_0_second.unsqueeze(-1).expand(-1, nms_bbox_preds_0_first.size(-1)))
            nms_ori_topk_bbox_preds_0_second = ori_bbox_preds[0].gather(
                0, ori_sel_bbox_inds_0_second.unsqueeze(-1).expand(-1, ori_bbox_preds[0].size(-1)))
            ori_topk_bbox_preds_0_second = nms_ori_topk_bbox_preds_0_second.gather(
                0, keep_0_second.unsqueeze(-1).expand(-1, nms_ori_topk_bbox_preds_0_second.size(-1)))
            new_topk_bbox_corners_0_second = new_topk_bbox_preds_0_second.reshape(-1, self.reg_max + 1)
            ori_topk_pred_corners_0_second = ori_topk_bbox_preds_0_second.reshape(-1, self.reg_max + 1)
            weight_targets_0_second = ori_cls_scores[0].reshape(-1, ori_num_classes)[
                ori_sel_bbox_inds_0_second].detach().sigmoid()
            weight_targets_0_second = weight_targets_0_second.max(dim=1)[0][keep_0_second.reshape(-1)]
            loss_dist_bbox_0_second = dist_loss_weight * self.loss_ld_second(new_topk_bbox_corners_0_second,
                                                                             ori_topk_pred_corners_0_second,
                                                                             weight=weight_targets_0_second[:,
                                                                                    None].expand(
                                                                                 -1, 4).reshape(-1), avg_factor=4.0)
        else:
            loss_dist_bbox_0_second = 0.0

        # ************************* image 1 *************************
        ori_bbox_preds_1 = self.integral(ori_bbox_preds[1])
        anchor_centers_1 = self.anchor_center(anchors[1])
        # ori decode bbox, shape (Num,4)
        decode_ori_bbox_pred_1 = distance2bbox(anchor_centers_1, ori_bbox_preds_1)

        ori_cls_conf_1 = ori_cls_scores[1].sigmoid()
        cls_conf_1, ids_1 = ori_cls_conf_1.max(dim=-1)

        # ============> first part
        if len(ori_sel_bbox_inds_1_first) > 0:
            thr_bboxes_1_first, thr_scores_1_first, thr_id_1_first = decode_ori_bbox_pred_1[ori_sel_bbox_inds_1_first], \
                cls_conf_1[ori_sel_bbox_inds_1_first], ids_1[ori_sel_bbox_inds_1_first]
            _, keep_1_first = batched_nms(thr_bboxes_1_first, thr_scores_1_first, thr_id_1_first, nms_cfg)
            nms_bbox_preds_1_first = new_bbox_preds[1].gather(
                0, ori_sel_bbox_inds_1_first.unsqueeze(-1).expand(-1, new_bbox_preds[1].size(-1)))
            new_topk_bbox_preds_1_first = nms_bbox_preds_1_first.gather(
                0, keep_1_first.unsqueeze(-1).expand(-1, nms_bbox_preds_1_first.size(-1)))
            nms_ori_topk_bbox_preds_1_first = ori_bbox_preds[1].gather(
                0, ori_sel_bbox_inds_1_first.unsqueeze(-1).expand(-1, ori_bbox_preds[1].size(-1)))
            ori_topk_bbox_preds_1_first = nms_ori_topk_bbox_preds_1_first.gather(
                0, keep_1_first.unsqueeze(-1).expand(-1, nms_ori_topk_bbox_preds_1_first.size(-1)))
            new_topk_bbox_corners_1_first = new_topk_bbox_preds_1_first.reshape(-1, self.reg_max + 1)
            ori_topk_pred_corners_1_first = ori_topk_bbox_preds_1_first.reshape(-1, self.reg_max + 1)

            weight_targets_1_first = ori_cls_scores[1].reshape(-1, ori_num_classes)[
                ori_sel_bbox_inds_1_first].detach().sigmoid()
            weight_targets_1_first = weight_targets_1_first.max(dim=1)[0][keep_1_first.reshape(-1)]
            loss_dist_bbox_1_first = dist_loss_weight * self.loss_ld_first(new_topk_bbox_corners_1_first,
                                                                           ori_topk_pred_corners_1_first,
                                                                           weight=weight_targets_1_first[:,
                                                                                  None].expand(
                                                                               -1, 4).reshape(-1), avg_factor=4.0)
        else:
            loss_dist_bbox_1_first = 0.0

        # ============> second part
        if len(ori_sel_bbox_inds_1_second) > 0:
            thr_bboxes_1_second, thr_scores_1_second, thr_id_1_second = decode_ori_bbox_pred_1[
                ori_sel_bbox_inds_1_second], \
                cls_conf_1[ori_sel_bbox_inds_1_second], ids_1[ori_sel_bbox_inds_1_second]
            _, keep_1_second = batched_nms(thr_bboxes_1_second, thr_scores_1_second, thr_id_1_second, nms_cfg)
            nms_bbox_preds_1_second = new_bbox_preds[1].gather(
                0, ori_sel_bbox_inds_1_second.unsqueeze(-1).expand(-1, new_bbox_preds[1].size(-1)))
            new_topk_bbox_preds_1_second = nms_bbox_preds_1_second.gather(
                0, keep_1_second.unsqueeze(-1).expand(-1, nms_bbox_preds_1_first.size(-1)))
            nms_ori_topk_bbox_preds_1_second = ori_bbox_preds[1].gather(
                0, ori_sel_bbox_inds_1_second.unsqueeze(-1).expand(-1, ori_bbox_preds[1].size(-1)))
            ori_topk_bbox_preds_1_second = nms_ori_topk_bbox_preds_1_second.gather(
                0, keep_1_second.unsqueeze(-1).expand(-1, nms_ori_topk_bbox_preds_1_second.size(-1)))
            new_topk_bbox_corners_1_second = new_topk_bbox_preds_1_second.reshape(-1, self.reg_max + 1)
            ori_topk_pred_corners_1_second = ori_topk_bbox_preds_1_second.reshape(-1, self.reg_max + 1)

            weight_targets_1_second = ori_cls_scores[1].reshape(-1, ori_num_classes)[
                ori_sel_bbox_inds_1_second].detach().sigmoid()
            weight_targets_1_second = weight_targets_1_second.max(dim=1)[0][keep_1_second.reshape(-1)]
            loss_dist_bbox_1_second = dist_loss_weight * self.loss_ld_second(new_topk_bbox_corners_1_second,
                                                                             ori_topk_pred_corners_1_second,
                                                                             weight=weight_targets_1_second[:,
                                                                                    None].expand(
                                                                                 -1, 4).reshape(-1), avg_factor=4.0)
        else:
            loss_dist_bbox_1_second = 0.0

        # *************************** end ***************************

        loss_dist_bbox = 0.5 * (
                loss_dist_bbox_0_first + loss_dist_bbox_1_first + loss_dist_bbox_0_second + loss_dist_bbox_1_second)

        # loss_dist_bbox = 0.5 * (loss_dist_bbox_0_first + loss_dist_bbox_1_first)
        ### distillation regression

        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_dfl=losses_dfl,
            loss_dist_cls=loss_dist_cls,
            loss_dist_bbox=loss_dist_bbox)
