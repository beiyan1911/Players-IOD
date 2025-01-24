from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmcv.utils import print_log

from mmdet.core.visualization import imshow_det_bboxes
from mmdet.utils import get_root_logger


class BaseDetector(nn.Module, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass

    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (list[Tensor]): List of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        batch_input_shape = tuple(imgs[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

    async def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if pretrained is not None:
            logger = get_root_logger()
            print_log(f'load model from: {pretrained}', logger=logger)

    async def aforward_test(self, *, img, img_metas, **kwargs):
        for var, name in [(img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(img)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(img)}) '
                             f'!= num of image metas ({len(img_metas)})')
        # TODO: remove the restriction of samples_per_gpu == 1 when prepared
        samples_per_gpu = img[0].size(0)
        assert samples_per_gpu == 1

        if num_augs == 1:
            return await self.async_simple_test(img[0], img_metas[0], **kwargs)
        else:
            raise NotImplementedError

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        # class_names = ['Tyler_Herro_14', 'Duncan_Robinson_55', 'Jimmy_Butler_22',  # 'Jae_Crowder_99',
        #                'Bam_Adebayo_13', 'Kendrick_Nunn_25', 'Goran_Dragic_7', 'Caldwell_Pope_1',
        #                'Danny_Green_14', 'LeBron_James_23', 'Anthony_Davis_3', 'Dwight_Howard_39',
        #                'Kyle_Kuzma_0', 'Rajon_Rondo_9', 'Alex_Caruso_4', 'Markieff_Morris_88',
        #
        #                'Mikal_Bridges_25', 'Chris_Paul_3', 'Devin_Booker_1', 'Deandre_Ayton_22',
        #                'Jae_Crowder_99', 'Cameron_Johnson_23', 'Frank_kaminsky_8', 'Cameron_Payne_15',
        #                'Torrey_Craig_12', 'PJ_Tucker_17', 'Jrue_Holidy_21', 'Giannis_antetokounmpo_34',
        #                'Khris_Middleton_22', 'Brook_Lopez_11', 'Bobby_Portis_9', 'Pat_Connaughton_24',
        #                'Jevon_Carter_5',
        #
        #                'GREEN_draymond_23', 'THOMPSON_klay_11', 'WIGGINS_andrew_22', 'CURRY_stephen_30',
        #                'PORTER_jr.otto_32', 'LOONEY_kevon_5', 'PAYTON2_gary_0', 'POOLE_jordan_3',
        #                'Lguodala_andre_9', 'BJELICA_nemanja_8', 'Williams3_robert_44', 'Brown_jaylen_7',
        #                'Tatum_jayson_0', 'Smart_marcus_36', 'Horford_al_42', 'White_derrick_9',
        #                'Pritchard_payton_11', 'Williams_grant_12'  ]

        # # ************************* for basketball all *************************************
        # class_names =[
        #             'Tyler Herro',      # 15            # 'Tyler_Herro_14',      # 15
        #             'Duncan Robinson',                  #  'Duncan_Robinson_55',
        #             'Jimmy Butler',                     #  'Jimmy_Butler_22',
        #             'Bam Adebayo',                      #  'Bam_Adebayo_13',
        #             'Kendrick Nunn',                    #  'Kendrick_Nunn_25',
        #             'Goran Dragic',                      #  'Goran_Dragic_7',
        #             'Kentavious Caldwell-Pope',                     #  'Caldwell_Pope_1',
        #             'Danny Green',                      #  'Danny_Green_14',
        #             'LeBron James',                     #  'LeBron_James_23',
        #             'Anthony Davis',                     #  'Anthony_Davis_3',
        #             'Dwight Howard',                    #  'Dwight_Howard_39',
        #             'Kyle Kuzma',                        #  'Kyle_Kuzma_0',
        #             'Rajon Rondo',                       #  'Rajon_Rondo_9',
        #             'Alex Caruso',                       #  'Alex_Caruso_4',
        #             'Markieff Morris',                  #  'Markieff_Morris_88',
        #                                                    #
        #             'Mikal Bridges',    # 17            #  'Mikal_Bridges_25',    # 17
        #             'Chris Paul',                        #  'Chris_Paul_3',
        #             'Devin Booker',                      #  'Devin_Booker_1',
        #             'Deandre Ayton',                    #  'Deandre_Ayton_22',
        #             'Jae Crowder',                      #  'Jae_Crowder_99',
        #             'Cameron Johnson',                  #  'Cameron_Johnson_23',
        #             'Frank kaminsky',                    #  'Frank_kaminsky_8',
        #             'Cameron Payne',                    #  'Cameron_Payne_15',
        #             'Torrey Craig',                     #  'Torrey_Craig_12',
        #             'P.J. Tucker',                        #  'PJ_Tucker_17',
        #             'Jrue Holidy',                      #  'Jrue_Holidy_21',
        #             'Giannis Antetokounmpo',            #  'Giannis_antetokounmpo_34',
        #             'Khris Middleton',                  #  'Khris_Middleton_22',
        #             'Brook Lopez',                      #  'Brook_Lopez_11',
        #             'Bobby Portis',                      #  'Bobby_Portis_9',
        #             'Pat Connaughton',                  #  'Pat_Connaughton_24',
        #             'Jevon Carter',                      #  'Jevon_Carter_5',
        #                                                    #
        #             'Draymond Green',                   #  'GREEN_draymond_23',
        #             'Klay Thompson',                    #  'THOMPSON_klay_11',
        #             'Andrew Wiggins',                   #  'WIGGINS_andrew_22',
        #             'Stephen Curry',                    #  'CURRY_stephen_30',
        #             'Otto Porter, Jr',                   #  'PORTER_jr.otto_32',
        #             'Kevon Looney',                      #  'LOONEY_kevon_5',
        #             'Gary Payton II',                      #  'PAYTON2_gary_0',
        #             'Jordan Poole',                      #  'POOLE_jordan_3',
        #             'Andre Iguodala',                    #  'Lguodala_andre_9',
        #             'Nemanja Bjelica',                   #  'BJELICA_nemanja_8',
        #             'Robert Williams III',                 #  'Williams3_robert_44',
        #             'Jaylen Brown',                      #  'Brown_jaylen_7',
        #             'Jayson Tatum',                      #  'Tatum_jayson_0',
        #             'Marcus Smart',                     #  'Smart_marcus_36',
        #             'Al Horford',                       #  'Horford_al_42',
        #             'Derrick White',                     #  'White_derrick_9',
        #             'Payton Pritchard',                 #  'Pritchard_payton_11',
        #             'Grant Williams'  ]                 #  'Williams_grant_12'  ]

        # ************************* for zhuke  and two ranks *************************************

        # class_names = [
        #     'Tyler Herro',       #    'Tyler_Herro_14': 0,  # =========> 热火 <=========
        #     'Duncan Robinson',   #    'Duncan_Robinson_55': 1,
        #     'Jimmy Butler',      #    'Jimmy_Butler_22': 2,
        #     'Jae Crowder',       #    'Jae_Crowder_99': 3,
        #     'Bam Adebayo',       #    'Bam_Adebayo_13': 4,
        #     'Andre Iguodala',    #    'Lguodala_andre_9': 5,
        #     'Kendrick Nunn',     #    'Kendrick_Nunn_25': 6,
        #     'Goran Dragic',      #    'Goran_Dragic_7': 7,
        #     'Kelly Olynyk',      #    'Kelly_Olynyk_9': 8,
        #     'Kentavious Caldwell-Pope',     #    'Caldwell_Pope_1': 9,  # =========> lakers <=========
        #     'Danny Green',       #    'Danny_Green_14': 10,
        #     'LeBron James',      #    'LeBron_James_23': 11,
        #     'Anthony Davis',     #    'Anthony_Davis_3': 12,
        #     'Dwight Howard',     #    'Dwight_Howard_39': 13,
        #     'Kyle Kuzma',        #    'Kyle_Kuzma_0': 14,
        #     'Rajon Rondo',       #    'Rajon_Rondo_9': 15,
        #     'Alex Caruso',       #    'Alex_Caruso_4': 16,
        #     'Markieff Morris']   #    'Markieff_Morris_88': 17}

        # ************************* for volleyball  *************************************

        # class_names = [
        #       'Kim Yeon Koung',  #                    #   '10_ Kim_Yeon_Koung': 0,  # =========> stage 1  19  <=========
        #       'Kim Su Ji',                            #   '11_ Kim_Su_Ji': 1,
        #       'AYDEMIR AKYOL Naz',                     #   '11_naz_aydemir_akyol': 2,
        #       'PARK Jeongah',                         #   '13_ Park_Jeongah': 3,
        #       'Meryem Boz',                            #   '13_meryem_boz': 4,
        #       'YANG Hyo Jin',                         #   '14_ Yang_Hyo_Jin': 5,
        #       'Eda Erdem Dundar',                      #   '14_eda_erdem_dundar': 6,
        #       'Zehra Gunes',                           #   '18_zehra_gunes': 7,
        #       'Pyo Seungju',                          #   '19_ Pyo_Seungju': 8,
        #       'Lee Soyoung',                           #   '1_ Lee_Soyoung': 9,
        #       'Simge Sebnem Akoz',                     #   '2_simge_sebnem_akoz': 10,
        #       'Yeum Hye Seon',                        #   '3_ Yeum_Hye_Seon': 11,
        #       'Cansu Ozbay',                           #   '3_cansu_ozbay': 12,
        #       'Kim Heejin',                           #   '4_ Kim_Heejin': 13,
        #       'Seyma Ercan',                           #   '5_seyma_ercan': 14,
        #       'Hande Baladin',                         #   '7_hande_baladin': 15,
        #       'Ebrar Karakust',                       #   '99_ebrar_karakust': 16,
        #       'OH Jiyoung',                           #   '9_ OH_Jiyoung': 17,
        #       'Meliha Ismailoglu',                     #   '9_meliha_ismailoglu': 18,
        #
        #       'Maja Ognjenovic',                      #   '10_maja_ognjenovic': 19,  # =========> stage 2  15  <=========
        #       'Anna Danesi',                          #   '11_anna_danesi': 20,
        #       'Sarah Luisa Fahr',                     #   '13_sarah_luisa_fahr': 21,
        #       'Elena Pietrini',                       #   '14_elena_pietrini': 22,
        #       'Milena Rasic',                         #   '16_milena_rasic': 23,
        #       'Miryam Fatime Sylla',                    #   '17_miryam_fatime_sylla': 24,
        #       'Silvija Popovic',                      #   '17_silvija_popovic': 25,
        #       'Paola Ogechi Egonu',                   #   '18_paola_ogechi_egonu': 26,
        #       'Tijana Boskovic',                      #   '18_tijana_boskovic': 27,
        #       'Bojana Milenkovic',                    #   '19_bojana_milenkovic': 28,
        #       'Bianka Busa',                           #   '1_bianka_busa': 29,
        #       'Mina Popovic',                          #   '5_mina_popovic': 30,
        #       'Monica De Gennaro',                     #   '6_monica_de_gennaro': 31,
        #       'Alessia Orro',                          #   '8_alessia_orro': 32,
        #       'Caterina Chiara Bosetti']               #   '9_caterina_chiara_bosetti': 33}

        # ************************* for volleyball V3 *************************************
        # class_names = [
        #     'Kim_Yeon_Koung',  # '10_ Kim_Yeon_Koung': 0,  # =========> stage 1 <====== 韩国队
        #     'Kim_Su_Ji',  # '11_ Kim_Su_Ji': 1,
        #     'Park_Jeongah',  # '13_ Park_Jeongah': 2,
        #     'Yang_Hyo_Jin',  # '14_ Yang_Hyo_Jin': 3,
        #     'Jeong_Jiyun',  # '16_ Jeong_Jiyun': 4,
        #     'Pyo_Seungju',  # '19_ Pyo_Seungju': 5,
        #     'Lee_Soyoung',  # '1_ Lee_Soyoung': 6,
        #     'Yeum_Hye_Seon',  # '3_ Yeum_Hye_Seon': 7,
        #     'Kim_Heejin',  # '4_ Kim_Heejin': 8,
        #     'Park_Eunjin',  # '8_ Park_Eunjin': 9,
        #     'OH_Jiyoung',  # '9_ OH_Jiyoung': 10,
        #     #
        #     # ********************                           ## ************************** # =========> stage 1 <====== 土耳其队
        #     'naz_aydemir_akyol',  # '11_naz_aydemir_akyol': 11,
        #     'meryem_boz',  # '13_meryem_boz': 12,
        #     '14_eda_erdem_dundar',  # '14_eda_erdem_dundar': 13,
        #     'zehra_gunes',  # '18_zehra_gunes': 14,
        #     'simge_sebnem_akoz',  # '2_simge_sebnem_akoz': 15,
        #     'cansu_ozbay',  # '3_cansu_ozbay': 16,
        #     'tugba_senoglu',  # '4_tugba_senoglu': 17,
        #     'seyma_ercan',  # '5_seyma_ercan': 18,
        #     'hande_baladin',  # '7_hande_baladin': 19,
        #     'ebrar_karakust',  # '99_ebrar_karakust': 20,
        #     'meliha_ismailoglu']  # '9_meliha_ismailoglu': 21}

        # ************************* for volleyball V4 *************************************
        class_names = [
            'Maja Ognjenovic',
            'Milena Rasic',
            'Silvija Popovic',
            'Tijana Boskovic',
            'Bojana Milenkovic',
            'Bianka Busa',
            'Mina Popovic',
            # ******************************  意大利
            'Cristina Chirichella',
            'Anna Danesi',
            'Elena Pietrini',
            'Paola Ogechi Egonu',
            'Monica De Gennaro',
            'Alessia Orro',
            'Caterina Chiara Bosetti']

        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            # class_names=class_names,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img
