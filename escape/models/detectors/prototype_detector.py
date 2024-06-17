import cv2
import math
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.image import imwrite
from mmcv.runner.checkpoint import load_state_dict
from mmcv.visualization.image import imshow

from mmpose.core.evaluation.top_down_eval import keypoints_from_heatmaps
from mmpose.models import builder
from mmpose.models.builder import POSENETS
from mmpose.models.detectors.base import BasePose
from ..utils.utils import get_backbone_deconv_state_dict


@POSENETS.register_module()
class PrototypeDetector(BasePose):
    """Prototype-based keypoint detectors.
    Args:
        backbone (dict): Backbone modules to extract feature.
        deconv (dict): Deconvolution modules to make feature size match with target heatmap size.
        keypoint_head (dict): Keypoint head to compute keypoint prototypes. This head is for global training.
        keypoint_adaptation (dict): Keypoint head to compute keypoint prototypes. This head is for few-shot testing.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
    """

    def __init__(self,
                 backbone,
                 deconv,
                 keypoint_head,
                 keypoint_adaptation=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()

        self.backbone = builder.build_backbone(backbone)
        self.deconv = builder.build_head(deconv)
        self.keypoint_head = builder.build_head(keypoint_head)
        if keypoint_adaptation is not None:
            self.keypoint_adaptation = builder.build_head(
                keypoint_adaptation).forward_test
            self.vmaped_keypoint_adaptation = torch.vmap(
                self.keypoint_adaptation)
        else:
            self.keypoint_adaptation = self.keypoint_head.forward_test
            self.vmaped_keypoint_adaptation = torch.vmap(
                self.keypoint_head.forward_test)

        self.init_weights(pretrained=pretrained)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.target_type = test_cfg.get('target_type', 'GaussianHeatMap')
        self.fewshot_testing = test_cfg.get('fewshot_testing', False)

        self.vmaped_extract_feature = torch.vmap(self.extract_feature)

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        if pretrained is not None:
            backbone, deconv = get_backbone_deconv_state_dict(pretrained)
            load_state_dict(self.backbone, backbone)
            self.deconv.init_weights(deconv)

    def extract_feature(self, img):
        feature = self.backbone(img)
        feature = self.deconv(feature)
        return feature

    def forward(self,
                img,
                target,
                target_weight,
                keypoint_index_onehot=None,
                img_q=None,
                img_metas=None,
                return_loss=True):
        """ Calls either forward_train or forward_test depending on whether
        return_loss=True.
            forward_train: global training.
            forward_test: calls either forward_global_test or forward_fewshot_test depending on whether self.fewshot_testing=True.
        """
        if return_loss:
            return self.forward_train(img, target, target_weight,
                                      keypoint_index_onehot, img_metas)
        else:
            return self.forward_test(img, target, target_weight,
                                     keypoint_index_onehot, img_q, img_metas)

    def forward_train(self, img, target, target_weight, keypoint_index_onehot,
                      img_metas):
        """ Global training step.

        Note:
            batch size: N
            number of keypoints: K
            number of img channels: imgC (Default: 3)
            total number (super-)keypoints in the dataset: L
            img height: imgH
            img weight: imgW
            heatmaps height: H
            heatmaps weight: W

        Args:
            img (torch.Tensor[NximgCximgHximgW]): Input images.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]): Weights across
                different joint types.
            keypoint_index_onehot (torch.Tensor[NxKxL]): One-hot ground-truth (super-)keypoints.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
        Returns:
            losses (dict): Losses and accuracy
        """
        feature = self.extract_feature(img)

        losses = self.keypoint_head.forward_train(
            img=img,
            target=target,
            target_weight=target_weight,
            feature=feature,
            keypoint_index_onehot=keypoint_index_onehot)
        return losses

    def forward_test(self, img, target, target_weight, keypoint_index_onehot,
                     img_q, img_metas):
        if self.fewshot_testing:
            return self.forward_fewshot_test(img, target, target_weight, img_q,
                                             img_metas)
        else:
            return self.forward_global_test(img, target, target_weight,
                                            keypoint_index_onehot, img_metas)

    def forward_global_test(self, img, target, target_weight,
                            keypoint_index_onehot, img_metas):
        """ Global testing step: estimates keypoints for images in conventional way as in Eq.(1) in the main paper.

        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C (Default: 3)
            img height: imgH
            img weight: imgW
            heatmaps height: H
            heatmaps weight: W

        Args:
            img (torch.Tensor[NxCximgHximgW]): Input images.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]): Weights across
                different joint types.
            keypoint_index_onehot (torch.Tensor[NxKxL]): One-hot ground-truth (super-)keypoints.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
        Returns:
            keypoint_result (dict): keypoint predictions, extracted keypoint features, target_weight
        """
        batch_size, _, img_height, img_width = img.shape

        feature = self.extract_feature(img)

        prototypes = self.keypoint_head.forward_global_test(
            keypoint_index_onehot)
        output = torch.einsum('bchw,blc->blhw', feature, prototypes)
        output_heatmap = output.detach().cpu().numpy()

        keypoint_result = self.global_decode(
            img_metas, output_heatmap, img_size=[img_width, img_height])
        keypoint_features = torch.einsum('bchw,blhw->blc', feature, target) \
                        / (target.unsqueeze(2).sum([-1,-2]) + 1e-12)
        keypoint_result['keypoint_features'] = keypoint_features.detach().cpu(
        ).numpy()
        keypoint_result['target_weight'] = target_weight.detach().cpu().numpy()
        return keypoint_result

    def forward_fewshot_test(self, img_s, target_s, target_weight_s, img_q,
                             img_metas):
        """ Few-shot testing step: computes prototypes for novel keypoints using support images and makes predictions for queries samples.

        Note:
            num_episodes: N
            num_supports: S
            num_queries: Q
            num_keypoints: K
            num_img_channel: C (Default: 3)
            img height: imgH
            img weight: imgW
            heatmaps height: H
            heatmaps weight: W

        Args:
            img_s (torch.Tensor[NxSxCximgHximgW]): Input support images.
            target_s (torch.Tensor[NxSxKxHxW]): Target heatmaps for support images.
            target_weight_s (torch.Tensor[NxSxKx1]): Weights across different joint types.
            img_q (torch.Tensor[NxQxCximgHximgW]): Input query images.
            img_metas (list(dict)): Information about data augmentation of query samples
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox 
        Returns:
            keypoint_result (dict): keypoint predictions for query images.
        """

        num_episodes, batch_size, _, img_height, img_width = img_q.shape
        num_shots = img_s.size(1)

        img_metas = sum(zip(*img_metas), ())

        if num_episodes == 1:
            img_s, target_s, target_weight_s, img_q = img_s.squeeze(
                0), target_s.squeeze(0), target_weight_s.squeeze(
                    0), img_q.squeeze(0)

            feature_s = self.extract_feature(img_s)
            feature_q = self.extract_feature(img_q)

            prototypes = self.keypoint_adaptation(img_s, feature_s, target_s,
                                                  target_weight_s)

            output = torch.einsum('bchw,blc->blhw', feature_q, prototypes)

        else:
            feature_s = self.vmaped_extract_feature(img_s)
            feature_q = self.vmaped_extract_feature(img_q)

            prototypes = self.vmaped_keypoint_adaptation(
                img_s, feature_s, target_s, target_weight_s)

            output = torch.einsum('ebchw,eblc->eblhw', feature_q,
                                  prototypes).flatten(0, 1)

        keypoint_result = self.decode(
            img_metas, output, img_size=[img_width, img_height])

        return keypoint_result

    def global_decode(self, img_metas, output, **kwargs):
        """Decode keypoints from heatmaps.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, H, W]): model predicted heatmaps.
        """
        batch_size = len(img_metas)

        if 'bbox_id' or 'query_bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None
        if 'keypoint_index' in img_metas[0]:
            keypoint_index = []

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['center']
            s[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])

            if 'bbox_id' in img_metas[i]:
                bbox_ids.append(img_metas[i]['bbox_id'])

            if 'keypoint_index' in img_metas[i]:
                keypoint_index.append(img_metas[i]['keypoint_index'])

        preds, maxvals = keypoints_from_heatmaps(
            output,
            c,
            s,
            unbiased=self.test_cfg.get('unbiased_decoding', False),
            post_process=self.test_cfg.get('post_process', 'default'),
            kernel=self.test_cfg.get('modulate_kernel', 11),
            valid_radius_factor=self.test_cfg.get('valid_radius_factor',
                                                  0.0546875),
            use_udp=self.test_cfg.get('use_udp', False),
            target_type=self.test_cfg.get('target_type', 'GaussianHeatMap'))

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids
        result['keypoint_index'] = keypoint_index
        return result

    def decode(self, img_metas, output, **kwargs):
        """Decode keypoints from heatmaps.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, H, W]): model predicted heatmaps.
        """
        kernel = self.test_cfg.get('pooling_kernel', 15)
        output = F.conv3d(output.unsqueeze(1), torch.ones(1,1,1,kernel,kernel).cuda(), padding='same').squeeze(1) \
             / F.conv3d(torch.ones_like(output).cuda().unsqueeze(1), torch.ones(1,1,1,kernel,kernel).cuda(), padding='same').squeeze(1)
        output = output.detach().cpu().numpy()

        batch_size = len(img_metas)

        if 'bbox_id' or 'query_bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['query_center']
            s[i, :] = img_metas[i]['query_scale']
            image_paths.append(img_metas[i]['query_image_file'])

            if 'query_bbox_score' in img_metas[i]:
                score[i] = np.array(
                    img_metas[i]['query_bbox_score']).reshape(-1)
            if 'bbox_id' in img_metas[i]:
                bbox_ids.append(img_metas[i]['bbox_id'])
            elif 'query_bbox_id' in img_metas[i]:
                bbox_ids.append(img_metas[i]['query_bbox_id'])

        preds, maxvals = keypoints_from_heatmaps(
            output,
            c,
            s,
            unbiased=self.test_cfg.get('unbiased_decoding', False),
            post_process=self.test_cfg.get('post_process', 'default'),
            kernel=self.test_cfg.get('modulate_kernel', 11),
            valid_radius_factor=self.test_cfg.get('valid_radius_factor',
                                                  0.0546875),
            use_udp=self.test_cfg.get('use_udp', False),
            target_type=self.test_cfg.get('target_type', 'GaussianHeatMap'))

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result

    # UNMODIFIED
    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_limb_color=None,
                    radius=4,
                    text_color=(255, 0, 0),
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_limb_color (np.array[Mx3]): Color of M limbs.
                If None, do not draw limbs.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """

        img = mmcv.imread(img)
        img = img.copy()
        img_h, img_w, _ = img.shape

        bbox_result = []
        pose_result = []
        for res in result:
            bbox_result.append(res['bbox'])
            pose_result.append(res['keypoints'])

        if len(bbox_result) > 0:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            mmcv.imshow_bboxes(
                img,
                bboxes,
                colors=bbox_color,
                top_k=-1,
                thickness=thickness,
                show=False,
                win_name=win_name,
                wait_time=wait_time,
                out_file=None)

            for person_id, kpts in enumerate(pose_result):
                # draw each point on image
                if pose_kpt_color is not None:
                    assert len(pose_kpt_color) == len(kpts), (
                        len(pose_kpt_color), len(kpts))
                    for kid, kpt in enumerate(kpts):
                        x_coord, y_coord, kpt_score = int(kpt[0]), int(
                            kpt[1]), kpt[2]
                        if kpt_score > kpt_score_thr:
                            img_copy = img.copy()
                            r, g, b = pose_kpt_color[kid]
                            cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                       radius, (int(r), int(g), int(b)), -1)
                            transparency = max(0, min(1, kpt_score))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)

                # draw limbs
                if skeleton is not None and pose_limb_color is not None:
                    assert len(pose_limb_color) == len(skeleton)
                    for sk_id, sk in enumerate(skeleton):
                        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1,
                                                                  1]))
                        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1,
                                                                  1]))
                        if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                                and pos1[1] < img_h and pos2[0] > 0
                                and pos2[0] < img_w and pos2[1] > 0
                                and pos2[1] < img_h
                                and kpts[sk[0] - 1, 2] > kpt_score_thr
                                and kpts[sk[1] - 1, 2] > kpt_score_thr):
                            img_copy = img.copy()
                            X = (pos1[0], pos2[0])
                            Y = (pos1[1], pos2[1])
                            mX = np.mean(X)
                            mY = np.mean(Y)
                            length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                            angle = math.degrees(
                                math.atan2(Y[0] - Y[1], X[0] - X[1]))
                            stickwidth = 2
                            polygon = cv2.ellipse2Poly(
                                (int(mX), int(mY)),
                                (int(length / 2), int(stickwidth)), int(angle),
                                0, 360, 1)

                            r, g, b = pose_limb_color[sk_id]
                            cv2.fillConvexPoly(img_copy, polygon,
                                               (int(r), int(g), int(b)))
                            transparency = max(
                                0,
                                min(
                                    1, 0.5 *
                                    (kpts[sk[0] - 1, 2] + kpts[sk[1] - 1, 2])))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)

        show, wait_time = 1, 1
        if show:
            height, width = img.shape[:2]
            max_ = max(height, width)

            factor = min(1, 800 / max_)
            enlarge = cv2.resize(
                img, (0, 0),
                fx=factor,
                fy=factor,
                interpolation=cv2.INTER_CUBIC)
            imshow(enlarge, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img
