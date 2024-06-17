import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.core.evaluation import pose_pck_accuracy
from mmpose.models import HEADS
from mmpose.models.builder import build_loss
from mmpose.models.heads import TopdownHeatmapBaseHead
from ..utils.utils import get_one_hot


@HEADS.register_module()
class GlobalHead(TopdownHeatmapBaseHead):
    """Global head.
    Args:
        out_channels (int): Number of feature channels.
        keypoints_num (int): Number of keypoints in the dataset.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
    """

    def __init__(self,
                 out_channels,
                 keypoints_num,
                 superkeypoints=None,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.loss = build_loss(loss_keypoint)
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatMap')
        self.init_weights(out_channels, keypoints_num, superkeypoints)

    def init_weights(self, out_channels, keypoints_num, superkeypoints):
        if superkeypoints is not None:
            checkpoint = superkeypoints['checkpoint']
            mname = superkeypoints['mname']

            superkeypoints = torch.load(checkpoint)

            if 'state_dict' in superkeypoints:
                superkeypoints = superkeypoints['state_dict']

            keypoints = superkeypoints[mname]
            self.keypoints = torch.nn.parameter.Parameter(
                data=keypoints, requires_grad=True)

        else:
            keypoints = torch.empty(keypoints_num, out_channels)
            nn.init.normal_(keypoints, std=0.001)
            self.keypoints = torch.nn.parameter.Parameter(
                data=keypoints, requires_grad=True)

    def forward_train(self, img, target, target_weight, feature,
                      keypoint_index_onehot, **kwargs):
        """ Global training step. Define the step for training the features extractor. See section 3.2 in the main paper.

        Note:
            batch size: N
            number of keypoints: K
            number of img channels: imgC (Default: 3)
            number of feature channels: C
            total number (super-)keypoints in the dataset: L
            img height: imgH
            img weight: imgW
            heatmaps height: H
            heatmaps weight: W

        Args:
            img (torch.Tensor[NximgCximgHximgW]): Input images.
            feature (torch.Tensor[NxCxHxW]): Image features extracted from features extractor.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]): Weights across
                different joint types.
            keypoint_index_onehot (torch.Tensor[NxKxL]): One-hot ground-truth (super-)keypoints.
        Returns:
            losses (dict): Losses and accuracy
        """

        keypoints = torch.einsum('bks,schw->bkchw', keypoint_index_onehot,
                                 self.keypoints.unsqueeze(-1).unsqueeze(-1))

        keypoints = keypoints.flatten(0, 1)
        batch_size = feature.size(0)
        feature = feature.view(1, -1, *feature.size()[2:])
        output = F.conv2d(
            feature, keypoints, stride=1, padding=0, groups=batch_size)
        output = output.view(batch_size, -1, *feature.size()[2:])

        losses = dict()
        keypoint_losses = self.get_loss(output, target, target_weight)
        losses.update(keypoint_losses)
        keypoint_accuracy = self.get_accuracy(output, target, target_weight)
        losses.update(keypoint_accuracy)
        return losses

    def forward_global_test(self, keypoint_index_onehot, **kwargs):
        keypoints = torch.einsum('bks,sc->bkc', keypoint_index_onehot,
                                 self.keypoints)
        return keypoints

    def get_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K
            heatmaps height: H
            heatmaps weight: W

        Args:
            output (torch.Tensor[NxKxHxW]): Output heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """

        losses = dict()

        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 4 and target_weight.dim() == 3
        losses['mse_loss'] = self.loss(output, target, target_weight)

        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K
            heatmaps height: H
            heatmaps weight: W

        Args:
            output (torch.Tensor[NxKxHxW]): Output heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """

        accuracy = dict()

        if self.target_type == 'GaussianHeatMap':
            _, avg_acc, _ = pose_pck_accuracy(
                output.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                target_weight.detach().cpu().numpy().squeeze(-1) > 0,
                thr=0.2)
            accuracy['acc_pose'] = float(avg_acc)

        return accuracy
