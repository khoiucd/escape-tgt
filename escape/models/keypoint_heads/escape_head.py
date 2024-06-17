import torch
import torch.nn as nn

from mmpose.models import HEADS, builder
from ..utils.superglue import MATCHING
from ..utils.utils import get_one_hot
from .fewshot_head import FewshotHead


@HEADS.register_module()
class ESCAPEHead(FewshotHead):
    """ESCAPE head.
    Args:
        matching_network (dict): Learnable matching network.
        superkeypoints (dict): Path to super-keypoints.
        alpha (float): Hyperparameter balancing prior and evidence in inferencing.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
    """

    def __init__(self,
                 matching_network,
                 superkeypoints=None,
                 alpha=0.5,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(loss_keypoint, train_cfg, test_cfg)
        self.matching_network = MATCHING.build(matching_network)

        self.alpha = alpha
        self.init_weights(superkeypoints)

    def init_weights(self, superkeypoints):
        checkpoint = superkeypoints['checkpoint']
        mname = superkeypoints['mname']

        superkeypoints = torch.load(checkpoint)

        if 'state_dict' in superkeypoints:
            superkeypoints = superkeypoints['state_dict']

        superkeypoints = superkeypoints[mname]
        self.superkeypoints = torch.nn.parameter.Parameter(
            data=superkeypoints, requires_grad=False)

        omega = torch.empty_like(self.superkeypoints)
        nn.init.normal_(omega, std=0.001)
        self.omega = torch.nn.parameter.Parameter(
            data=omega, requires_grad=True)

    def forward_train(self, img, feature, target, target_weight,
                      keypoint_index_onehot, **kwargs):
        """ Global training step. Define the step for training the learnable matching network.

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

        # Compute matching probability. See section 5.2 in the main paper.
        Q = self.matching_network(img, target, target_weight, self.omega)

        # Get ground-truth super-keypoints assignments.
        superkeypoints_target_onehot = keypoint_index_onehot

        losses = dict()
        matching_losses = self.get_loss(Q, target_weight,
                                        superkeypoints_target_onehot)
        losses.update(matching_losses)
        matching_accuracy = self.get_accuracy(Q, target_weight,
                                              superkeypoints_target_onehot)
        losses.update(matching_accuracy)
        return losses

    def forward_test(self, img, feature, target, target_weight, **kwargs):
        # Compute the first term of Eq.(10)
        prototypes_evi = torch.einsum('bchw,blhw->blc', feature, target) \
                        / (target.unsqueeze(2).sum([-1,-2]) + 1e-12)
        prototypes_evi = (prototypes_evi * target_weight).sum(0, keepdim=True) \
                        / (target_weight + 1e-12).sum(0, keepdim=True)

        # Compute the second term of Eq.(10)
        Q = self.matching_network(
            img, target, target_weight, self.omega, fuse_score=True)
        Q = torch.exp(Q)[:, :-1]
        prototypes_pri = torch.einsum('bls,sc->blc', Q, self.superkeypoints)
        prototypes_pri = (prototypes_pri * target_weight).sum(0, keepdim=True) \
                        / (target_weight + 1e-12).sum(0, keepdim=True)

        prototypes = self.alpha * prototypes_evi + (
            1 - self.alpha) * prototypes_pri
        return prototypes

    def forward_global_test(self, keypoint_index_onehot, **kwargs):
        superkeypoints_target_onehot = torch.einsum(
            'bks,sc->bkc', keypoint_index_onehot,
            self.keypoint_2_superkeypoint_onehot)
        keypoints = torch.einsum('bls,sc->blc', superkeypoints_target_onehot,
                                 self.superkeypoints)
        return keypoints

    def get_loss(self, Q, target_weight, superkeypoints_target_onehot):
        bin_target_onehot = 1 - (superkeypoints_target_onehot *
                                 target_weight).sum(1)
        loss = - (Q[:,:-1] * superkeypoints_target_onehot * target_weight).sum([-1,-2]) \
                / (superkeypoints_target_onehot * target_weight).sum([-1,-2])
        loss = loss - (Q[:, -1] *
                       bin_target_onehot).sum(-1) / bin_target_onehot.sum(-1)
        loss = loss.mean()
        return {'matching_loss': loss}

    def get_accuracy(self, Q, target_weight, superkeypoints_target_onehot):
        acc = ((Q[:,:-1].argmax(-1) == superkeypoints_target_onehot.argmax(-1)).float() * target_weight.squeeze(-1)).sum(-1) \
                / target_weight.squeeze(-1).sum(-1)
        acc = acc.mean()
        return {'acc_matching': acc}
