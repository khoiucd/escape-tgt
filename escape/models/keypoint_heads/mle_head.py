import torch

from mmpose.models import HEADS
from .fewshot_head import FewshotHead


@HEADS.register_module()
class MLEHead(FewshotHead):
    """Maximum likelihood estimate(MLE) head.
    Args:
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
    """

    def __init__(self, loss_keypoint=None, train_cfg=None, test_cfg=None):
        super().__init__(loss_keypoint, train_cfg, test_cfg)

    def forward_test(self, img, feature, target, target_weight, **kwargs):
        # Compute MLE prototypes as in Eq.(2) in the main paper.
        prototypes = torch.einsum('bchw,blhw->blc', feature, target) \
                        / (target.unsqueeze(2).sum([-1,-2]) + 1e-12)
        prototypes = (prototypes * target_weight).sum(0, keepdim=True) \
                        / (target_weight + 1e-12).sum(0, keepdim=True)
        return prototypes
