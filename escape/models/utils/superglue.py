import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.runner import BaseModule
from mmcv.runner.checkpoint import load_state_dict
from mmcv.utils import Registry

from mmpose.models import builder
from mmpose.models.utils.ops import resize
from .modules import GraphAttentionNetwork
from .utils import (get_backbone_deconv_state_dict, log_optimal_transport,
                    norm_clamp)

MATCHING = Registry('MatchingNetwork')


def posemb_sincos_1d(patches, dim=256, temperature=10000, dtype=torch.float32):
    _, n, _, device, dtype = *patches.shape, patches.device, patches.dtype

    n = torch.arange(n, device=device)
    assert (dim %
            2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1. / (temperature**omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim=1)
    return pe.type(dtype)


@MATCHING.register_module()
class SuperGlue(BaseModule):

    def __init__(self,
                 backbone,
                 deconv,
                 pretrained,
                 in_channels=2048,
                 dim=256,
                 depth=6,
                 heads=8,
                 mlp_dim=2048,
                 dim_head=64):
        super().__init__()
        self.backbone = builder.build_backbone(backbone)
        self.deconv = builder.build_backbone(deconv)
        self.dim = dim

        self.to_embedding1 = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, dim),
            nn.LayerNorm(dim),
        )

        self.to_embedding2 = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, dim),
            nn.LayerNorm(dim),
        )

        self.GAT = GraphAttentionNetwork(dim, depth, heads, dim_head, mlp_dim)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        if pretrained is not None:
            backbone, deconv = get_backbone_deconv_state_dict(pretrained)
            load_state_dict(self.backbone, backbone)
            self.deconv.init_weights(deconv)

    def get_pe_geo(self, feature, target, target_weight):
        feature = rearrange(feature, 'b c h w -> b (h w) c')
        target = rearrange(target, 'b l h w -> b (h w) l')
        target = target.argmax(1)
        PE_geo = posemb_sincos_1d(feature, dim=self.dim)[target]
        PE_geo = torch.where(target_weight > 0, PE_geo + 1, 0)
        return PE_geo

    def extract_feature(self, img):
        feature = self.backbone(img)
        feature = self.deconv(feature)
        return feature

    def forward(self, img, target, target_weight, omega, fuse_score=False):
        batch_size, _, img_height, img_width = img.shape
        feature = self.extract_feature(img)

        keypoint_feature = torch.einsum('bchw,blhw->blc', feature, target) \
                        / (target.unsqueeze(2).sum([-1,-2]) + 1e-12)
        PE_geo = self.get_pe_geo(feature, target, target_weight)

        omega = omega.unsqueeze(0)
        PE_abs = posemb_sincos_1d(omega, dim=self.dim)

        keypoint_feature = self.to_embedding1(keypoint_feature)
        keypoint_feature = keypoint_feature + PE_geo

        omega = self.to_embedding2(omega)
        omega = omega + PE_abs

        x, y = self.GAT(keypoint_feature, omega)
        y = norm_clamp(y, 100)
        # x = norm_clamp(x, 100)

        scores = torch.einsum('bnc,bmc->bnm', x, y)
        # score_max = scores.view(batch_size, -1).max(-1)[0].mean()
        scores = scores / self.dim**.5

        if fuse_score:
            scores = (scores * target_weight).sum(
                0, keepdim=True) / (
                    target_weight.sum(0, keepdim=True) + 1e-12)
            target_weight = (target_weight.sum(0, keepdim=True) > 0).float()

        Q = log_optimal_transport(scores, self.bin_score, 50, target_weight)

        return Q
