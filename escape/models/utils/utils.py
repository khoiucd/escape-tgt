import torch
import torch.nn.functional as F
from collections import OrderedDict
from mmcv.runner.checkpoint import _load_checkpoint


def get_backbone_deconv_state_dict(pretrained):
    checkpoint = _load_checkpoint(pretrained, 'cpu')
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {pretrained}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict_tmp = checkpoint['state_dict']
    else:
        state_dict_tmp = checkpoint
    backbone, deconv = OrderedDict(), None
    # strip prefix of state_dict
    for k, v in state_dict_tmp.items():
        if k.startswith('module.backbone.'):
            backbone[k[16:]] = v
        elif k.startswith('module.'):
            backbone[k[7:]] = v
        elif k.startswith('backbone.'):
            backbone[k[9:]] = v
        elif k.startswith('deconv.'):
            if deconv is None:
                deconv = OrderedDict()
            deconv[k[7:]] = v
        else:
            backbone[k] = v
    return backbone, deconv


def get_one_hot(y_s, num_classes):
    eye = torch.eye(num_classes).to(y_s.device)
    one_hot = []
    for y_task in y_s:
        one_hot.append(eye[y_task].unsqueeze(0))
    one_hot = torch.cat(one_hot, 0)
    return one_hot


def norm_clamp(input, max=100):
    input_norm = torch.sqrt(
        (input**2).sum(-1)).max(-1)[0].unsqueeze(-1).unsqueeze(-1).detach()
    input_ceil = torch.clamp(input_norm, max=max)
    return input / input_norm * input_ceil


def KNN_cluster(keypoints, meta, NN_num=6, threshold=0.9):
    print(f"cluster {meta['keypoint_num']} keypoints")
    keypoint_index_2_class = {}
    internal_index = {}
    for c, keypoint_index in meta['keypoint_index'].items():
        keypoint_index_2_class.update({k: c for k in keypoint_index})
        internal_index.update({k: e for e, k in enumerate(keypoint_index)})

    keypoints = torch.from_numpy(keypoints)

    normalized_keypoints = F.normalize(keypoints, dim=-1)
    cosine_sim = normalized_keypoints @ normalized_keypoints.transpose(0, 1)

    for k in range(meta['keypoint_num']):
        c = keypoint_index_2_class[k]
        cosine_sim[k][meta['keypoint_index'][c]] = -2

    values, _ = torch.topk(cosine_sim, NN_num, dim=-1)

    scores = torch.where(values > threshold, values, 0).sum(-1) / (
        torch.where(values > threshold, 1, 0).sum(-1) + 1e-12)

    superkeypoints = []
    superkeypoint_num = 0
    keypoint_2_superkeypoint = [-1 for _ in range(meta['keypoint_num'])]
    while scores.max() > -2:
        peak = scores.argmax().item()
        new_superkeypoint = [peak]
        neighbor_sim = cosine_sim[peak]
        while neighbor_sim.max() > threshold:
            neighbor = neighbor_sim.argmax().item()
            new_superkeypoint.append(neighbor)
            neighbor_sim[meta['keypoint_index'][
                keypoint_index_2_class[neighbor]]] = -2

        scores[new_superkeypoint] = -2
        for k in new_superkeypoint:
            keypoint_2_superkeypoint[k] = superkeypoint_num

        new_superkeypoint = keypoints[peak].view(-1,
                                                 keypoints.shape[-1]).mean(0)
        superkeypoints.append(new_superkeypoint)
        superkeypoint_num += 1

    superkeypoints = torch.stack(superkeypoints, 0)

    print(f'number of super-keypoints: {superkeypoint_num}')
    return superkeypoints, keypoint_2_superkeypoint, superkeypoint_num


def logsumexp_stable(x, tg_w, dim=1):
    x = torch.where(tg_w > 0, x, -float('inf'))
    maxX = x.max(dim)[0].detach()
    x = torch.where(tg_w > 0, (x - maxX.unsqueeze(dim)).exp(),
                    0).sum(dim).log()
    x = x + maxX
    return x


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor,
                            log_nu: torch.Tensor, iters: int,
                            tg_w: torch.Tensor) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    batch_size, m, n = Z.size()
    Z = torch.where(tg_w > 0, Z, 0)
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = torch.where(
            tg_w.sum(2) > 0,
            log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2), 0)
        v = log_nu - logsumexp_stable(Z + u.unsqueeze(2), tg_w, dim=1)
    Z = (Z + u.unsqueeze(2) + v.unsqueeze(1))
    return Z


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor,
                          iters: int, tg_w: torch.Tensor) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = torch.ones(1)
    ms, ns = tg_w.sum(1), (n * one).to(scores)
    tg_w = torch.cat([tg_w, torch.ones(b, 1, 1).to(tg_w.device)], dim=1)

    bins = alpha.expand(b, 1, n)

    couplings = torch.cat([scores, bins], dim=1)

    norm = -(ns).log()
    log_mu = torch.cat([norm.expand(b, m), (ns[None] - ms).log() + norm],
                       dim=-1)
    log_nu = norm[None].expand(b, n)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters, tg_w)
    Z = Z - norm.unsqueeze(-1)
    return Z
