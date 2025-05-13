import torch


def normalize_feats(feats, epsilon=1e-10):
    # (b, w*h, c)
    norms = torch.linalg.norm(feats, dim=-1)[:, :, None]
    norm_feats = feats / (norms + epsilon)
    return norm_feats
