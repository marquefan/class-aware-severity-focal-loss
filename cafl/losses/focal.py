import torch
from torch import Tensor

def sigmoid_focal_from_logits(
    logits: Tensor, targets: Tensor, gamma: float = 2.0, eps: float = 1e-6
) -> Tensor:
    """
    Vectorized sigmoid focal term in logits space.
    logits: (N, K), targets in {0,1} same shape.
    Returns per-(N,K) loss WITHOUT any alpha/weights and WITHOUT reduction.
    """
    # sigmoid
    p = torch.sigmoid(logits)
    # pt = p if y=1 else (1-p)
    pt = torch.where(targets > 0.5, p, 1.0 - p)
    # numerical clamps
    pt = pt.clamp(min=eps, max=1.0 - eps)
    focal = (1.0 - pt).pow(gamma)
    return -focal * torch.log(pt)
