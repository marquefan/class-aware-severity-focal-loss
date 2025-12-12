import torch
from torch import Tensor

def effective_number_weights(
    counts: Tensor, beta: float, normalize_mean1: bool = True, eps: float = 1e-12
) -> Tensor:
    """
    counts: (K,) number of training instances per class (positives).
    Returns (K,) weights w_c^(f) = (1 - beta) / (1 - beta^{n_c}).
    Optionally normalize to have mean 1 for scale stability.
    """
    counts = counts.to(dtype=torch.float32)
    beta = float(beta)
    w = (1.0 - beta) / (1.0 - torch.pow(torch.clamp(torch.tensor(beta), 0.0, 0.999999), counts).clamp_min(eps))
    if normalize_mean1:
        w = w * (counts.numel() / (w.sum().clamp_min(eps)))
    return w
