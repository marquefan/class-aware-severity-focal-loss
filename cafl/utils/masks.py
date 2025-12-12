import torch

BETWEEN_THRESHOLDS = -2  # matches torchvision's convention

def valid_mask_from_matched(matched_idxs: torch.Tensor) -> torch.Tensor:
    return (matched_idxs != BETWEEN_THRESHOLDS)

def positive_mask_from_matched(matched_idxs: torch.Tensor) -> torch.Tensor:
    return (matched_idxs >= 0)
