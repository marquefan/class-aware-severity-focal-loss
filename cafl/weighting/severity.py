import torch
from typing import Dict, Optional

class SeverityMap:
    """
    Holds per-class severity weights. Class IDs are 1..K (as in torchvision targets).
    """
    def __init__(self, num_classes: int, mapping: Optional[Dict[int, float]] = None, normalize_mean1: bool = True):
        self.num_classes = num_classes
        vec = torch.ones(num_classes, dtype=torch.float32)
        if mapping is not None:
            for k, w in mapping.items():
                assert 1 <= k <= num_classes, f"severity key {k} out of range"
                vec[k - 1] = float(w)
        if normalize_mean1:
            vec = vec * (num_classes / vec.sum())
        self._vec = vec

    def vector(self) -> torch.Tensor:
        return self._vec
