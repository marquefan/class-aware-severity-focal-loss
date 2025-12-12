from typing import Dict
import torch

def loss_parts_to_log(parts: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(v.detach().cpu()) for k, v in parts.items()}
