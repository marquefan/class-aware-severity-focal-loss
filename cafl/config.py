from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class CAFLConfig:
    num_classes: int
    gamma: float = 2.0                    # focal focusing parameter
    beta: float = 0.999                   # effective-number beta
    embed_dim: int = 32                   # class embedding dimension (similarity)
    apply_weights_to_negatives: bool = False  # usually False for stability
    normalize: str = "pos"                # "pos" or "weight_sum"
    severity_map: Optional[Dict[int, float]] = None  # class_id(1..K) -> severity weight
    severity_normalize_mean1: bool = True # rescale severities to have mean 1
    similarity_clamp_eps: float = 1e-6    # numeric clamp for cosine
    epsilon: float = 1e-6                 # numeric clamp for log/pt
    learn_embeddings: bool = True         # set False to freeze E
    warmup_freeze_epochs: int = 0         # >0 to freeze E initially

    def __post_init__(self):
        if not (0.0 < self.beta < 1.0):
            raise ValueError(f"beta must be in (0, 1), got {self.beta}")
        if self.gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {self.gamma}")
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {self.embed_dim}")
        if self.normalize not in ("pos", "weight_sum"):
            raise ValueError(f"normalize must be 'pos' or 'weight_sum', got {self.normalize!r}")
