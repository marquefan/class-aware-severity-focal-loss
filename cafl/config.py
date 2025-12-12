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
