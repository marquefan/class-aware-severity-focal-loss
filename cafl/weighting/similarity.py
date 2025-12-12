import torch
from torch import nn, Tensor

class ClassEmbeddingSimilarity(nn.Module):
    """
    Learnable class embeddings E in R^{K x d}; returns W^(s) = 1 - cos(E, E).
    """
    def __init__(self, num_classes: int, embed_dim: int = 32, clamp_eps: float = 1e-6, learn: bool = True):
        super().__init__()
        self.clamp_eps = clamp_eps
        E = torch.randn(num_classes, embed_dim) * 0.02
        self.E = nn.Parameter(E, requires_grad=learn)

    @torch.no_grad()
    def freeze(self):
        self.E.requires_grad_(False)

    @torch.no_grad()
    def unfreeze(self):
        self.E.requires_grad_(True)

    def forward(self) -> Tensor:
        # normalize rows
        E = self.E
        E = E / (E.norm(dim=1, keepdim=True).clamp_min(self.clamp_eps))
        # cosine matrix (K x K)
        S = E @ E.t()
        S = S.clamp(-1.0 + self.clamp_eps, 1.0 - self.clamp_eps)
        W = 1.0 - S
        return W  # (K, K)
