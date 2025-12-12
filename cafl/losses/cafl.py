from typing import Dict, Optional, Tuple
import torch
from torch import nn, Tensor

from .focal import sigmoid_focal_from_logits

class CAFLoss(nn.Module):
    """
    Clinical-Aware Focal Loss:
      L = w_f(c) * w_m(c) * phi(c,k) * FL(logits, targets)
    where phi(c,k) = 1 if k==c else w_s(c,k).
    """
    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        epsilon: float = 1e-6,
        apply_weights_to_negatives: bool = False,
        normalize: str = "pos",  # "pos" or "weight_sum"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.epsilon = epsilon
        self.apply_weights_to_negatives = apply_weights_to_negatives
        assert normalize in ("pos", "weight_sum")
        self.normalize = normalize

        # Buffers set externally each epoch / step
        self.register_buffer("w_effnum", torch.ones(num_classes), persistent=False)  # (K,)
        self.register_buffer("w_severity", torch.ones(num_classes), persistent=False) # (K,)
        self.register_buffer("W_similarity", torch.ones(num_classes, num_classes), persistent=False) # (K,K)

    @torch.no_grad()
    def set_effective_number_weights(self, w: Tensor):
        assert w.shape == (self.num_classes,)
        self.w_effnum = w.to(self.w_effnum.device, dtype=torch.float32)

    @torch.no_grad()
    def set_severity_weights(self, w: Tensor):
        assert w.shape == (self.num_classes,)
        self.w_severity = w.to(self.w_severity.device, dtype=torch.float32)

    @torch.no_grad()
    def set_similarity_matrix(self, W: Tensor):
        assert W.shape == (self.num_classes, self.num_classes)
        self.W_similarity = W.to(self.W_similarity.device, dtype=torch.float32)

    def forward(
        self,
        logits: Tensor,          # (N,K) valid anchors only or all with mask
        targets: Tensor,         # (N,K) in {0,1}
        pos_class_idx: Tensor,   # (N,) int64, -1 for negatives, else class index in [0..K-1]
        valid_mask: Tensor,      # (N,) boolean (True=use)
        num_pos_global: int,     # integer normalizer (DDP global positives)
        W_similarity: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Returns: (scalar total loss, parts dict)
        """
        device = logits.device
        N, K = logits.shape
        assert K == self.num_classes, "num_classes mismatch"
        assert targets.shape == logits.shape
        assert valid_mask.shape[0] == N and pos_class_idx.shape[0] == N

        # base focal per-(N,K)
        focal = sigmoid_focal_from_logits(logits, targets, gamma=self.gamma, eps=self.epsilon)

        # build alpha_star = w_f(c)*w_m(c) per anchor (positives only unless configured)
        alpha_anchor = torch.ones(N, dtype=torch.float32, device=device)
        pos_mask = pos_class_idx >= 0  # positives among VALID anchors
        pos_idx = pos_class_idx[pos_mask].clamp(min=0)

        if pos_mask.any():
            wf = self.w_effnum[pos_idx]             # (num_pos,)
            wm = self.w_severity[pos_idx]           # (num_pos,)
            alpha_anchor[pos_mask] = wf * wm

        if self.apply_weights_to_negatives:
            # Optionally, you could define background weights; by default keep 1
            pass

        # Similarity gate phi(c,k)
        # For positives: phi_k = 1 for true class, w_s[c,k] for others
        phi = torch.ones_like(focal)  # (N,K)
        if pos_mask.any():
            W = W_similarity if W_similarity is not None else self.W_similarity            # Gather similarity rows for the GT class of each positive anchor
            rows = W[pos_class_idx[pos_mask]]  # (num_pos, K)
            phi[pos_mask] = rows
            # Set diagonal (true class) to 1
            phi[pos_mask, pos_class_idx[pos_mask]] = 1.0

        # Broadcast alpha per anchor to (N,K)
        alpha = alpha_anchor.unsqueeze(1).expand_as(focal)

        # assemble weighted loss
        loss_mat = alpha * phi * focal

        # mask invalid anchors entirely
        if valid_mask is not None:
            vm = valid_mask.to(dtype=focal.dtype, device=device).unsqueeze(1)
            loss_mat = loss_mat * vm

        # reduction / normalization
        if self.normalize == "pos":
            denom = max(1, int(num_pos_global))
        else:
            denom = float(loss_mat.detach().sum().clamp_min(self.epsilon))

        total = loss_mat.sum() / denom

        parts = {
            "focal_raw_sum": focal.sum(),
            "alpha_mean": alpha_anchor.mean(),
            "phi_mean": phi.mean(),
            "loss_sum": loss_mat.sum(),
            "denom": torch.tensor(float(denom), device=device),
            "num_pos": torch.tensor(float(num_pos_global), device=device),
            "total": total.detach(),
        }
        return total, parts
