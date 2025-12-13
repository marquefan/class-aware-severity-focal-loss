from typing import List, Dict, Optional
import torch
from torch import nn, Tensor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetHead
from ..losses.cafl import CAFLoss
from ..utils.masks import BETWEEN_THRESHOLDS
from ..utils.ddp import global_num_pos

class RetinanetCAFLClassificationHead(RetinaNetClassificationHead):
    """
    Drop-in replacement for torchvision's classification head that calls CAFLoss.
    If `similarity_module` is given, we compute W^(s) = 1 - cos(E,E) per batch
    so gradients flow back into the class embeddings E.
    """
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int,
                 cafl: CAFLoss, similarity_module: Optional[nn.Module] = None):
        super().__init__(in_channels, num_anchors, num_classes)
        self.cafl = cafl
        self.num_classes = num_classes
        self.similarity_module = similarity_module

    def compute_loss(
        self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, List[Tensor]], matched_idxs: List[Tensor]
    ) -> Tensor:
        cls_logits = head_outputs["cls_logits"]  # list of (A_i, K)
        logits_all, targets_all, pos_class_all, valid_all = [], [], [], []
        num_pos_local = 0


        

        for tgt, logits_i, midx in zip(targets, cls_logits, matched_idxs):
            A_i, K = logits_i.shape
            if (midx >= 0).any():
                max_idx = int(midx[midx >= 0].max().item())
                if max_idx >= labels0.numel():
                    raise ValueError(
                        f"matched_idxs points past GT labels: max_idx={max_idx}, num_gt={labels0.numel()}"
                    )
            assert K == self.num_classes
            y = torch.zeros_like(logits_i)
            labels = tgt["labels"].to(dtype=torch.long)       # labels in [1..K]
            labels0 = labels - 1                               # to [0..K-1]
            fg = (midx >= 0)
            bt = (midx == BETWEEN_THRESHOLDS)
            valid = ~bt

            if fg.any():
                y[fg, labels0[midx[fg]]] = 1.0

            pos_class = torch.full((A_i,), -1, dtype=torch.long, device=logits_i.device)
            if fg.any():
                pos_class[fg] = labels0[midx[fg]]

            logits_all.append(logits_i)
            targets_all.append(y)
            pos_class_all.append(pos_class)
            valid_all.append(valid)
            num_pos_local += int(fg.sum().item())

        logits = torch.cat(logits_all, dim=0)
        y = torch.cat(targets_all, dim=0)
        pos_class = torch.cat(pos_class_all, dim=0)
        valid = torch.cat(valid_all, dim=0)

        num_pos_global = global_num_pos(num_pos_local)

        # compute similarity per batch (learnable) if provided
        W_sim = self.similarity_module() if self.similarity_module is not None else None

        total, _ = self.cafl(
            logits=logits, targets=y, pos_class_idx=pos_class, valid_mask=valid,
            num_pos_global=num_pos_global, W_similarity=W_sim
        )
        return total

def swap_in_cafl_head(model: nn.Module, cafl: CAFLoss, similarity_module: Optional[nn.Module] = None) -> nn.Module:
    """
    Replace RetinaNet's classification head with the CAFL version.
    If `similarity_module` is provided, it will be used each batch to produce W^(s).
    """
    in_channels = model.backbone.out_channels
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]
    num_classes = model.head.classification_head.num_classes
    new_cls_head = RetinanetCAFLClassificationHead(in_channels, num_anchors, num_classes, cafl, similarity_module)
    new_head = RetinaNetHead(in_channels, num_anchors, num_classes)
    new_head.classification_head = new_cls_head
    new_head.regression_head = model.head.regression_head
    model.head = new_head
    return model
