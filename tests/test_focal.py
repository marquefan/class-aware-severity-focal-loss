import torch
import pytest
from cafl.losses.focal import sigmoid_focal_from_logits


def test_gamma0_equals_bce():
    """With gamma=0, focal loss should equal binary cross-entropy."""
    torch.manual_seed(0)
    logits = torch.randn(8, 4)
    targets = (torch.rand(8, 4) > 0.5).float()

    focal = sigmoid_focal_from_logits(logits, targets, gamma=0.0)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    assert torch.allclose(focal, bce, atol=1e-5), "gamma=0 focal must equal BCE"


def test_output_shape():
    logits = torch.randn(10, 7)
    targets = torch.zeros(10, 7)
    out = sigmoid_focal_from_logits(logits, targets, gamma=2.0)
    assert out.shape == (10, 7)


def test_nonnegative():
    torch.manual_seed(1)
    logits = torch.randn(16, 5)
    targets = (torch.rand(16, 5) > 0.5).float()
    out = sigmoid_focal_from_logits(logits, targets, gamma=2.0)
    assert (out >= 0).all(), "focal loss values must be non-negative"


def test_perfect_predictions_near_zero():
    """Very confident correct predictions should have near-zero focal loss (gamma>0)."""
    # Predict class 0 confidently: large positive logit for class 0, target 1
    logits = torch.tensor([[10.0, -10.0]])
    targets = torch.tensor([[1.0, 0.0]])
    out = sigmoid_focal_from_logits(logits, targets, gamma=2.0)
    assert out.max() < 0.01, "Confident correct predictions should have tiny focal loss"
