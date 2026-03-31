import torch
import pytest
from cafl.losses.cafl import CAFLoss
from cafl.losses.focal import sigmoid_focal_from_logits


def _make_inputs(N=8, K=4, seed=42):
    torch.manual_seed(seed)
    logits = torch.randn(N, K)
    # 3 positives, rest negatives
    pos_class = torch.full((N,), -1, dtype=torch.long)
    pos_class[:3] = torch.tensor([0, 1, 2])
    targets = torch.zeros(N, K)
    for i in range(3):
        targets[i, pos_class[i]] = 1.0
    valid = torch.ones(N, dtype=torch.bool)
    return logits, targets, pos_class, valid


def test_identity_weights_equals_focal():
    """All weights=1 and identity similarity → total == sum(focal) / num_pos."""
    N, K = 8, 4
    logits, targets, pos_class, valid = _make_inputs(N, K)

    loss_fn = CAFLoss(num_classes=K, gamma=2.0)
    # defaults: w_effnum=1, w_severity=1, W_similarity=eye (1 on diag, 1 off-diag in our init)
    # explicitly set similarity to identity (all 1s = no reweighting for positives either)
    loss_fn.set_similarity_matrix(torch.ones(K, K))

    total, parts = loss_fn(logits, targets, pos_class, valid, num_pos_global=3)

    # Manual: focal summed over valid anchors / 3
    focal = sigmoid_focal_from_logits(logits, targets, gamma=2.0)
    expected = focal.sum() / 3
    assert torch.allclose(total, expected, atol=1e-5), \
        f"Expected {expected.item():.6f}, got {total.item():.6f}"


def test_severity_scales_loss():
    """Doubling severity for all classes should roughly double the classification loss on positives."""
    N, K = 8, 4
    logits, targets, pos_class, valid = _make_inputs(N, K)

    loss_fn1 = CAFLoss(num_classes=K)
    loss_fn1.set_similarity_matrix(torch.ones(K, K))
    loss_fn1.set_severity_weights(torch.ones(K))
    total1, _ = loss_fn1(logits, targets, pos_class, valid, num_pos_global=3)

    loss_fn2 = CAFLoss(num_classes=K)
    loss_fn2.set_similarity_matrix(torch.ones(K, K))
    loss_fn2.set_severity_weights(torch.full((K,), 2.0))
    total2, _ = loss_fn2(logits, targets, pos_class, valid, num_pos_global=3)

    # With severity=2 everywhere, the positive anchor contributions are doubled
    # (negatives stay at weight=1 via alpha=1 but severity is only applied to positives)
    assert total2 > total1, "Higher severity should increase loss"


def test_valid_mask_zeros_out():
    """Masking all anchors as invalid should produce zero loss."""
    N, K = 8, 4
    logits, targets, pos_class, _ = _make_inputs(N, K)
    valid = torch.zeros(N, dtype=torch.bool)  # all invalid

    loss_fn = CAFLoss(num_classes=K)
    total, parts = loss_fn(logits, targets, pos_class, valid, num_pos_global=1)
    assert total.item() == 0.0, "All-invalid mask must give zero loss"


def test_no_positives_no_crash():
    """num_pos_global=0 should not divide by zero."""
    N, K = 8, 4
    logits, targets, _, valid = _make_inputs(N, K)
    pos_class = torch.full((N,), -1, dtype=torch.long)  # all negatives

    loss_fn = CAFLoss(num_classes=K)
    total, parts = loss_fn(logits, targets, pos_class, valid, num_pos_global=0)
    assert torch.isfinite(total), "Loss must be finite even with zero positives"


def test_parts_keys():
    N, K = 8, 4
    logits, targets, pos_class, valid = _make_inputs(N, K)
    loss_fn = CAFLoss(num_classes=K)
    _, parts = loss_fn(logits, targets, pos_class, valid, num_pos_global=3)
    for key in ("focal_raw_sum", "alpha_mean", "phi_mean", "loss_sum", "denom", "num_pos", "total"):
        assert key in parts, f"Missing key '{key}' in parts"


def test_num_classes_mismatch_raises():
    loss_fn = CAFLoss(num_classes=4)
    logits = torch.randn(8, 7)  # wrong K
    targets = torch.zeros(8, 7)
    pos_class = torch.full((8,), -1, dtype=torch.long)
    valid = torch.ones(8, dtype=torch.bool)
    with pytest.raises(AssertionError):
        loss_fn(logits, targets, pos_class, valid, num_pos_global=0)
