import torch
import pytest
from cafl.weighting.effective_number import effective_number_weights


def test_equal_counts_equal_weights():
    """Equal class counts should produce equal weights."""
    counts = torch.tensor([100.0, 100.0, 100.0])
    w = effective_number_weights(counts, beta=0.999, normalize_mean1=False)
    assert torch.allclose(w[0], w[1]) and torch.allclose(w[1], w[2]), \
        "Equal counts must yield equal weights"


def test_normalize_mean1():
    counts = torch.tensor([10.0, 50.0, 200.0])
    w = effective_number_weights(counts, beta=0.999, normalize_mean1=True)
    assert abs(w.mean().item() - 1.0) < 1e-4, "Normalized weights should have mean 1"


def test_rare_class_gets_higher_weight():
    """Rare class (count=1) should get higher weight than common class (count=1000)."""
    counts = torch.tensor([1.0, 1000.0])
    w = effective_number_weights(counts, beta=0.999, normalize_mean1=False)
    assert w[0] > w[1], "Rare class must receive higher weight"


def test_zero_counts_no_crash():
    """Zero-count class must not produce NaN or inf."""
    counts = torch.tensor([0.0, 10.0, 50.0])
    w = effective_number_weights(counts, beta=0.999)
    assert torch.isfinite(w).all(), "Weights must be finite even for zero-count classes"


def test_output_shape():
    counts = torch.tensor([5.0, 10.0, 20.0, 40.0])
    w = effective_number_weights(counts, beta=0.99)
    assert w.shape == (4,)
