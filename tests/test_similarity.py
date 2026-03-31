import torch
import pytest
from cafl.weighting.similarity import ClassEmbeddingSimilarity


def test_output_shape():
    sim = ClassEmbeddingSimilarity(num_classes=7, embed_dim=32)
    W = sim()
    assert W.shape == (7, 7), f"Expected (7,7), got {W.shape}"


def test_diagonal_is_zero():
    """W[c,c] = 1 - cos(e_c, e_c) = 1 - 1 = 0."""
    sim = ClassEmbeddingSimilarity(num_classes=5, embed_dim=16)
    W = sim()
    diag = torch.diagonal(W)
    assert torch.allclose(diag, torch.zeros(5), atol=1e-5), \
        f"Diagonal should be ~0, got {diag}"


def test_values_in_range():
    """Cosine similarity ∈ [-1,1], so W = 1 - cos ∈ [0,2]."""
    sim = ClassEmbeddingSimilarity(num_classes=6, embed_dim=8)
    W = sim()
    assert (W >= -1e-5).all(), "W values must be >= 0"
    assert (W <= 2.0 + 1e-5).all(), "W values must be <= 2"


def test_freeze_unfreeze():
    sim = ClassEmbeddingSimilarity(num_classes=4, embed_dim=8, learn=True)
    assert sim.E.requires_grad
    sim.freeze()
    assert not sim.E.requires_grad
    sim.unfreeze()
    assert sim.E.requires_grad


def test_gradients_flow():
    sim = ClassEmbeddingSimilarity(num_classes=4, embed_dim=8, learn=True)
    W = sim()
    loss = W.sum()
    loss.backward()
    assert sim.E.grad is not None, "Gradients should flow back to embeddings"


def test_frozen_no_gradients():
    """When frozen, E should not require grad and W should not be part of the autograd graph."""
    sim = ClassEmbeddingSimilarity(num_classes=4, embed_dim=8, learn=True)
    sim.freeze()
    assert not sim.E.requires_grad, "E should not require grad after freeze()"
    W = sim()
    assert not W.requires_grad, "Output W should not require grad when E is frozen"
