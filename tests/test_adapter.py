import torch
import pytest
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from cafl.losses.cafl import CAFLoss
from cafl.adapters.torchvision_retinanet import swap_in_cafl_head, RetinanetCAFLClassificationHead


NUM_CLASSES = 7


def _make_model():
    model = retinanet_resnet50_fpn_v2(weights=None, num_classes=NUM_CLASSES)
    cafl = CAFLoss(num_classes=NUM_CLASSES)
    return swap_in_cafl_head(model, cafl), cafl


def test_swap_replaces_head():
    model, _ = _make_model()
    assert isinstance(model.head.classification_head, RetinanetCAFLClassificationHead)


def test_forward_no_crash():
    """Model with CAFL head should forward without error on a dummy image+target."""
    model, _ = _make_model()
    model.train()
    images = [torch.rand(3, 224, 224)]
    targets = [{
        "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        "labels": torch.tensor([1], dtype=torch.long),
    }]
    loss_dict = model(images, targets)
    assert "classification" in loss_dict
    assert "bbox_regression" in loss_dict
    assert torch.isfinite(loss_dict["classification"])


def test_last_parts_populated_after_forward():
    """After a forward pass, last_parts should be set on the classification head."""
    model, _ = _make_model()
    model.train()
    images = [torch.rand(3, 224, 224)]
    targets = [{
        "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        "labels": torch.tensor([1], dtype=torch.long),
    }]
    model(images, targets)
    cls_head = model.head.classification_head
    assert hasattr(cls_head, "last_parts"), "last_parts should be set after forward"
    assert "alpha_mean" in cls_head.last_parts


def test_bounds_check_does_not_crash_on_valid_input():
    """The fixed bounds check should not crash when matched_idx is within range."""
    model, _ = _make_model()
    model.train()
    images = [torch.rand(3, 300, 300)]
    targets = [{
        "boxes": torch.tensor([[20.0, 20.0, 80.0, 80.0], [100.0, 100.0, 200.0, 200.0]]),
        "labels": torch.tensor([2, 5], dtype=torch.long),
    }]
    # Should not raise
    loss_dict = model(images, targets)
    assert torch.isfinite(loss_dict["classification"])
