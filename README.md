# CAFL: Clinical-Aware Focal Loss for RetinaNet

Install (editable):
pip install -e .

Usage:
- Create `CAFLoss`, set effective-number, severity, and similarity each epoch/step.
- Call `swap_in_cafl_head(model, cafl)` to replace RetinaNet's classification head.
- Train as usual; the model will return losses including your CAFL classification loss.

See `examples/retina_train.py` for a minimal working example and ablation configs.
