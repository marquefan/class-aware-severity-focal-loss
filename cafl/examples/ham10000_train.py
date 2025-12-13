"""
Train RetinaNet+CAFL on HAM10000 (bounding boxes).
Adjust the paths, severity map, and hyperparams as needed.
"""
import os
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Callable

import torch
from torch.utils.data import DataLoader

from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from tqdm.auto import tqdm

from cafl import (
    CAFLConfig, CAFLoss,
    effective_number_weights, SeverityMap, ClassEmbeddingSimilarity,
)
from cafl.adapters.torchvision_retinanet import swap_in_cafl_head
from cafl.examples.ham10000_detection import (
    HAM10000Detection, DEFAULT_CLASS_MAP, det_collate
)

#helps with fragmented memory on long runs
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
# modest matmul precision = less VRAM (Ampere+)
try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass

# ------------- Class & severity setup ----------------------------------------
CLASS_MAP = DEFAULT_CLASS_MAP 

SEVERITY = {
    "akiec": 3,
    "bcc":   3,
    "bkl":   1,
    "df":    1,
    "nv":    1,
    "mel":   4,   # melanoma urgent
    "vasc":  2,
}
SEVERITY_ID = {CLASS_MAP[k]: v for k, v in SEVERITY.items()}

# ------------- Minimal training params ---------------------------------------
BATCH_SIZE = 4
LR = 2e-4
EPOCHS = 5
NUM_WORKERS = 4

def _detect_repo_root() -> Path:
    """Try to locate the repo root (looks for pyproject) or fall back to parent dirs."""
    resolved = Path(__file__).resolve()
    for parent in resolved.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    parent_list = list(resolved.parents)
    return parent_list[-1] if parent_list else resolved.parent


REPO_ROOT = _detect_repo_root()


def repo_path(*parts: str) -> Path:
    """Resolve a path relative to the repository root."""
    return REPO_ROOT.joinpath(*parts)


def log_path(label: str, path: Path) -> None:
    """Print a friendly path + existence check for quick debugging."""
    p = Path(path).expanduser()
    resolved = p.resolve()
    print(f"{label}: {resolved} (exists: {resolved.exists()})")


CONFIG_ENV_FILE = repo_path("config.env")


def _path_from_config(value: str) -> Path:
    """Coerce config paths, treating relative values as repo-root relative."""
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    return repo_path(value)


def _bool_from_config(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config_env(path: Path) -> Dict[str, str]:
    """Parse a simple KEY=VALUE config file (shell-style, no exports)."""
    if not path.exists():
        return {}
    data: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, val = stripped.split("=", 1)
            data[key.strip()] = val.strip()
    return data


def env_defaults_from_config() -> Dict[str, Any]:
    """Convert config.env values into argparse defaults."""
    raw = load_config_env(CONFIG_ENV_FILE)
    if not raw:
        return {}

    def _get(key: str, caster: Callable[[str], Any]) -> Optional[Any]:
        if key not in raw:
            return None
        try:
            return caster(raw[key])
        except Exception:
            print(f"[config.env] Failed to parse {key}={raw[key]!r}, using fallback defaults.")
            return None

    mapping: Dict[str, Any] = {}
    config_spec = {
        "images_dir": ("IMAGES_DIR", _path_from_config),
        "train_csv": ("TRAIN_CSV", _path_from_config),
        "val_csv": ("VAL_CSV", _path_from_config),
        "epochs": ("EPOCHS", int),
        "batch_size": ("BATCH_SIZE", int),
        "accum_steps": ("ACCUM_STEPS", int),
        "num_workers": ("NUM_WORKERS", int),
        "lr": ("LR", float),
        "weight_decay": ("WEIGHT_DECAY", float),
        "beta": ("BETA", float),
        "gamma": ("GAMMA", float),
        "embed_dim": ("EMBED_DIM", int),
        "warmup_freeze_epochs": ("WARMUP_FREEZE_EPOCHS", int),
        "seed": ("SEED", int),
        "min_size": ("MIN_SIZE", int),
        "max_size": ("MAX_SIZE", int),
        "wandb": ("WANDB", _bool_from_config),
        "wandb_project": ("WANDB_PROJECT", str),
        "wandb_entity": ("WANDB_ENTITY", str),
        "wandb_run_name": ("WANDB_RUN_NAME", str),
        "wandb_mode": ("WANDB_MODE", str),
        "disable_cafl": ("DISABLE_CAFL", _bool_from_config),
    }

    for dest, (key, caster) in config_spec.items():
        val = _get(key, caster)
        if val is not None and val != "":
            mapping[dest] = val
    if mapping:
        print(f"Loaded defaults from {CONFIG_ENV_FILE}")
    return mapping


def init_wandb_run(
    args: argparse.Namespace,
    train_ds: HAM10000Detection,
    val_ds: Optional[HAM10000Detection],
    num_classes: int,
    model: torch.nn.Module,
) -> Optional[Any]:
    """Initialize a Weights & Biases run when requested."""
    if not getattr(args, "wandb", False):
        return None

    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Weights & Biases logging requested but the package is missing. "
            "Install it via `pip install wandb`."
        ) from exc

    train_counts = train_ds.class_counts()
    val_size = len(val_ds) if val_ds is not None else 0

    config: Dict[str, Any] = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "beta": args.beta,
        "gamma": args.gamma,
        "embed_dim": args.embed_dim,
        "warmup_freeze_epochs": args.warmup_freeze_epochs,
        "min_size": args.min_size,
        "max_size": args.max_size,
        "seed": args.seed,
        "num_classes": num_classes,
        "train_size": len(train_ds),
        "val_size": val_size,
        "class_map": CLASS_MAP,
        "severity_map": SEVERITY,
        "train_class_counts": train_counts,
        "images_dir": str(args.images_dir),
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv) if args.val_csv else None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "disable_cafl": args.disable_cafl,
    }

    mode = args.wandb_mode or os.environ.get("WANDB_MODE")
    wandb.init(
        project=args.wandb_project or "cafl",
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=mode,
        config=config,
        settings=wandb.Settings(start_method="fork"),
    )
    
    log_freq = max(1, len(train_ds) // max(1, args.batch_size))
    wandb.watch(model, log="gradients", log_freq=log_freq)
    return wandb


def configure_wandb_metrics(wandb_run) -> None:
    if wandb_run is None:
        return
    wandb_run.define_metric("epoch", hidden=True)
    wandb_run.define_metric("optimizer/lr", summary="last", step_metric="epoch")

    for metric in (
        "train/loss",
        "train/cls_loss",
        "train/bbox_loss",
    ):
        wandb_run.define_metric(metric, step_metric="epoch")

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.
    boxes1: (N, 4), boxes2: (M, 4) in (x1, y1, x2, y2) format.
    Returns: (N, M) IoU matrix.
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))

    # areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    # intersections
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])   # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])   # (N, M, 2)
    wh = (rb - lt).clamp(min=0)                          # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]                    # (N, M)

    # unions
    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou


def evaluate_detector(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    iou_thresh: float = 0.5,
    score_thresh: float = 0.5,
) -> Dict[str, Any]:
    """
    Simple detection evaluation:
    - Per-class TP / FP / FN at IoU >= iou_thresh and score >= score_thresh.
    - Melanoma recall / precision.
    - Severity-weighted false negatives (sum of severity for missed GTs).
    """
    model.eval()
    num_classes = len(CLASS_MAP)
    mel_class_id = CLASS_MAP["mel"]

    stats = {
        c: {"tp": 0, "fp": 0, "fn": 0}
        for c in range(num_classes)
    }
    severity_weighted_fn = 0.0
    total_gt = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for out, tgt in zip(outputs, targets):
                gt_boxes = tgt["boxes"]
                gt_labels = tgt["labels"]

                total_gt += gt_labels.numel()

                if gt_boxes.numel() == 0:
                    # no GT in this image; all predictions are false positives
                    pred_labels = out["labels"]
                    pred_scores = out["scores"]
                    keep = pred_scores >= score_thresh
                    for c in pred_labels[keep].tolist():
                        stats[c]["fp"] += 1
                    continue

                pred_boxes = out["boxes"]
                pred_labels = out["labels"]
                pred_scores = out["scores"]

                # Filter low-confidence predictions
                keep = pred_scores >= score_thresh
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                pred_scores = pred_scores[keep]

                if pred_boxes.numel() == 0:
                    # no predictions -> all GT are FN
                    for c in gt_labels.tolist():
                        stats[c]["fn"] += 1
                        severity_weighted_fn += SEVERITY_ID.get(int(c), 1)
                    continue

                # Greedy matching per class
                for c in range(num_classes):
                    gt_idx = (gt_labels == c).nonzero(as_tuple=False).flatten()
                    pred_idx = (pred_labels == c).nonzero(as_tuple=False).flatten()

                    if gt_idx.numel() == 0 and pred_idx.numel() == 0:
                        continue

                    if gt_idx.numel() == 0:
                        # only predictions -> all FP
                        stats[c]["fp"] += int(pred_idx.numel())
                        continue

                    gt_c_boxes = gt_boxes[gt_idx]
                    pred_c_boxes = pred_boxes[pred_idx]

                    if pred_c_boxes.numel() == 0:
                        # only GT -> all FN
                        stats[c]["fn"] += int(gt_idx.numel())
                        severity_weighted_fn += SEVERITY_ID.get(int(c), 1) * int(gt_idx.numel())
                        continue

                    ious = box_iou(pred_c_boxes, gt_c_boxes)  # (P, G)
                    # record matches
                    matched_gt = set()
                    # iterate predictions in descending score order
                    scores_c = pred_scores[pred_idx]
                    order = torch.argsort(scores_c, descending=True)

                    for pi in order.tolist():
                        iou_row = ious[pi]
                        best_iou, gi = (iou_row.max(dim=0))
                        gi = int(gi.item())
                        if best_iou >= iou_thresh and gi not in matched_gt:
                            # true positive
                            stats[c]["tp"] += 1
                            matched_gt.add(gi)
                        else:
                            # false positive
                            stats[c]["fp"] += 1

                    # any GT not matched -> FN
                    missed = gt_idx.numel() - len(matched_gt)
                    if missed > 0:
                        stats[c]["fn"] += missed
                        severity_weighted_fn += SEVERITY_ID.get(int(c), 1) * missed

    # derive precision / recall per class
    per_class = {}
    for c in range(num_classes):
        tp = stats[c]["tp"]
        fp = stats[c]["fp"]
        fn = stats[c]["fn"]
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        per_class[c] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
        }

    mel_stats = per_class[mel_class_id]
    mel_precision = mel_stats["precision"]
    mel_recall = mel_stats["recall"]

    overall_tp = sum(s["tp"] for s in stats.values())
    overall_fp = sum(s["fp"] for s in stats.values())
    overall_fn = sum(s["fn"] for s in stats.values())

    overall_precision = overall_tp / (overall_tp + overall_fp + 1e-6)
    overall_recall = overall_tp / (overall_tp + overall_fn + 1e-6)

    return {
        "per_class": per_class,                       # dict[class_id] -> stats
        "mel_precision": mel_precision,
        "mel_recall": mel_recall,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "severity_weighted_fn": severity_weighted_fn,
        "total_gt": total_gt,
    }




def parse_args():
    # Defaults: look relative to repo root so local runs just work.
    default_images = repo_path("HAM10000_images")
    default_csv    = repo_path("ham10000_bboxes.csv")

    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", type=Path, default=default_images,
                   help="Directory with HAM10000 images.")
    p.add_argument("--train-csv", type=Path, default=default_csv,
                   help="CSV with bounding boxes + labels for training.")
    p.add_argument("--val-csv", type=Path, default=None,
                   help="Optional CSV for validation.")

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)      
    p.add_argument("--accum-steps", type=int, default=1,           # set >1 to simulate bigger batch
                   help="Gradient accumulation steps")
    p.add_argument("--num-workers", type=int, default=2)           # keep modest to fit most machines
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.999)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--warmup-freeze-epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=17)

    # Resize used by detector (much cheaper than 800/1333)
    p.add_argument("--min-size", type=int, default=450)
    p.add_argument("--max-size", type=int, default=600)

    # Wandb logging
    p.add_argument("--wandb", action="store_true",
                   help="Enable Weights & Biases experiment logging.")
    p.add_argument("--wandb-project", type=str, default=None,
                   help="W&B project name (defaults to 'cafl').")
    p.add_argument("--wandb-entity", type=str, default=None,
                   help="Optional W&B entity (team).")
    p.add_argument("--wandb-run-name", type=str, default=None,
                   help="Optional custom run name.")
    p.add_argument("--wandb-mode", type=str, default=None,
                   help="Override W&B mode (online/offline/disabled).")
    p.add_argument("--disable-cafl", action="store_true",
                   help="Skip CAFL head/losses and run vanilla RetinaNet.")
    
        # Loss / ablation mode
    p.add_argument(
        "--loss-mode",
        type=str,
        default="full_cafl",
        choices=[
            "baseline",         # A0: vanilla RetinaNet focal loss
            "en_only",          # A1: CAFL, effective-number only
            "severity_only",    # A2: CAFL, severity only
            "en_severity",      # A3: CAFL, EN + severity (no similarity)
            "full_cafl",        # A4: CAFL, EN + severity + similarity
            "similarity_only",  # A5: CAFL, similarity only
        ],
        help=(
            "Which loss configuration to use. "
            "'baseline' uses vanilla RetinaNet focal loss. "
            "Other modes enable CAFL with different components."
        ),
    )


    env_defaults = env_defaults_from_config()
    if env_defaults:
        p.set_defaults(**env_defaults)

    return p.parse_args()


def main():
    args = parse_args()

    # Helpful logs
    log_path("Images dir", args.images_dir)
    log_path("Train CSV",  args.train_csv)
    if args.val_csv:
        log_path("Val CSV", args.val_csv)

    # Basic existence checks
    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {args.images_dir}")
    if not args.train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {args.train_csv}")
    if args.val_csv and not Path(args.val_csv).exists():
        raise FileNotFoundError(f"Val CSV not found: {args.val_csv}")

    # Repro-ish defaults
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASS_MAP)

    # ------------------- Datasets & loaders -----------------------------------
    train_ds = HAM10000Detection(args.images_dir, args.train_csv, class_map=CLASS_MAP)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=det_collate,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_ds = None
    val_loader = None
    if args.val_csv:
        val_ds = HAM10000Detection(args.images_dir, Path(args.val_csv), class_map=CLASS_MAP)
        val_loader = DataLoader(
            val_ds,
            batch_size=max(1, args.batch_size // 2),
            shuffle=False,
            collate_fn=det_collate,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # Enable CAFL for all modes except the pure baseline
    cafl_enabled = (args.loss_mode != "baseline") and (not args.disable_cafl)


    # ------------------- Model & CAFL wiring ----------------------------------
    model = retinanet_resnet50_fpn_v2(weights=None, num_classes=num_classes)
    image_mean = getattr(model.transform, "image_mean", [0.485, 0.456, 0.406])
    image_std  = getattr(model.transform, "image_std",  [0.229, 0.224, 0.225])
    model.transform = GeneralizedRCNNTransform(
        min_size=450,
        max_size=600,
        image_mean=image_mean,
        image_std=image_std,
        size_divisible=32
    )

    cfg = None
    cafl = None
    similarity = None

    cfg = None
    cafl = None
    similarity = None

    if cafl_enabled:
        cfg = CAFLConfig(
            num_classes=num_classes,
            gamma=args.gamma,
            beta=args.beta,
            embed_dim=args.embed_dim,
            apply_weights_to_negatives=False,
            normalize="pos",
            severity_map=SEVERITY_ID,           # class_id -> severity
            severity_normalize_mean1=True,
            learn_embeddings=True,
            warmup_freeze_epochs=args.warmup_freeze_epochs,
        )

        cafl = CAFLoss(
            num_classes=num_classes,
            gamma=cfg.gamma,
            apply_weights_to_negatives=cfg.apply_weights_to_negatives,
            normalize=cfg.normalize,
        )

        # Only create similarity module for modes that use it
        if args.loss_mode in {"full_cafl", "similarity_only"}:
            similarity = ClassEmbeddingSimilarity(
                num_classes, embed_dim=cfg.embed_dim, learn=cfg.learn_embeddings
            )
            print(f"Setting up model with CAFL head (mode={args.loss_mode}, with similarity)...")
            model = swap_in_cafl_head(model, cafl, similarity_module=similarity)
        else:
            similarity = None
            print(f"Setting up model with CAFL head (mode={args.loss_mode}, no similarity)...")
            model = swap_in_cafl_head(model, cafl, similarity_module=None)
    else:
        print("Running vanilla RetinaNet (CAFL disabled).")



    model.to(device)

    wandb_run = init_wandb_run(
        args=args,
        train_ds=train_ds,
        val_ds=val_ds,
        num_classes=num_classes,
        model=model,
    )
    configure_wandb_metrics(wandb_run)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if cafl_enabled:
        # ---- Class weights according to loss-mode ----
        counts_vec = train_ds.class_counts_tensor().to(device)  # positives per class

        # Default: all ones (no weighting)
        w_eff = torch.ones(num_classes, device=device)
        sev_vec = torch.ones(num_classes, device=device)

        # Effective-number weights
        if args.loss_mode in {"en_only", "en_severity", "full_cafl"}:
            w_eff = effective_number_weights(counts_vec, beta=cfg.beta, normalize_mean1=True)

        # Severity weights
        if args.loss_mode in {"severity_only", "en_severity", "full_cafl"}:
            sev_vec = SeverityMap(
                num_classes, mapping=SEVERITY_ID, normalize_mean1=cfg.severity_normalize_mean1
            ).vector().to(device)

        cafl.set_effective_number_weights(w_eff)
        cafl.set_severity_weights(sev_vec)

        # Similarity warmup (only if we actually created similarity)
        if similarity is not None and cfg.warmup_freeze_epochs > 0:
            similarity.freeze()



    accum = max(1, args.accum_steps)
    warmup_freeze_epochs = cfg.warmup_freeze_epochs if cafl_enabled else 0

    print(f"Training set size: {len(train_ds)} images, classes={num_classes}")
    print(f"FP32 training, batch_size={args.batch_size}, accum_steps={accum}, "
          f"min/max={args.min_size}/{args.max_size}")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        if cafl_enabled and similarity is not None and epoch == warmup_freeze_epochs:
            similarity.unfreeze()

        running = 0.0
        steps = 0
        epoch_cls = 0.0
        epoch_bbox = 0.0
        optimizer.zero_grad(set_to_none=True)

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1} [train]",
            leave=False,
        )
        for step, (images, targets) in enumerate(train_bar, start=1):
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            # forward in full precision to keep CAFL happy
            loss_dict = model(images, targets)  # {"classification": ..., "bbox_regression": ...}
            loss = loss_dict["classification"] + loss_dict["bbox_regression"]
            loss = loss / accum  # for gradient accumulation

            loss.backward()

            if step % accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            cls_loss = float(loss_dict["classification"].detach().cpu())
            bbox_loss = float(loss_dict["bbox_regression"].detach().cpu())
            loss_value = float(loss.detach().cpu()) * accum

            running += loss_value
            steps += 1
            epoch_cls += cls_loss
            epoch_bbox += bbox_loss
            train_bar.set_postfix({
                "loss": f"{loss_value:.3f}",
                "cls": f"{cls_loss:.3f}",
                "bbox": f"{bbox_loss:.3f}",
            })

            del images, targets, loss_dict, loss  # free asap

        epoch_loss = running / max(1, steps)
        epoch_cls_loss = epoch_cls / max(1, steps)
        epoch_bbox_loss = epoch_bbox / max(1, steps)
        print(f"train loss: {epoch_loss:.4f}")

        # (optional) simple eval stub
        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate_detector(model, val_loader, device)
            print(
                f"val: mel_recall={val_metrics['mel_recall']:.3f}, "
                f"mel_prec={val_metrics['mel_precision']:.3f}, "
                f"overall_recall={val_metrics['overall_recall']:.3f}, "
                f"overall_prec={val_metrics['overall_precision']:.3f}, "
                f"severity_weighted_fn={val_metrics['severity_weighted_fn']:.1f}"
            )


        if wandb_run:
            lr = optimizer.param_groups[0]["lr"]
            metrics = {
                "epoch": epoch + 1,
                "train/loss": epoch_loss,
                "train/cls_loss": epoch_cls_loss,
                "train/bbox_loss": epoch_bbox_loss,
                "optimizer/lr": lr,
            }
            if val_metrics is not None:
                metrics.update({
                    "val/mel_recall": val_metrics["mel_recall"],
                    "val/mel_precision": val_metrics["mel_precision"],
                    "val/overall_recall": val_metrics["overall_recall"],
                    "val/overall_precision": val_metrics["overall_precision"],
                    "val/severity_weighted_fn": val_metrics["severity_weighted_fn"],
                    "val/total_gt": val_metrics["total_gt"],
                })
            wandb_run.log(metrics, step=epoch + 1)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Training done.")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
