"""
Ablation runner for CAFL on HAM10000.

Runs all 6 loss modes sequentially, then prints a comparison table and saves a CSV.
Each mode gets its own checkpoint sub-directory under --output-dir. Final val metrics
are written to <output-dir>/<mode>/final_metrics.json by the training script.

Usage (after running ham10000_split.py to create train/val CSVs):

    python -m cafl.examples.ham10000_ablation \\
        --images-dir HAM10000_images \\
        --train-csv  splits/ham10000_bboxes_val0_test1/train.csv \\
        --val-csv    splits/ham10000_bboxes_val0_test1/val.csv \\
        --epochs 20 \\
        --output-dir ablation_results

To run only a subset of modes:
    python -m cafl.examples.ham10000_ablation ... --modes baseline full_cafl

To skip modes that already finished:
    python -m cafl.examples.ham10000_ablation ... --skip-existing
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

LOSS_MODES = [
    "baseline",        # A0: vanilla RetinaNet focal loss
    "en_only",         # A1: effective-number weighting only
    "severity_only",   # A2: severity weighting only
    "en_severity",     # A3: EN + severity, no similarity
    "similarity_only", # A4: similarity only
    "full_cafl",       # A5: all components (EN + severity + similarity)
]

REPORT_METRICS = [
    ("map50",               "mAP@0.5"),
    ("map50_95",            "mAP@0.5:0.95"),
    ("mel_recall",          "Mel Recall"),
    ("mel_precision",       "Mel Prec"),
    ("overall_recall",      "Overall Recall"),
    ("overall_precision",   "Overall Prec"),
    ("severity_weighted_fn","Sev-FN"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all CAFL ablation modes and compare results."
    )
    # Training args forwarded verbatim to ham10000_train.py
    p.add_argument("--images-dir",  type=Path, required=True)
    p.add_argument("--train-csv",   type=Path, required=True)
    p.add_argument("--val-csv",     type=Path, required=True)
    p.add_argument("--epochs",      type=int,  default=20)
    p.add_argument("--batch-size",  type=int,  default=2)
    p.add_argument("--accum-steps", type=int,  default=4)
    p.add_argument("--lr",          type=float,default=2e-4)
    p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--num-workers", type=int,  default=2)
    p.add_argument("--min-size",    type=int,  default=450)
    p.add_argument("--max-size",    type=int,  default=600)
    p.add_argument("--seed",        type=int,  default=17)
    p.add_argument("--beta",        type=float,default=0.999)
    p.add_argument("--gamma",       type=float,default=2.0)
    p.add_argument("--embed-dim",   type=int,  default=32)
    p.add_argument("--warmup-freeze-epochs", type=int, default=1)
    p.add_argument("--lr-scheduler",type=str,  default="cosine",
                   choices=["none", "cosine", "step"])

    # Optional W&B — each mode gets its own named run
    p.add_argument("--wandb", action="store_true",
                   help="Enable W&B logging for each mode.")
    p.add_argument("--wandb-project", type=str, default="cafl_ablation")
    p.add_argument("--wandb-entity",  type=str, default=None)

    # Ablation-specific
    p.add_argument("--output-dir", type=Path, default=Path("ablation_results"),
                   help="Root directory for per-mode checkpoints and the summary.")
    p.add_argument("--modes", nargs="+", default=LOSS_MODES, choices=LOSS_MODES,
                   help="Subset of loss modes to run (default: all 6).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip a mode if final_metrics.json already exists.")
    return p.parse_args()


def _build_cmd(args: argparse.Namespace, mode: str, checkpoint_dir: Path) -> List[str]:
    cmd = [
        sys.executable, "-m", "cafl.examples.ham10000_train",
        "--images-dir",            str(args.images_dir),
        "--train-csv",             str(args.train_csv),
        "--val-csv",               str(args.val_csv),
        "--epochs",                str(args.epochs),
        "--batch-size",            str(args.batch_size),
        "--accum-steps",           str(args.accum_steps),
        "--lr",                    str(args.lr),
        "--weight-decay",          str(args.weight_decay),
        "--num-workers",           str(args.num_workers),
        "--min-size",              str(args.min_size),
        "--max-size",              str(args.max_size),
        "--seed",                  str(args.seed),
        "--beta",                  str(args.beta),
        "--gamma",                 str(args.gamma),
        "--embed-dim",             str(args.embed_dim),
        "--warmup-freeze-epochs",  str(args.warmup_freeze_epochs),
        "--lr-scheduler",          args.lr_scheduler,
        "--loss-mode",             mode,
        "--checkpoint-dir",        str(checkpoint_dir),
    ]
    if args.wandb:
        cmd += ["--wandb",
                "--wandb-project", args.wandb_project,
                "--wandb-run-name", f"ablation_{mode}"]
        if args.wandb_entity:
            cmd += ["--wandb-entity", args.wandb_entity]
    return cmd


def _print_table(results: List[Dict[str, Any]]) -> None:
    col_w = 16
    header_row = ["mode"] + [label for _, label in REPORT_METRICS] + ["time(min)", "status"]
    header = "".join(f"{c:<{col_w}}" for c in header_row)
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        cells = [f"{r['mode']:<{col_w}}"]
        for key, _ in REPORT_METRICS:
            v = r.get(key)
            cells.append(f"{v:<{col_w}.4f}" if v is not None else f"{'—':<{col_w}}")
        mins = r.get("wall_seconds", 0) / 60
        cells.append(f"{mins:<{col_w}.1f}")
        cells.append(f"{r.get('status','?'):<{col_w}}")
        print("".join(cells))
    print(sep + "\n")


def _save_csv(results: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["mode"]
        + [k for k, _ in REPORT_METRICS]
        + ["wall_seconds", "status"]
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Summary CSV saved: {path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"CAFL Ablation Study")
    print(f"  Modes  : {args.modes}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Output : {args.output_dir.resolve()}")
    print()

    results: List[Dict[str, Any]] = []

    for mode in args.modes:
        checkpoint_dir = args.output_dir / mode
        metrics_json = checkpoint_dir / "final_metrics.json"

        # Skip if already done
        if args.skip_existing and metrics_json.exists():
            with metrics_json.open() as f:
                saved = json.load(f)
            row = {**saved, "mode": mode, "status": "skipped (existing)"}
            print(f"[{mode}] Skipped — loaded from {metrics_json}")
            results.append(row)
            continue

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cmd = _build_cmd(args, mode, checkpoint_dir)

        print(f"[{mode}] Starting... ", flush=True)
        t0 = time.time()
        ret = subprocess.run(cmd)
        elapsed = time.time() - t0

        row: Dict[str, Any] = {
            "mode": mode,
            "wall_seconds": round(elapsed, 1),
        }

        if ret.returncode != 0:
            print(f"[{mode}] FAILED (exit {ret.returncode}) after {elapsed/60:.1f} min")
            row["status"] = "FAILED"
            for key, _ in REPORT_METRICS:
                row[key] = None
        else:
            row["status"] = "ok"
            if metrics_json.exists():
                with metrics_json.open() as f:
                    saved = json.load(f)
                for key, _ in REPORT_METRICS:
                    row[key] = saved.get(key)
                print(
                    f"[{mode}] Done in {elapsed/60:.1f} min — "
                    f"mAP@0.5={row.get('map50', '?'):.4f}, "
                    f"mel_recall={row.get('mel_recall', '?'):.4f}"
                )
            else:
                print(f"[{mode}] Done but no val metrics found (was --val-csv provided?)")
                for key, _ in REPORT_METRICS:
                    row[key] = None

        results.append(row)

    _print_table(results)
    _save_csv(results, args.output_dir / "ablation_summary.csv")


if __name__ == "__main__":
    main()
