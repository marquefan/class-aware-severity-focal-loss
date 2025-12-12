"""Stratified k-fold splitting utility for HAM10000 bounding boxes.

Given a CSV that already contains bounding boxes and class labels (e.g.
`ham10000_bboxes_with_labels.csv`), this script partitions the data at the
image level so that every image's boxes land in the same split. A stratified
k-fold assignment keeps class ratios balanced, and you can pick which folds
serve as validation/test (default: 10-fold with 80/10/10 train/val/test).
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stratified train/val/test splits for HAM10000 annotations."
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        required=True,
        help="CSV that includes bounding boxes and a 'dx' label column.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("splits"),
        help="Base directory where split CSVs will be written.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=10,
        help="Number of folds for the stratified assignment (default 10 for 80/10/10).",
    )
    parser.add_argument(
        "--val-fold",
        type=int,
        default=0,
        help="Which fold index to use for the validation split (0-indexed).",
    )
    parser.add_argument(
        "--test-fold",
        type=int,
        default=1,
        help="Which fold index to use for the test split (must differ from val fold).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed controlling the stratified shuffle order.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag appended to the output directory name (e.g., 'fold0').",
    )
    return parser.parse_args()


def _group_image_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "image_id" not in df.columns or "dx" not in df.columns:
        missing = {"image_id", "dx"} - set(df.columns)
        raise ValueError(f"Source CSV is missing required columns: {missing}")
    grouped = df[["image_id", "dx"]].drop_duplicates("image_id").reset_index(drop=True)
    return grouped


def _assign_folds(image_labels: pd.DataFrame, num_folds: int, seed: int) -> Dict[str, int]:
    if num_folds < 3:
        raise ValueError("num_folds must be >= 3 to produce train/val/test splits.")
    rng = random.Random(seed)
    fold_by_image: Dict[str, int] = {}
    for label, group in image_labels.groupby("dx"):
        ids = group["image_id"].tolist()
        rng.shuffle(ids)
        for idx, image_id in enumerate(ids):
            fold_by_image[image_id] = idx % num_folds
    return fold_by_image


def _split_indices(
    fold_assignments: Dict[str, int],
    val_fold: int,
    test_fold: int,
) -> Tuple[set[str], set[str], set[str]]:
    if val_fold == test_fold:
        raise ValueError("Validation and test folds must be different.")
    train_ids, val_ids, test_ids = set(), set(), set()
    for image_id, fold in fold_assignments.items():
        if fold == val_fold:
            val_ids.add(image_id)
        elif fold == test_fold:
            test_ids.add(image_id)
        else:
            train_ids.add(image_id)
    return train_ids, val_ids, test_ids


def _ensure_xyxy_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = {"x", "y", "w", "h"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Source CSV missing bounding-box columns: {missing}")

    df = df.copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df["x_min"] = df["x"]
    df["y_min"] = df["y"]
    df["x_max"] = df["x"] + df["w"]
    df["y_max"] = df["y"] + df["h"]
    return df


def _write_split(df: pd.DataFrame, image_ids: Iterable[str], path: Path) -> None:
    subset = df[df["image_id"].isin(image_ids)]
    path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(path, index=False)
    print(f"Saved {len(subset)} rows to {path}")


def main() -> None:
    args = parse_args()
    if not args.source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {args.source_csv}")

    boxes = pd.read_csv(args.source_csv)
    boxes = _ensure_xyxy_columns(boxes)
    image_labels = _group_image_labels(boxes)
    fold_assignments = _assign_folds(image_labels, args.num_folds, args.seed)

    train_ids, val_ids, test_ids = _split_indices(
        fold_assignments, args.val_fold, args.test_fold
    )

    stem = args.source_csv.stem
    tag = args.tag or f"val{args.val_fold}_test{args.test_fold}"
    split_dir = args.output_dir / f"{stem}_{tag}"

    _write_split(boxes, train_ids, split_dir / "train.csv")
    _write_split(boxes, val_ids, split_dir / "val.csv")
    _write_split(boxes, test_ids, split_dir / "test.csv")

    total_images = len(image_labels)
    print(
        f"Summary: train={len(train_ids)} ({len(train_ids)/total_images:.1%}), "
        f"val={len(val_ids)} ({len(val_ids)/total_images:.1%}), "
        f"test={len(test_ids)} ({len(test_ids)/total_images:.1%})"
    )
    print(f"Splits ready under {split_dir.resolve()}")


if __name__ == "__main__":
    main()
