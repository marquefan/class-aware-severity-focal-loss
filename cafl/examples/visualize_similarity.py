"""
Visualize the learned CAFL class-similarity matrix.

Loads the similarity_matrix.pt saved after training and produces:
  - A heatmap of W^(s) = 1 - cos(E, E)  (low = similar, high = dissimilar)
  - A t-SNE / PCA plot of the raw class embeddings E (optional)

Usage:
    python -m cafl.examples.visualize_similarity \\
        --matrix checkpoints/similarity_matrix.pt \\
        --out    similarity_heatmap.png

    # To also plot embeddings (needs the full checkpoint):
    python -m cafl.examples.visualize_similarity \\
        --matrix checkpoints/similarity_matrix.pt \\
        --embeddings checkpoints/epoch_020.pt \\
        --out similarity_heatmap.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import torch


CLASS_NAMES: List[str] = [
    "akiec", "bcc", "bkl", "df", "nv", "mel", "vasc"
]

# Rough clinical groupings for annotation
CLINICAL_GROUPS = {
    "akiec": "malignant/pre-malignant",
    "bcc":   "malignant/pre-malignant",
    "mel":   "malignant/pre-malignant",
    "bkl":   "benign",
    "df":    "benign",
    "nv":    "benign",
    "vasc":  "vascular",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize CAFL similarity matrix.")
    p.add_argument("--matrix", type=Path, required=True,
                   help="Path to similarity_matrix.pt (saved by ham10000_train.py).")
    p.add_argument("--embeddings", type=Path, default=None,
                   help="Optional: path to a checkpoint .pt to extract E for t-SNE/PCA.")
    p.add_argument("--class-names", nargs="+", default=CLASS_NAMES,
                   help="Class names in label order (0-indexed).")
    p.add_argument("--out", type=Path, default=Path("similarity_heatmap.png"),
                   help="Output PNG path.")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def _load_embeddings_from_checkpoint(ckpt_path: Path) -> Optional[torch.Tensor]:
    """Try to extract E (class embeddings) from a training checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)  # top-level state dict or model key
    for key in state:
        if "similarity_module.E" in key or key.endswith(".E"):
            return state[key]
    return None


def plot_heatmap(W: torch.Tensor, class_names: List[str], out: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    K = len(class_names)
    data = W.numpy()

    fig, ax = plt.subplots(figsize=(K + 1, K))

    # Use a diverging colormap: blue = similar (low W), red = dissimilar (high W)
    im = ax.imshow(data, cmap="RdYlBu_r", vmin=0, vmax=2)
    plt.colorbar(im, ax=ax, label="Dissimilarity (1 − cos)")

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_title("CAFL Learned Class Similarity Matrix\n(lower = more similar)")

    # Annotate cells with values
    for i in range(K):
        for j in range(K):
            v = data[i, j]
            color = "white" if v > 1.2 or v < 0.3 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"Heatmap saved: {out}")
    plt.close(fig)


def plot_embeddings(E: torch.Tensor, class_names: List[str], out: Path, dpi: int) -> None:
    """2-D PCA of class embeddings (falls back from t-SNE if sklearn unavailable)."""
    import matplotlib.pyplot as plt
    import numpy as np

    E_np = E.detach().numpy()
    K = len(class_names)

    # Normalize rows (unit sphere)
    E_np = E_np / (np.linalg.norm(E_np, axis=1, keepdims=True) + 1e-8)

    try:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2).fit_transform(E_np)
        method = "PCA"
    except ImportError:
        # Manual PCA via SVD
        U, S, Vt = np.linalg.svd(E_np - E_np.mean(0), full_matrices=False)
        coords = U[:, :2] * S[:2]
        method = "PCA (manual SVD)"

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, K))
    for i, (name, c) in enumerate(zip(class_names, colors)):
        ax.scatter(coords[i, 0], coords[i, 1], color=c, s=120, zorder=3)
        ax.annotate(
            name,
            (coords[i, 0], coords[i, 1]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=9,
        )
    ax.set_title(f"Class Embeddings ({method})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.3)

    embed_out = out.parent / (out.stem + "_embeddings" + out.suffix)
    fig.tight_layout()
    fig.savefig(embed_out, dpi=dpi, bbox_inches="tight")
    print(f"Embedding plot saved: {embed_out}")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not args.matrix.exists():
        raise FileNotFoundError(f"Matrix file not found: {args.matrix}")

    W = torch.load(args.matrix, map_location="cpu")
    K = W.shape[0]
    class_names = args.class_names[:K]

    print(f"Loaded similarity matrix: shape={list(W.shape)}")
    print(f"  min={W.min():.4f}, max={W.max():.4f}, diagonal={W.diagonal().mean():.4f}")

    # Print text summary
    id2name = {i: n for i, n in enumerate(class_names)}
    print("\nMost similar pairs (lowest W[i,j], i≠j):")
    off_diag = [(W[i, j].item(), i, j)
                for i in range(K) for j in range(K) if i != j]
    for v, i, j in sorted(off_diag)[:5]:
        print(f"  {id2name[i]:6s} ↔ {id2name[j]:6s}  W={v:.4f}")

    print("\nMost dissimilar pairs (highest W[i,j], i≠j):")
    for v, i, j in sorted(off_diag, reverse=True)[:5]:
        print(f"  {id2name[i]:6s} ↔ {id2name[j]:6s}  W={v:.4f}")

    plot_heatmap(W, class_names, args.out, args.dpi)

    # Plot embeddings if a checkpoint was provided
    if args.embeddings is not None:
        E = _load_embeddings_from_checkpoint(args.embeddings)
        if E is not None:
            plot_embeddings(E, class_names, args.out, args.dpi)
        else:
            print("Could not find embedding tensor E in checkpoint — skipping embedding plot.")


if __name__ == "__main__":
    main()
