"""Visualizacao PCA: pre-manifold vs manifold (side-by-side)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from utils import (
    set_seed, create_model, tokenize_texts, get_U_and_coords,
    aggregate_heads_mean, silhouette, get_all_texts_and_labels,
    CLASS_NAMES, FIGURES_DIR, logger,
)


def main():
    set_seed(42)
    model, config = create_model()
    device = next(model.parameters()).device

    texts, labels = get_all_texts_and_labels()
    input_ids = tokenize_texts(texts).to(device)

    U_all, coords_all, x_pre = get_U_and_coords(model, input_ids, layer_idx=-1)
    _, coords_mean = aggregate_heads_mean(U_all, coords_all)

    # Media sobre tokens
    coords_avg = coords_mean.mean(dim=1).cpu().numpy()  # [B, D]
    x_pre_avg = x_pre.mean(dim=1).cpu().numpy()  # [B, d_model]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(CLASS_NAMES)))
    class_colors = {name: colors[i] for i, name in enumerate(CLASS_NAMES)}

    for ax_idx, (data, title_prefix) in enumerate([
        (x_pre_avg, "Pre-Manifold"),
        (coords_avg, "Manifold Coords"),
    ]):
        pca = PCA(n_components=2)
        proj = pca.fit_transform(data)
        sil = silhouette(data, labels)

        ax = axes[ax_idx]
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            mask = labels == cls_idx
            ax.scatter(
                proj[mask, 0], proj[mask, 1],
                c=[class_colors[cls_name]], label=cls_name,
                alpha=0.7, s=30, edgecolors="k", linewidths=0.3,
            )
        ax.set_title(f"{title_prefix}\nsilhouette={sil:.4f}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        logger.info("%s silhouette: %.4f", title_prefix, sil)

    plt.suptitle("PCA: Pre-Manifold vs Manifold", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = FIGURES_DIR / "pca.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Salvo: %s", out_path)


if __name__ == "__main__":
    main()
