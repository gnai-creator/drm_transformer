"""Figure 1: Painel combinado 2x3 com propriedades geometricas do DRM Transformer."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import (
    set_seed, create_model, tokenize_texts, get_U_and_coords,
    aggregate_heads_mean, project_on_axes, compute_geodesic_dist,
    compute_euclidean_dist, silhouette, separation_ratio,
    get_all_texts_and_labels, load_results,
    CLASS_NAMES, FIGURES_DIR, logger,
)


def main():
    set_seed(42)
    model, config = create_model()
    device = next(model.parameters()).device

    texts, labels = get_all_texts_and_labels()
    input_ids = tokenize_texts(texts).to(device)

    n_layers = len(model.blocks)
    layer_indices = [0, n_layers // 2, n_layers - 1]

    # --- Extrair dados no ultimo layer ---
    U_all, coords_all, x_pre = get_U_and_coords(model, input_ids, layer_idx=-1)
    U_mean, coords_mean = aggregate_heads_mean(U_all, coords_all)

    coords_avg = coords_mean.mean(dim=1)  # [B, D]
    U_avg = U_mean.mean(dim=1)  # [B, D, r]
    x_pre_avg = x_pre.mean(dim=1).cpu().numpy()  # [B, d_model]
    r = U_avg.shape[-1]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(CLASS_NAMES)))
    class_colors = {name: colors[i] for i, name in enumerate(CLASS_NAMES)}

    # =====================================================
    # (0,0) PCA scatter (manifold coords) colored by class
    # =====================================================
    from sklearn.decomposition import PCA

    coords_np = coords_avg.cpu().numpy()
    pca = PCA(n_components=2)
    proj_pca = pca.fit_transform(coords_np)
    sil_val = silhouette(coords_np, labels)

    ax = axes[0, 0]
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = labels == cls_idx
        ax.scatter(
            proj_pca[mask, 0], proj_pca[mask, 1],
            c=[class_colors[cls_name]], label=cls_name,
            alpha=0.7, s=25, edgecolors="k", linewidths=0.3,
        )
    ax.set_title(f"PCA Manifold Coords\nsilhouette={sil_val:.4f}", fontsize=10)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=8)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=8)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # =====================================================
    # (0,1) Axis heatmap (axes vs classes)
    # =====================================================
    proj_axes = project_on_axes(coords_avg, U_avg).cpu().numpy()  # [B, r]

    heatmap = np.zeros((r, len(CLASS_NAMES)))
    for cls_idx in range(len(CLASS_NAMES)):
        mask = labels == cls_idx
        heatmap[:, cls_idx] = proj_axes[mask].mean(axis=0)

    ax = axes[0, 1]
    im = ax.imshow(heatmap, aspect="auto", cmap="RdBu_r")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(r))
    ax.set_yticklabels([f"Ax{i}" for i in range(r)], fontsize=8)
    ax.set_title("Axis Activation per Class", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)

    for i in range(r):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{heatmap[i, j]:.3f}",
                    ha="center", va="center", fontsize=6,
                    color="white" if abs(heatmap[i, j]) > abs(heatmap).max() * 0.5 else "black")

    # =====================================================
    # (0,2) Cosine similarity matrix between axes
    # =====================================================
    U_flat = U_mean.reshape(-1, U_mean.shape[-2], U_mean.shape[-1])  # [N, D, r]
    U_mean_axes = U_flat.mean(dim=0)  # [D, r]
    U_normed = torch.nn.functional.normalize(U_mean_axes, dim=0)
    cosine_mat = (U_normed.T @ U_normed).cpu().numpy()  # [r, r]

    ax = axes[0, 2]
    im2 = ax.imshow(cosine_mat, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(r))
    ax.set_xticklabels([f"Ax{i}" for i in range(r)], fontsize=8)
    ax.set_yticks(range(r))
    ax.set_yticklabels([f"Ax{i}" for i in range(r)], fontsize=8)
    ax.set_title("Cosine Similarity Between Axes", fontsize=10)
    plt.colorbar(im2, ax=ax, shrink=0.8)

    for i in range(r):
        for j in range(r):
            ax.text(j, i, f"{cosine_mat[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(cosine_mat[i, j]) > 0.5 else "black")

    # =====================================================
    # (1,0) Separation ratio bar: U_real vs U_zero vs U_random
    # =====================================================
    sep_real = separation_ratio(coords_np, labels)
    sil_real = sil_val

    # U_zero: projecao com U=0 (so coords)
    sil_zero = silhouette(coords_np, labels)  # coords nao mudam
    sep_zero = separation_ratio(coords_np, labels)

    # Para separacao, medir via projecao nos eixos
    proj_real_np = proj_axes
    proj_zero_np = project_on_axes(coords_avg, torch.zeros_like(U_avg)).cpu().numpy()
    proj_rand_np = project_on_axes(coords_avg, torch.randn_like(U_avg)).cpu().numpy()

    sil_proj_real = silhouette(proj_real_np, labels)
    sil_proj_zero = silhouette(proj_zero_np, labels)
    sil_proj_rand = silhouette(proj_rand_np, labels)

    sep_proj_real = separation_ratio(proj_real_np, labels)
    sep_proj_zero = separation_ratio(proj_zero_np, labels)
    sep_proj_rand = separation_ratio(proj_rand_np, labels)

    ax = axes[1, 0]
    x_pos = np.arange(2)
    width = 0.25

    ax.bar(x_pos - width, [sil_proj_real, sep_proj_real], width,
           label="U_real", color="#2ecc71", edgecolor="k")
    ax.bar(x_pos, [sil_proj_zero, sep_proj_zero], width,
           label="U_zero", color="#95a5a6", edgecolor="k")
    ax.bar(x_pos + width, [sil_proj_rand, sep_proj_rand], width,
           label="U_random", color="#e74c3c", edgecolor="k")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Silhouette", "Sep. Ratio"], fontsize=9)
    ax.set_title("Ablation: Separation Metrics", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # =====================================================
    # (1,1) Geometry-semantics correlation bar
    # =====================================================
    from scipy.stats import pearsonr
    from sklearn.metrics.pairwise import cosine_distances

    d_sem = cosine_distances(x_pre_avg)
    idx_tri = np.triu_indices_from(d_sem, k=1)
    d_sem_flat = d_sem[idx_tri]

    d_geom = compute_geodesic_dist(coords_avg, U_avg).cpu().numpy()
    d_euc = compute_euclidean_dist(coords_avg).cpu().numpy()
    U_rand = torch.randn_like(U_avg)
    d_rand = compute_geodesic_dist(coords_avg, U_rand).cpu().numpy()

    corr_geom, _ = pearsonr(d_geom[idx_tri], d_sem_flat)
    corr_euc, _ = pearsonr(d_euc[idx_tri], d_sem_flat)
    corr_rand, _ = pearsonr(d_rand[idx_tri], d_sem_flat)

    ax = axes[1, 1]
    bar_x = np.arange(3)
    bar_colors = ["#2ecc71", "#95a5a6", "#e74c3c"]
    bar_vals = [corr_geom, corr_euc, corr_rand]
    bars = ax.bar(bar_x, bar_vals, color=bar_colors, edgecolor="k")
    ax.set_xticks(bar_x)
    ax.set_xticklabels(["Geodesic", "Euclidean", "Random"], fontsize=9)
    ax.set_ylabel("Pearson r", fontsize=9)
    ax.set_title("Geometry-Semantics Correlation", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, bar_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # =====================================================
    # (1,2) Axis activation per layer
    # =====================================================
    ax = axes[1, 2]
    layer_data = {}

    for li in layer_indices:
        U_li, coords_li, _ = get_U_and_coords(model, input_ids, layer_idx=li)
        U_li_mean, coords_li_mean = aggregate_heads_mean(U_li, coords_li)
        proj_li = project_on_axes(
            coords_li_mean.mean(dim=1), U_li_mean.mean(dim=1),
        )  # [B, r]
        layer_data[li] = proj_li.abs().mean(dim=0).cpu().numpy()  # [r]

    axis_colors = plt.cm.Set2(np.linspace(0, 1, r))
    for ax_idx in range(r):
        vals = [layer_data[li][ax_idx] for li in layer_indices]
        ax.plot(range(len(layer_indices)), vals, "o-",
                color=axis_colors[ax_idx], label=f"Axis {ax_idx}",
                linewidth=1.5, markersize=5)

    ax.set_xticks(range(len(layer_indices)))
    ax.set_xticklabels([f"L{li}" for li in layer_indices], fontsize=9)
    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel("Mean |Activation|", fontsize=9)
    ax.set_title("Axis Activation Across Layers", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # =====================================================
    # Finalizacao
    # =====================================================
    plt.suptitle(
        "DRM Transformer Geometric Properties",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out_path = FIGURES_DIR / "figure1_combined.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Salvo: %s", out_path)


if __name__ == "__main__":
    main()
