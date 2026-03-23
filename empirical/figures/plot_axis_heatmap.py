"""Heatmap: ativacao media por eixo semantico vs classe."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    set_seed, create_model, tokenize_texts, get_U_and_coords,
    aggregate_heads_mean, project_on_axes, get_all_texts_and_labels,
    CLASS_NAMES, FIGURES_DIR, logger,
)


def main():
    set_seed(42)
    model, config = create_model()
    device = next(model.parameters()).device

    texts, labels = get_all_texts_and_labels()
    input_ids = tokenize_texts(texts, vocab_size=config.vocab_size).to(device)

    U_all, coords_all, _ = get_U_and_coords(model, input_ids, layer_idx=-1)
    U_mean, coords_mean = aggregate_heads_mean(U_all, coords_all)

    # Projecao media por amostra
    proj = project_on_axes(
        coords_mean.mean(dim=1), U_mean.mean(dim=1),
    ).cpu().numpy()  # [B, r]

    r = proj.shape[-1]

    # Media por classe
    heatmap = np.zeros((r, len(CLASS_NAMES)))
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = labels == cls_idx
        heatmap[:, cls_idx] = proj[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(heatmap, aspect="auto", cmap="RdBu_r")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(r))
    ax.set_yticklabels([f"Axis {i}" for i in range(r)])
    ax.set_xlabel("Classe")
    ax.set_ylabel("Eixo Semantico")
    ax.set_title("Ativacao Media por Eixo vs Classe")

    # Anotar valores
    for i in range(r):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{heatmap[i, j]:.3f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(heatmap[i, j]) > heatmap.max() * 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Projecao media")
    plt.tight_layout()

    out_path = FIGURES_DIR / "axis_heatmap.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Salvo: %s", out_path)


if __name__ == "__main__":
    main()
