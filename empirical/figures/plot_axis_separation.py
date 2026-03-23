"""Bar chart: separacao por eixo com ablation (U_real vs U_zero vs U_random)."""

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
    aggregate_heads_mean, project_on_axes, get_all_texts_and_labels,
    CLASS_NAMES, CLASS_LABELS, FIGURES_DIR, logger,
)


def _axis_max_separation(proj, labels):
    """Calcula max separacao entre classes por eixo."""
    r = proj.shape[-1]
    seps = []
    for ax in range(r):
        vals = proj[:, ax]
        max_sep = 0.0
        for i in range(len(CLASS_NAMES)):
            for j in range(i + 1, len(CLASS_NAMES)):
                m1 = vals[labels == i].mean()
                m2 = vals[labels == j].mean()
                max_sep = max(max_sep, abs(m1 - m2))
        seps.append(max_sep)
    return seps


def main():
    set_seed(42)
    model, config = create_model()
    device = next(model.parameters()).device

    texts, labels = get_all_texts_and_labels()
    input_ids = tokenize_texts(texts, vocab_size=config.vocab_size).to(device)

    U_all, coords_all, _ = get_U_and_coords(model, input_ids, layer_idx=-1)
    U_mean, coords_mean = aggregate_heads_mean(U_all, coords_all)

    coords_avg = coords_mean.mean(dim=1)
    U_avg = U_mean.mean(dim=1)
    r = U_avg.shape[-1]

    # U_real
    proj_real = project_on_axes(coords_avg, U_avg).cpu().numpy()
    sep_real = _axis_max_separation(proj_real, labels)

    # U_zero
    proj_zero = project_on_axes(
        coords_avg, torch.zeros_like(U_avg),
    ).cpu().numpy()
    sep_zero = _axis_max_separation(proj_zero, labels)

    # U_random (media de 5 runs)
    sep_rand_runs = []
    for _ in range(5):
        proj_rand = project_on_axes(
            coords_avg, torch.randn_like(U_avg),
        ).cpu().numpy()
        sep_rand_runs.append(_axis_max_separation(proj_rand, labels))
    sep_rand = np.mean(sep_rand_runs, axis=0)
    sep_rand_std = np.std(sep_rand_runs, axis=0)

    # Plot
    x = np.arange(r)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, sep_real, width, label="U_real", color="#2ecc71", edgecolor="k")
    ax.bar(x, sep_zero, width, label="U_zero (G=I)", color="#95a5a6", edgecolor="k")
    ax.bar(
        x + width, sep_rand, width, label="U_random",
        color="#e74c3c", edgecolor="k", yerr=sep_rand_std, capsize=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Axis {i}" for i in range(r)])
    ax.set_xlabel("Eixo Semantico")
    ax.set_ylabel("Max Separacao Inter-Classe")
    ax.set_title("Separacao por Eixo: Ablation (U_real vs U_zero vs U_random)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = FIGURES_DIR / "axis_separation.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Salvo: %s", out_path)


if __name__ == "__main__":
    main()
