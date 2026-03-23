"""H2: Verifica se a geometria separa categorias semanticas.

Compara 3 condicoes: U_real, U_zero, U_random (ablation).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
from utils import (
    set_seed, create_model, tokenize_texts, get_U_and_coords,
    aggregate_heads_mean, find_best_head, project_on_axes,
    separation_ratio, silhouette, get_all_texts_and_labels,
    CLASS_NAMES, save_results, logger,
)


def _compute_metrics(coords_np, proj_np, labels, condition_name):
    """Computa metricas de separacao para uma condicao."""
    # Silhouette nas coords
    sil_coords = silhouette(coords_np, labels)
    sep_coords = separation_ratio(coords_np, labels)

    # Por eixo: separacao entre classes
    n_axes = proj_np.shape[-1]
    per_axis = []
    for ax in range(n_axes):
        vals = proj_np[:, ax]
        class_means = {}
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            mask = labels == cls_idx
            if mask.sum() > 0:
                class_means[cls_name] = vals[mask].mean()

        max_sep = 0.0
        best_pair = ""
        names = list(class_means.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                sep = abs(class_means[n1] - class_means[n2])
                if sep > max_sep:
                    max_sep = sep
                    best_pair = f"{n1}-{n2}"

        per_axis.append({"axis": ax, "separation": float(max_sep), "pair": best_pair})

    per_axis.sort(key=lambda x: x["separation"], reverse=True)

    logger.info("  [%s] silhouette=%.4f, separation_ratio=%.4f",
                condition_name, sil_coords, sep_coords)
    for entry in per_axis:
        logger.info("    Axis %d: sep=%.4f (%s)",
                     entry["axis"], entry["separation"], entry["pair"])

    return {
        "silhouette": sil_coords,
        "separation_ratio": sep_coords,
        "per_axis": per_axis,
    }


def run(seed: int = 42):
    """Testa separacao semantica com ablation."""
    set_seed(seed)
    model, config = create_model()
    device = next(model.parameters()).device

    texts, labels = get_all_texts_and_labels()
    input_ids = tokenize_texts(texts).to(device)

    # Extrair no ultimo layer
    U_all, coords_all, _ = get_U_and_coords(model, input_ids, layer_idx=-1)

    # --- Mean heads ---
    U_mean, coords_mean = aggregate_heads_mean(U_all, coords_all)

    # Media por amostra (sobre tokens)
    coords_avg = coords_mean.mean(dim=1).cpu().numpy()  # [B, D]
    U_avg = U_mean.mean(dim=1)  # [B, D, r]

    proj_real = project_on_axes(
        coords_mean.mean(dim=1), U_avg,
    ).cpu().numpy()  # [B, r]

    results = {"mean_heads": {}, "best_head": {}}

    logger.info("=== Mean Heads ===")

    # U_real
    results["mean_heads"]["U_real"] = _compute_metrics(
        coords_avg, proj_real, labels, "U_real",
    )

    # U_zero
    U_zero = torch.zeros_like(U_avg)
    proj_zero = project_on_axes(
        coords_mean.mean(dim=1), U_zero,
    ).cpu().numpy()
    results["mean_heads"]["U_zero"] = _compute_metrics(
        coords_avg, proj_zero, labels, "U_zero",
    )

    # U_random
    U_rand = torch.randn_like(U_avg)
    proj_rand = project_on_axes(
        coords_mean.mean(dim=1), U_rand,
    ).cpu().numpy()
    results["mean_heads"]["U_random"] = _compute_metrics(
        coords_avg, proj_rand, labels, "U_random",
    )

    # --- Best head ---
    best_h = find_best_head(
        coords_all.mean(dim=2),  # [B, H, D]
        labels,
    )
    logger.info("\n=== Best Head: %d ===", best_h)

    coords_best_t = coords_all[:, best_h].mean(dim=1)
    coords_best = coords_best_t.cpu().numpy()
    U_best = U_all[:, best_h].mean(dim=1)
    proj_best = project_on_axes(coords_best_t, U_best).cpu().numpy()

    results["best_head"]["head_idx"] = best_h
    results["best_head"]["U_real"] = _compute_metrics(
        coords_best, proj_best, labels, "U_real",
    )

    return results


def main():
    all_seeds = {}
    for seed in [42, 123, 7]:
        all_seeds[f"seed_{seed}"] = run(seed)

    # Resumo: media sobre seeds
    for condition in ["U_real", "U_zero", "U_random"]:
        sils = [
            all_seeds[f"seed_{s}"]["mean_heads"][condition]["silhouette"]
            for s in [42, 123, 7]
        ]
        logger.info(
            "\n[RESUMO] %s silhouette: %.4f +/- %.4f",
            condition, np.mean(sils), np.std(sils),
        )

    save_results({"semantic_separation": all_seeds})
    logger.info("Teste de separacao semantica concluido.")


if __name__ == "__main__":
    main()
