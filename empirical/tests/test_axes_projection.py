"""H1: Verifica ativacao dos eixos semanticos por layer."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
from utils import (
    set_seed, create_model, tokenize_texts, get_U_and_coords,
    aggregate_heads_mean, project_on_axes, get_all_texts_and_labels,
    save_results, logger,
)


def run(seed: int = 42):
    """Executa teste de projecao nos eixos por layer."""
    set_seed(seed)
    model, config = create_model()
    device = next(model.parameters()).device

    texts, _ = get_all_texts_and_labels()
    input_ids = tokenize_texts(texts[:20], vocab_size=config.vocab_size).to(device)

    n_layers = len(model.blocks)
    layer_indices = [0, n_layers // 2, n_layers - 1]

    results = {}

    for li in layer_indices:
        U_all, coords_all, _ = get_U_and_coords(model, input_ids, layer_idx=li)
        U_mean, coords_mean = aggregate_heads_mean(U_all, coords_all)

        proj = project_on_axes(coords_mean, U_mean)  # [B, T, r]
        proj_flat = proj.reshape(-1, proj.shape[-1])  # [N, r]

        layer_key = f"layer_{li}"
        results[layer_key] = {}

        logger.info("=== Layer %d ===", li)
        for ax in range(proj_flat.shape[-1]):
            vals = proj_flat[:, ax]
            mean_val = vals.mean().item()
            std_val = vals.std().item()
            results[layer_key][f"axis_{ax}"] = {
                "mean": mean_val, "std": std_val,
            }
            logger.info(
                "  Axis %d -> mean: %.4f, std: %.4f",
                ax, mean_val, std_val,
            )

    return results


def main():
    all_results = {}
    for seed in [42, 123, 7]:
        all_results[f"seed_{seed}"] = run(seed)

    save_results({"axes_projection": all_results})
    logger.info("Teste de projecao nos eixos concluido.")


if __name__ == "__main__":
    main()
