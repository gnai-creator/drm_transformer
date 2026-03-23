"""H1: Verifica ortogonalidade e colapso dos eixos semanticos."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
from utils import (
    set_seed, create_model, tokenize_texts, get_U_and_coords,
    aggregate_heads_mean, get_all_texts_and_labels,
    save_results, logger,
)


def run(seed: int = 42):
    """Analisa estatisticas dos eixos U por layer."""
    set_seed(seed)
    model, config = create_model()
    device = next(model.parameters()).device

    texts, _ = get_all_texts_and_labels()
    input_ids = tokenize_texts(texts[:20]).to(device)

    n_layers = len(model.blocks)
    layer_indices = [0, n_layers // 2, n_layers - 1]

    results = {}

    for li in layer_indices:
        U_all, coords_all, _ = get_U_and_coords(model, input_ids, layer_idx=li)
        U_mean, _ = aggregate_heads_mean(U_all, coords_all)

        # U_mean: [B, T, D, r]
        U_flat = U_mean.reshape(-1, U_mean.shape[-2], U_mean.shape[-1])  # [N, D, r]
        r = U_flat.shape[-1]

        # Norma por eixo
        norms = U_flat.pow(2).sum(dim=-2).sqrt().mean(dim=0)  # [r]

        # Variancia entre tokens por eixo
        variance = U_flat.var(dim=0).mean(dim=0)  # [r]

        # Matriz de similaridade cosseno entre eixos
        # Eixo medio: [D, r]
        U_avg = U_flat.mean(dim=0)  # [D, r]
        U_normed = torch.nn.functional.normalize(U_avg, dim=0)  # [D, r]
        cosine_matrix = (U_normed.T @ U_normed).cpu().numpy()  # [r, r]

        layer_key = f"layer_{li}"
        results[layer_key] = {
            "norms": norms.cpu().tolist(),
            "variance": variance.cpu().tolist(),
            "cosine_matrix": cosine_matrix.tolist(),
        }

        logger.info("=== Layer %d ===", li)
        logger.info("  Normas por eixo: %s",
                     [f"{n:.4f}" for n in norms.tolist()])
        logger.info("  Variancia por eixo: %s",
                     [f"{v:.6f}" for v in variance.tolist()])
        logger.info("  Cosine similarity matrix:")
        for i in range(r):
            row = " ".join(f"{cosine_matrix[i, j]:+.3f}" for j in range(r))
            logger.info("    [%s]", row)

    return results


def main():
    all_results = {}
    for seed in [42, 123, 7]:
        all_results[f"seed_{seed}"] = run(seed)

    save_results({"axis_statistics": all_results})
    logger.info("Teste de estatisticas dos eixos concluido.")


if __name__ == "__main__":
    main()
