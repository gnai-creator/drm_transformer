"""H3: Correlacao entre distancia geodesica e distancia semantica.

Usa embeddings pre-manifold para d_sem (evita correlacionar modelo consigo).
Compara: geodesica (U_real) vs euclidiana (G=I) vs random U (ablation).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
from scipy.stats import pearsonr
from utils import (
    set_seed, create_model, tokenize_texts, get_U_and_coords,
    aggregate_heads_mean, compute_geodesic_dist, compute_euclidean_dist,
    get_all_texts_and_labels, save_results, logger,
)


def _pairwise_cosine_dist(x: np.ndarray) -> np.ndarray:
    """Distancia cosseno pairwise."""
    from sklearn.metrics.pairwise import cosine_distances
    return cosine_distances(x)


def _upper_tri(mat: np.ndarray) -> np.ndarray:
    """Extrai triangulo superior (sem diagonal)."""
    idx = np.triu_indices_from(mat, k=1)
    return mat[idx]


def run(seed: int = 42):
    """Testa correlacao geometria-semantica."""
    set_seed(seed)
    model, config = create_model()
    device = next(model.parameters()).device

    texts, labels = get_all_texts_and_labels()
    input_ids = tokenize_texts(texts, vocab_size=config.vocab_size).to(device)

    U_all, coords_all, x_pre = get_U_and_coords(model, input_ids, layer_idx=-1)
    U_mean, coords_mean = aggregate_heads_mean(U_all, coords_all)

    # Media sobre tokens -> representacao por amostra
    coords_avg = coords_mean.mean(dim=1)  # [B, D]
    U_avg = U_mean.mean(dim=1)  # [B, D, r]
    x_pre_avg = x_pre.mean(dim=1).cpu().numpy()  # [B, d_model]

    # d_sem: distancia cosseno no espaco pre-manifold
    d_sem = _pairwise_cosine_dist(x_pre_avg)
    d_sem_flat = _upper_tri(d_sem)

    results = {}

    # d_geom: distancia geodesica com U real
    d_geom = compute_geodesic_dist(coords_avg, U_avg).cpu().numpy()
    d_geom_flat = _upper_tri(d_geom)
    corr_geom, p_geom = pearsonr(d_geom_flat, d_sem_flat)
    results["geodesic_semantic"] = {"corr": corr_geom, "p_value": p_geom}
    logger.info("Geodesic-semantic  corr=%.4f (p=%.2e)", corr_geom, p_geom)

    # d_euc: distancia euclidiana (G=I, sem U)
    d_euc = compute_euclidean_dist(coords_avg).cpu().numpy()
    d_euc_flat = _upper_tri(d_euc)
    corr_euc, p_euc = pearsonr(d_euc_flat, d_sem_flat)
    results["euclidean_semantic"] = {"corr": corr_euc, "p_value": p_euc}
    logger.info("Euclidean-semantic  corr=%.4f (p=%.2e)", corr_euc, p_euc)

    # d_rand: geodesica com U random (ablation)
    U_rand = torch.randn_like(U_avg)
    d_rand = compute_geodesic_dist(coords_avg, U_rand).cpu().numpy()
    d_rand_flat = _upper_tri(d_rand)
    corr_rand, p_rand = pearsonr(d_rand_flat, d_sem_flat)
    results["random_semantic"] = {"corr": corr_rand, "p_value": p_rand}
    logger.info("Random-semantic     corr=%.4f (p=%.2e)", corr_rand, p_rand)

    return results


def main():
    all_seeds = {}
    for seed in [42, 123, 7]:
        logger.info("\n=== Seed %d ===", seed)
        all_seeds[f"seed_{seed}"] = run(seed)

    # Resumo
    for condition in ["geodesic_semantic", "euclidean_semantic", "random_semantic"]:
        corrs = [
            all_seeds[f"seed_{s}"][condition]["corr"]
            for s in [42, 123, 7]
        ]
        logger.info(
            "\n[RESUMO] %s: corr=%.4f +/- %.4f",
            condition, np.mean(corrs), np.std(corrs),
        )

    save_results({"geometry_correlation": all_seeds})
    logger.info("Teste de correlacao geometria-semantica concluido.")


if __name__ == "__main__":
    main()
