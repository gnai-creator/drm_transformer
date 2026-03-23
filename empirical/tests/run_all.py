"""Runner master: executa todos os testes, salva results.json e gera figuras."""

import argparse
import sys
from pathlib import Path

_test_dir = Path(__file__).resolve().parent
_figures_dir = _test_dir.parent / "figures"
sys.path.insert(0, str(_test_dir))
sys.path.insert(0, str(_figures_dir))

import logging
from utils import load_results, save_results, RESULTS_PATH, FIGURES_DIR, logger, set_checkpoint, set_output_dir, _load_vocab_mapping

# Importar modulos de teste
import test_axes_projection
import test_axis_statistics
import test_semantic_separation
import test_geometry_correlation
import test_calibration

# Importar modulos de plot
import plot_pca
import plot_tsne
import plot_axis_heatmap
import plot_axis_separation
import plot_combined
import plot_calibration


def main():
    parser = argparse.ArgumentParser(description="DRM Transformer - Avaliacao Empirica")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint .pt para avaliar (sem = pesos aleatorios)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Diretorio de saida para results.json e figures/ (default: empirical/)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Diretorio dos dados para carregar vocab_mapping.json",
    )
    args = parser.parse_args()

    if args.data_dir:
        _load_vocab_mapping(args.data_dir)

    if args.output_dir:
        set_output_dir(args.output_dir)
        logger.info("[OUTPUT] %s", args.output_dir)

    if args.checkpoint:
        set_checkpoint(args.checkpoint)
        logger.info("[CHECKPOINT] %s", args.checkpoint)
    else:
        logger.info("[MODEL] Sem checkpoint — pesos aleatorios")

    logger.info("=" * 60)
    logger.info("DRM Transformer - Avaliacao Empirica Completa")
    logger.info("=" * 60)

    # --- Testes ---
    logger.info("\n[1/5] Axes Projection...")
    test_axes_projection.main()

    logger.info("\n[2/5] Axis Statistics...")
    test_axis_statistics.main()

    logger.info("\n[3/5] Semantic Separation...")
    test_semantic_separation.main()

    logger.info("\n[4/5] Geometry Correlation...")
    test_geometry_correlation.main()

    logger.info("\n[5/5] Calibration (ECE, MCE, Brier, PPL)...")
    test_calibration.main()

    # --- Figuras ---
    logger.info("\n" + "-" * 60)
    logger.info("Gerando figuras...")
    logger.info("-" * 60)

    logger.info("\n[fig 1/6] PCA...")
    plot_pca.main()

    logger.info("[fig 2/6] t-SNE...")
    plot_tsne.main()

    logger.info("[fig 3/6] Axis Heatmap...")
    plot_axis_heatmap.main()

    logger.info("[fig 4/6] Axis Separation...")
    plot_axis_separation.main()

    logger.info("[fig 5/6] Combined Panel...")
    plot_combined.main()

    logger.info("[fig 6/6] Calibration...")
    plot_calibration.main()

    # Resumo final
    import utils as _u
    results = load_results()
    logger.info("\n" + "=" * 60)
    logger.info("Resultados salvos em: %s", _u.RESULTS_PATH)
    logger.info("Figuras salvas em: %s", _u.FIGURES_DIR)
    logger.info("Chaves: %s", list(results.keys()))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
