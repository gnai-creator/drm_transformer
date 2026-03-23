"""Runner master: executa todos os testes e salva results.json."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
from utils import load_results, save_results, RESULTS_PATH, logger

# Importar modulos de teste
import test_axes_projection
import test_axis_statistics
import test_semantic_separation
import test_geometry_correlation


def main():
    logger.info("=" * 60)
    logger.info("DRM Transformer - Avaliacao Empirica Completa")
    logger.info("=" * 60)

    logger.info("\n[1/4] Axes Projection...")
    test_axes_projection.main()

    logger.info("\n[2/4] Axis Statistics...")
    test_axis_statistics.main()

    logger.info("\n[3/4] Semantic Separation...")
    test_semantic_separation.main()

    logger.info("\n[4/4] Geometry Correlation...")
    test_geometry_correlation.main()

    # Resumo final
    results = load_results()
    logger.info("\n" + "=" * 60)
    logger.info("Resultados salvos em: %s", RESULTS_PATH)
    logger.info("Chaves: %s", list(results.keys()))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
