"""Script unico de reproducao do baseline DRM Transformer.

Executa todo o pipeline de reproducao do zero:
1. Gera dataset baseline (Wikipedia EN 10M tokens)
2. Verifica integridade via SHA256
3. Treina baseline small_1m
4. Roda ablacoes (full, no_gravity, no_gamma, no_variable_dim)
5. Avalia perplexity de todas as variantes
6. Gera relatorio consolidado

Uso:
    python scripts/repro_baseline.py
    python scripts/repro_baseline.py --skip-data       # pula geracao de dados
    python scripts/repro_baseline.py --skip-ablations   # so baseline
    python scripts/repro_baseline.py --seed 123
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _run(cmd: list, description: str) -> bool:
    """Executa comando e retorna True se sucesso."""
    logger.info("=" * 60)
    logger.info("[STEP] %s", description)
    logger.info("  cmd: %s", " ".join(cmd))
    logger.info("=" * 60)

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error("[FAIL] %s (code=%d, %.0fs)", description, result.returncode, elapsed)
        return False

    logger.info("[OK] %s (%.0fs)", description, elapsed)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Reproducao completa do baseline DRM Transformer"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--skip-data", action="store_true",
                        help="Pular geracao de dados (usa existentes)")
    parser.add_argument("--skip-ablations", action="store_true",
                        help="Apenas baseline, sem ablacoes")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Pular avaliacao final")
    args = parser.parse_args()

    py = sys.executable
    t_global = time.time()
    steps_ok = 0
    steps_total = 0

    # 1. Dataset baseline
    if not args.skip_data:
        steps_total += 1
        if _run([py, "scripts/prepare_baseline_data.py"],
                "Gerar dataset baseline (Wikipedia EN 10M tokens)"):
            steps_ok += 1
        else:
            logger.error("[ABORT] Falha na geracao de dados")
            return 1

        # Verificar integridade
        steps_total += 1
        if _run([py, "scripts/prepare_baseline_data.py", "--verify"],
                "Verificar integridade SHA256"):
            steps_ok += 1
        else:
            logger.error("[ABORT] Dataset corrompido")
            return 1
    else:
        logger.info("[SKIP] Geracao de dados")

    # 2. Treinar baseline
    steps_total += 1
    baseline_cmd = [
        py, "scripts/train_distributed.py",
        "--config", "configs/baselines/small_1m.yaml",
        "--seed", str(args.seed),
    ]
    if args.deterministic:
        baseline_cmd.append("--deterministic")

    if _run(baseline_cmd, "Treinar baseline small_1m"):
        steps_ok += 1
    else:
        logger.error("[ABORT] Falha no treino baseline")
        return 1

    # 3. Ablacoes
    if not args.skip_ablations:
        steps_total += 1
        ablation_cmd = [
            py, "scripts/run_ablations.py",
            "--seed", str(args.seed),
        ]
        if args.deterministic:
            ablation_cmd.append("--deterministic")

        if _run(ablation_cmd, "Rodar ablacoes (4 variantes)"):
            steps_ok += 1
        else:
            logger.warning("[WARN] Ablacoes falharam, continuando...")

    # 4. Avaliacao
    if not args.skip_eval:
        steps_total += 1
        if _run([py, "scripts/eval_standard.py", "--all-ablations",
                 "--output", "eval_results.json"],
                "Avaliar perplexity de todas as variantes"):
            steps_ok += 1
        else:
            logger.warning("[WARN] Avaliacao falhou")

    # 5. Relatorio final
    total_time = time.time() - t_global
    logger.info("")
    logger.info("=" * 60)
    logger.info("[REPRO] Concluido: %d/%d steps OK em %.0fs (%.1f min)",
                steps_ok, steps_total, total_time, total_time / 60)
    logger.info("=" * 60)

    # Coletar resultados
    report = {
        "seed": args.seed,
        "deterministic": args.deterministic,
        "steps_ok": steps_ok,
        "steps_total": steps_total,
        "total_time_s": round(total_time),
    }

    # Baseline metrics
    baseline_metrics = Path("checkpoints/baseline_1m/metrics.json")
    if baseline_metrics.exists():
        with open(baseline_metrics) as f:
            report["baseline"] = json.load(f)
        logger.info("[BASELINE] val_loss=%.4f ppl=%.2f",
                    report["baseline"].get("best_val_loss", -1),
                    report["baseline"].get("best_val_ppl", -1))

    # Ablation results
    ablation_results = Path("results_ablations.json")
    if ablation_results.exists():
        with open(ablation_results) as f:
            report["ablations"] = json.load(f)

    # Salvar relatorio
    report_path = Path("repro_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("[SAVED] %s", report_path)

    return 0 if steps_ok == steps_total else 1


if __name__ == "__main__":
    sys.exit(main())
