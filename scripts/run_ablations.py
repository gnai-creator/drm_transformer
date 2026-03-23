"""Roda matriz de ablacoes e consolida resultados.

Executa cada config de ablacao sequencialmente, coleta metrics.json
de cada run e gera results_ablations.md com tabela comparativa.

Uso:
    python scripts/run_ablations.py
    python scripts/run_ablations.py --seed 42 --deterministic
    python scripts/run_ablations.py --only full,no_gravity
    python scripts/run_ablations.py --collect-only   # so consolida resultados existentes

Requisitos:
    - Dataset baseline gerado (python scripts/prepare_baseline_data.py)
"""

import argparse
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ABLATION_DIR = Path("configs/ablations")
RESULTS_FILE = Path("results_ablations.md")

ABLATIONS = {
    "full": "Modelo completo (gravity + gamma + variable_dim)",
    "no_gravity": "Sem campo gravitacional",
    "no_gamma": "Sem gamma-scaling (Lorentz)",
    "no_variable_dim": "Sem DimensionalGate",
}


def run_ablation(name: str, seed: int, deterministic: bool) -> dict:
    """Executa uma ablacao e retorna metricas."""
    config_path = ABLATION_DIR / f"{name}.yaml"
    if not config_path.exists():
        logger.error("[SKIP] Config nao encontrada: %s", config_path)
        return {}

    logger.info("=" * 60)
    logger.info("[ABLATION] %s: %s", name, ABLATIONS.get(name, ""))
    logger.info("=" * 60)

    cmd = [
        sys.executable, "scripts/train_distributed.py",
        "--config", str(config_path),
        "--seed", str(seed),
    ]
    if deterministic:
        cmd.append("--deterministic")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error("[FAIL] %s retornou codigo %d", name, result.returncode)
        return {"name": name, "status": "FAIL", "wall_time_s": round(elapsed)}

    # Ler metrics.json
    metrics_path = Path(f"checkpoints/ablations/{name}/metrics.json")
    if not metrics_path.exists():
        logger.error("[FAIL] metrics.json nao encontrado: %s", metrics_path)
        return {"name": name, "status": "FAIL", "wall_time_s": round(elapsed)}

    with open(metrics_path) as f:
        metrics = json.load(f)

    metrics["name"] = name
    metrics["status"] = "OK"
    metrics["wall_time_s"] = round(elapsed)

    # Ler training_log para contar NaN/skip grads
    log_path = Path(f"checkpoints/ablations/{name}/training_log.json")
    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)
        train_entries = [e for e in log if e.get("type") != "eval"]
        if train_entries:
            metrics["final_train_loss"] = train_entries[-1].get("loss")

    logger.info("[DONE] %s: val_loss=%.4f, ppl=%.2f, time=%ds",
                name,
                metrics.get("best_val_loss", -1),
                metrics.get("best_val_ppl", -1),
                metrics["wall_time_s"])

    return metrics


def collect_results() -> list:
    """Coleta metrics.json de todas as ablacoes ja rodadas."""
    results = []
    for name in ABLATIONS:
        metrics_path = Path(f"checkpoints/ablations/{name}/metrics.json")
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics["name"] = name

        # Training log para train loss final
        log_path = Path(f"checkpoints/ablations/{name}/training_log.json")
        if log_path.exists():
            with open(log_path) as f:
                log = json.load(f)
            train_entries = [e for e in log if e.get("type") != "eval"]
            if train_entries:
                metrics["final_train_loss"] = train_entries[-1].get("loss")

        results.append(metrics)
    return results


def generate_table(results: list) -> str:
    """Gera tabela markdown com resultados."""
    lines = [
        "# Resultados Ablacoes DRM Transformer",
        "",
        f"Gerado em: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Matriz de Ablacao",
        "",
        "| Variante | Gravity | Gamma | VarDim | Train Loss | Val Loss | Val PPL | Steps | Tok/s | Skip Grads | Tempo |",
        "|----------|---------|-------|--------|-----------|----------|---------|-------|-------|------------|-------|",
    ]

    feature_map = {
        "full": (True, True, True),
        "no_gravity": (False, True, True),
        "no_gamma": (True, False, True),
        "no_variable_dim": (True, True, False),
    }

    for r in results:
        name = r.get("name", "?")
        gravity, gamma, vardim = feature_map.get(name, ("?", "?", "?"))

        train_loss = r.get("final_train_loss")
        val_loss = r.get("best_val_loss")
        val_ppl = r.get("best_val_ppl")
        steps = r.get("total_steps", "?")
        tok_s = r.get("avg_tokens_per_s", "?")
        skip = r.get("skip_grads", "?")
        wall = r.get("wall_time_s") or r.get("total_time_s")

        def _check(v):
            return "Y" if v else "N"

        def _fmt(v, fmt=".4f"):
            return f"{v:{fmt}}" if v is not None else "-"

        def _time(s):
            if s is None:
                return "-"
            if s < 60:
                return f"{s}s"
            return f"{s // 60}m{s % 60}s"

        lines.append(
            f"| {name} | {_check(gravity)} | {_check(gamma)} | {_check(vardim)} "
            f"| {_fmt(train_loss)} | {_fmt(val_loss)} | {_fmt(val_ppl, '.2f')} "
            f"| {steps} | {tok_s} | {skip} | {_time(wall)} |"
        )

    lines.extend([
        "",
        "## Legenda",
        "",
        "| Variante | Descricao |",
        "|----------|-----------|",
    ])
    for name, desc in ABLATIONS.items():
        lines.append(f"| {name} | {desc} |")

    lines.extend([
        "",
        "## Comando para Reproduzir",
        "",
        "```bash",
        "python scripts/prepare_baseline_data.py",
        "python scripts/run_ablations.py --seed 42 --deterministic",
        "```",
    ])

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Roda ablacoes DRM Transformer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--only", default="", help="Ablacoes especificas (virgula)")
    parser.add_argument("--collect-only", action="store_true",
                        help="Apenas consolida resultados existentes")
    args = parser.parse_args()

    if args.collect_only:
        results = collect_results()
    else:
        names = [n.strip() for n in args.only.split(",")] if args.only else list(ABLATIONS)

        results = []
        for name in names:
            if name not in ABLATIONS:
                logger.warning("[SKIP] Ablacao desconhecida: %s", name)
                continue
            r = run_ablation(name, args.seed, args.deterministic)
            if r:
                results.append(r)

    if not results:
        logger.error("[ERROR] Nenhum resultado encontrado")
        return

    # Gerar tabela
    table = generate_table(results)
    RESULTS_FILE.write_text(table)
    logger.info("\n%s", table)
    logger.info("[SAVED] %s", RESULTS_FILE)

    # Salvar JSON bruto
    json_path = RESULTS_FILE.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("[SAVED] %s", json_path)


if __name__ == "__main__":
    main()
