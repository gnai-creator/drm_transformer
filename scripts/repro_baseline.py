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
                 "--output", "checkpoints/baseline_1m/eval_results.json"],
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
    ablation_results = save_dir / "results_ablations.json"
    if ablation_results.exists():
        with open(ablation_results) as f:
            report["ablations"] = json.load(f)

    # KPIs consolidados
    report["kpis"] = _compute_kpis(report)
    _print_kpis(report["kpis"])

    # Salvar relatorio no diretorio do baseline
    save_dir = Path("checkpoints/baseline_1m")
    save_dir.mkdir(parents=True, exist_ok=True)

    report_path = save_dir / "repro_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("[SAVED] %s", report_path)

    # Gerar plots
    _generate_plots(save_dir)

    return 0 if steps_ok == steps_total else 1


def _generate_plots(save_dir: Path) -> None:
    """Gera plots de training a partir do training_log.json."""
    log_path = save_dir / "training_log.json"
    if not log_path.exists():
        logger.warning("[PLOTS] training_log.json nao encontrado em %s", save_dir)
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("[PLOTS] matplotlib nao instalado, pulando plots")
        return

    with open(log_path) as f:
        log = json.load(f)

    train_entries = [e for e in log if e.get("type") != "eval"]
    eval_entries = [e for e in log if e.get("type") == "eval"]

    if not train_entries:
        logger.warning("[PLOTS] Nenhuma entrada de treino no log")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DRM Transformer Baseline Training Report", fontsize=14, fontweight="bold")

    # 1. Train Loss
    ax = axes[0, 0]
    steps = [e["step"] for e in train_entries]
    losses = [e["loss"] for e in train_entries]
    ax.plot(steps, losses, linewidth=0.8, color="#2196F3")
    ax.set_xlabel("Step")
    ax.set_ylabel("Train Loss")
    ax.set_title("Train Loss")
    ax.grid(True, alpha=0.3)

    # 2. Val Loss + PPL
    ax = axes[0, 1]
    if eval_entries:
        eval_steps = [e["step"] for e in eval_entries]
        val_losses = [e["val_loss"] for e in eval_entries]
        val_ppls = [e.get("val_ppl", 0) for e in eval_entries]
        ax.plot(eval_steps, val_losses, "o-", color="#F44336", markersize=3, label="Val Loss")
        ax2 = ax.twinx()
        ax2.plot(eval_steps, val_ppls, "s--", color="#FF9800", markersize=3, label="Val PPL")
        ax2.set_ylabel("Perplexity", color="#FF9800")
        ax.set_xlabel("Step")
        ax.set_ylabel("Val Loss", color="#F44336")
        ax.set_title("Validation Loss & Perplexity")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No eval data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Validation Loss & Perplexity")
    ax.grid(True, alpha=0.3)

    # 3. Learning Rate
    ax = axes[1, 0]
    if "lr" in train_entries[0]:
        lrs = [e["lr"] for e in train_entries]
        ax.plot(steps, lrs, linewidth=0.8, color="#4CAF50")
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)

    # 4. Tokens/s
    ax = axes[1, 1]
    if "tokens_per_sec" in train_entries[0]:
        tps = [e["tokens_per_sec"] for e in train_entries]
        ax.plot(steps, tps, linewidth=0.8, color="#9C27B0")
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens/s")
    ax.set_title("Throughput")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = save_dir / "training_report.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[PLOTS] Salvo: %s", plot_path)


def _compute_kpis(report: dict) -> dict:
    """Computa KPIs consolidados a partir do relatorio.

    KPIs:
      - reproducibility: desvio de PPL entre runs (requer 2+ runs)
      - reliability: % de steps que terminaram sem erro
      - cost: tokens/s e horas por experimento
      - quality: melhor PPL do baseline e das ablacoes
      - comparability: % de experimentos com manifest completo
    """
    kpis = {}
    baseline = report.get("baseline", {})

    # --- Reliability: % runs sem erro ---
    steps_ok = report.get("steps_ok", 0)
    steps_total = report.get("steps_total", 1)
    kpis["reliability"] = {
        "steps_ok": steps_ok,
        "steps_total": steps_total,
        "success_rate_pct": round(100.0 * steps_ok / max(steps_total, 1), 1),
    }

    # --- Cost: tokens/s e tempo ---
    kpis["cost"] = {
        "avg_tokens_per_s": baseline.get("avg_tokens_per_s"),
        "total_time_s": report.get("total_time_s"),
        "total_time_min": round(report.get("total_time_s", 0) / 60, 1),
        "baseline_time_s": baseline.get("total_time_s"),
        "baseline_tokens": baseline.get("total_tokens"),
    }

    # --- Quality: PPL baseline + ablacoes ---
    quality = {
        "baseline_val_ppl": baseline.get("best_val_ppl"),
        "baseline_val_loss": baseline.get("best_val_loss"),
    }

    ablations = report.get("ablations", [])
    if ablations:
        best_ablation = None
        worst_ablation = None
        for ab in ablations:
            name = ab.get("name", "?")
            ppl = ab.get("best_val_ppl")
            if ppl is None:
                continue
            quality[f"{name}_val_ppl"] = ppl
            if best_ablation is None or ppl < best_ablation[1]:
                best_ablation = (name, ppl)
            if worst_ablation is None or ppl > worst_ablation[1]:
                worst_ablation = (name, ppl)

        if best_ablation:
            quality["best_variant"] = best_ablation[0]
            quality["best_variant_ppl"] = best_ablation[1]
        if worst_ablation:
            quality["worst_variant"] = worst_ablation[0]
            quality["worst_variant_ppl"] = worst_ablation[1]
        if best_ablation and worst_ablation:
            quality["ppl_spread"] = round(worst_ablation[1] - best_ablation[1], 2)

    kpis["quality"] = quality

    # --- Comparability: % experimentos com manifest ---
    manifest_dirs = [
        Path("checkpoints/baseline_1m"),
        Path("checkpoints/ablations/full"),
        Path("checkpoints/ablations/no_gravity"),
        Path("checkpoints/ablations/no_gamma"),
        Path("checkpoints/ablations/no_variable_dim"),
    ]
    n_with_manifest = sum(
        1 for d in manifest_dirs
        if (d / "run_manifest.json").exists()
    )
    n_with_metrics = sum(
        1 for d in manifest_dirs
        if (d / "metrics.json").exists()
    )
    n_total = sum(1 for d in manifest_dirs if d.exists())
    kpis["comparability"] = {
        "experiments_total": n_total,
        "with_manifest": n_with_manifest,
        "with_metrics": n_with_metrics,
        "manifest_pct": round(100.0 * n_with_manifest / max(n_total, 1), 1),
        "complete_pct": round(
            100.0 * min(n_with_manifest, n_with_metrics) / max(n_total, 1), 1
        ),
    }

    # --- Reproducibility: placeholder (requer 2+ runs) ---
    kpis["reproducibility"] = {
        "note": "Rodar 2x com mesma seed para medir desvio de PPL",
        "ppl_deviation": None,
        "seeds_tested": [report.get("seed")],
    }

    return kpis


def _print_kpis(kpis: dict) -> None:
    """Imprime dashboard de KPIs no console."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  KPI DASHBOARD")
    logger.info("=" * 60)

    # Reliability
    rel = kpis.get("reliability", {})
    logger.info("")
    logger.info("  RELIABILITY")
    logger.info("    Success rate:    %s%% (%d/%d steps)",
                rel.get("success_rate_pct", "?"),
                rel.get("steps_ok", 0),
                rel.get("steps_total", 0))

    # Cost
    cost = kpis.get("cost", {})
    logger.info("")
    logger.info("  COST")
    logger.info("    Tokens/s:        %s", cost.get("avg_tokens_per_s", "?"))
    logger.info("    Total time:      %s min", cost.get("total_time_min", "?"))

    # Quality
    qual = kpis.get("quality", {})
    logger.info("")
    logger.info("  QUALITY")
    logger.info("    Baseline PPL:    %s", qual.get("baseline_val_ppl", "?"))
    if qual.get("best_variant"):
        logger.info("    Best variant:    %s (PPL=%s)",
                    qual["best_variant"], qual.get("best_variant_ppl", "?"))
    if qual.get("worst_variant"):
        logger.info("    Worst variant:   %s (PPL=%s)",
                    qual["worst_variant"], qual.get("worst_variant_ppl", "?"))
    if qual.get("ppl_spread") is not None:
        logger.info("    PPL spread:      %s", qual["ppl_spread"])

    # Comparability
    comp = kpis.get("comparability", {})
    logger.info("")
    logger.info("  COMPARABILITY")
    logger.info("    Manifest:        %s%% (%d/%d)",
                comp.get("manifest_pct", "?"),
                comp.get("with_manifest", 0),
                comp.get("experiments_total", 0))
    logger.info("    Complete:        %s%%", comp.get("complete_pct", "?"))

    # Reproducibility
    repro = kpis.get("reproducibility", {})
    logger.info("")
    logger.info("  REPRODUCIBILITY")
    if repro.get("ppl_deviation") is not None:
        logger.info("    PPL deviation:   %s", repro["ppl_deviation"])
    else:
        logger.info("    PPL deviation:   (pendente — rodar 2x com mesma seed)")

    logger.info("")
    logger.info("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
