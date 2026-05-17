"""Treina ablacoes, extrai vetores DRM e roda Voronoi/Foliation.

Uso rapido:
    python scripts/run_ablation_foliation.py --seed 42 --deterministic

O script e conservador por padrao:
    - pula treinos quando o checkpoint final ja existe;
    - pula extracao quando os vetores ja existem;
    - pula Voronoi quando foliation_results.json ja existe.

Use --force-train, --force-extract ou --force-foliation para refazer etapas.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASELINE_CONFIG = Path("configs/baselines/small_3.5M.yaml")
BASELINE_DIR = Path("checkpoints/baseline_3.5m")
ABLATION_CONFIG_DIR = Path("configs/ablations")
ABLATION_CHECKPOINT_DIR = Path("checkpoints/ablations")

ABLATIONS = [
    "full",
    "no_gravity",
    "no_gamma",
    "no_variable_dim",
    "no_torus",
    "annealed_torus",
    "generic_geometry",
]


def run(cmd: list[str], dry_run: bool = False) -> None:
    logger.info("[CMD] %s", " ".join(cmd))
    if dry_run:
        return

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def prepare_data(args: argparse.Namespace) -> None:
    train_dir = Path("data/baseline/train")
    val_dir = Path("data/baseline/val")
    has_train = any(train_dir.glob("*.npy")) or any(train_dir.glob("*.bin"))
    has_val = any(val_dir.glob("*.npy")) or any(val_dir.glob("*.bin"))

    if has_train and has_val and not args.force_prepare_data:
        logger.info("[DATA][SKIP] shards baseline ja existem")
    else:
        logger.info("[DATA] preparando baseline publico")
        cmd = [sys.executable, "scripts/prepare_baseline_data.py"]
        if args.prepare_max_tokens:
            cmd.extend(["--max-tokens", str(args.prepare_max_tokens)])
        run(cmd, dry_run=args.dry_run)

    logger.info("[DATA] verificando baseline")
    run([sys.executable, "scripts/prepare_baseline_data.py", "--verify"], dry_run=args.dry_run)


def selected_ablations(value: str) -> list[str]:
    if not value:
        return ABLATIONS

    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(ABLATIONS))
    if unknown:
        raise SystemExit(f"Ablacoes desconhecidas: {', '.join(unknown)}")
    return names


def config_path(name: str, rerun_full: bool) -> Path:
    if name == "full" and not rerun_full:
        return BASELINE_CONFIG
    return ABLATION_CONFIG_DIR / f"{name}.yaml"


def checkpoint_dir(name: str, rerun_full: bool) -> Path:
    if name == "full" and not rerun_full:
        return BASELINE_DIR
    return ABLATION_CHECKPOINT_DIR / name


def checkpoint_path(name: str, rerun_full: bool, checkpoint_name: str) -> Path:
    return checkpoint_dir(name, rerun_full) / checkpoint_name


def resolved_extract_device(value: str) -> str:
    if value != "auto":
        return value

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def train_variant(name: str, args: argparse.Namespace) -> None:
    ckpt = checkpoint_path(name, args.rerun_full, args.checkpoint_name)
    if ckpt.exists() and not args.force_train:
        logger.info("[TRAIN][SKIP] %s ja existe: %s", name, ckpt)
        return

    cfg = config_path(name, args.rerun_full)
    if not cfg.exists():
        raise SystemExit(f"Config nao encontrada para {name}: {cfg}")

    cmd = [
        sys.executable,
        "scripts/train_distributed.py",
        "--config",
        str(cfg),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.deterministic:
        cmd.append("--deterministic")
    if args.data_dir:
        cmd.extend(["--data-dir", args.data_dir])
    if args.eval_data_dir:
        cmd.extend(["--eval-data-dir", args.eval_data_dir])

    logger.info("[TRAIN] %s", name)
    run(cmd, dry_run=args.dry_run)


def extract_variant(name: str, args: argparse.Namespace, out_dir: Path) -> None:
    coords = out_dir / "drm_coords.npy"
    if coords.exists() and not args.force_extract:
        logger.info("[EXTRACT][SKIP] %s ja existe: %s", name, coords)
        return

    ckpt = checkpoint_path(name, args.rerun_full, args.checkpoint_name)
    if not ckpt.exists() and not args.dry_run:
        raise SystemExit(f"Checkpoint nao encontrado para {name}: {ckpt}")

    cmd = [
        sys.executable,
        "scripts/extract_drm_vectors.py",
        "--checkpoint",
        str(ckpt),
        "--data-dir",
        args.extract_data_dir or args.data_dir or "data",
        "--output-dir",
        str(out_dir),
        "--max-tokens",
        str(args.max_tokens),
        "--max-seqs",
        str(args.max_seqs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        resolved_extract_device(args.device),
        "--label",
        "drm",
    ]

    logger.info("[EXTRACT] %s", name)
    run(cmd, dry_run=args.dry_run)


def foliation_variant(name: str, args: argparse.Namespace, out_dir: Path) -> None:
    result_path = out_dir / "foliation_results.json"
    if result_path.exists() and not args.force_foliation:
        logger.info("[FOLIATION][SKIP] %s ja existe: %s", name, result_path)
        return

    coords = out_dir / "drm_coords.npy"
    g_diag = out_dir / "drm_G_diag.npy"
    gamma = out_dir / "drm_gamma.npy"

    required = [coords, g_diag, gamma]
    missing = [str(path) for path in required if not path.exists()]
    if missing and not args.dry_run:
        raise SystemExit(f"Vetores ausentes para {name}: {', '.join(missing)}")

    cmd = [
        sys.executable,
        "scripts/voronoi_foliation_drm.py",
        "--coords",
        str(coords),
        "--G-diag",
        str(g_diag),
        "--gamma",
        str(gamma),
        "--output-dir",
        str(out_dir),
        "--n-seeds",
        str(args.n_seeds),
        "--homology-points",
        str(args.homology_points),
        "--homology-long-bar-ratio",
        str(args.homology_long_bar_ratio),
        "--homology-restarts",
        str(args.homology_restarts),
        "--n-restarts",
        str(args.n_restarts),
        "--homology-projection",
        args.homology_projection,
        "--homology-density-quantile",
        str(args.homology_density_quantile),
    ]
    if args.use_gamma_distance:
        cmd.append("--use-gamma-distance")

    logger.info("[FOLIATION] %s", name)
    run(cmd, dry_run=args.dry_run)


def load_summary(name: str, out_dir: Path) -> dict:
    path = out_dir / "foliation_results.json"
    if not path.exists():
        return {"name": name, "status": "missing"}

    with open(path, encoding="utf-8") as f:
        result = json.load(f)

    homology = result.get("homology", {})
    stability = result.get("stability", {})
    ltsa = result.get("ltsa", {})
    coherence = result.get("coherence", {})

    homology_counts = homology.get("homology", {})
    h1 = homology_counts.get("H1", {})
    h2 = homology_counts.get("H2", {})

    return {
        "name": name,
        "status": "ok",
        "topology": homology.get("topology", "?"),
        "h1_long": h1.get("long_bars"),
        "h2_long": h2.get("long_bars"),
        "t2_stable_fraction": homology.get("t2_stable_fraction"),
        "foliation_score": result.get("foliation_score"),
        "ari": stability.get("mean_ari"),
        "ari_std": stability.get("std_ari"),
        "mean_eff_dim": ltsa.get("mean_eff_dim"),
        "coherent_fraction": coherence.get("coherent_fraction"),
        "path": str(path),
    }


def write_summary(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    lines = [
        "# Ablations Foliation Summary",
        "",
        f"Gerado em: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "| Variante | Topologia | H1 | H2 | T2 stable | F | ARI | EffDim | Coerencia |",
        "|----------|-----------|----|----|-----------|---|-----|--------|-----------|",
    ]

    def fmt(value, digits: int = 3) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    for row in rows:
        lines.append(
            f"| {row['name']} | {row.get('topology', '-')} "
            f"| {fmt(row.get('h1_long'), 0)} | {fmt(row.get('h2_long'), 0)} "
            f"| {fmt(row.get('t2_stable_fraction'))} "
            f"| {fmt(row.get('foliation_score'))} | {fmt(row.get('ari'))} "
            f"| {fmt(row.get('mean_eff_dim'))} | {fmt(row.get('coherent_fraction'))} |"
        )

    lines.extend([
        "",
        "Criterio toroidal: `H1=2`, `H2=1` e `T2 stable >= 0.60`.",
        "",
    ])
    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina ablacoes e roda extracao + Voronoi/Foliation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="",
                        help="Override opcional do diretorio de treino")
    parser.add_argument("--eval-data-dir", default="")
    parser.add_argument("--extract-data-dir", default="data",
                        help="Diretorio usado para extrair vetores DRM")
    parser.add_argument("--output-dir", default="eval-results/ablations_foliation")
    parser.add_argument("--only", default="", help="Lista separada por virgula")
    parser.add_argument("--checkpoint-name", default="final.pt")
    parser.add_argument("--prepare-data", action="store_true",
                        help="Prepara e verifica data/baseline antes do treino")
    parser.add_argument("--force-prepare-data", action="store_true")
    parser.add_argument("--prepare-max-tokens", type=int, default=0,
                        help="Tokens para scripts/prepare_baseline_data.py")
    parser.add_argument("--rerun-full", action="store_true",
                        help="Treina full em checkpoints/ablations/full")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-foliation", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--force-foliation", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=100_000)
    parser.add_argument("--max-seqs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-seeds", type=int, default=80)
    parser.add_argument("--n-restarts", type=int, default=10)
    parser.add_argument("--homology-points", type=int, default=1200)
    parser.add_argument("--homology-restarts", type=int, default=5)
    parser.add_argument("--homology-long-bar-ratio", type=float, default=0.75)
    parser.add_argument("--homology-projection", choices=["none", "pca2", "pca3"],
                        default="none")
    parser.add_argument("--homology-density-quantile", type=float, default=0.0)
    parser.add_argument("--use-gamma-distance", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (0.0 < args.homology_long_bar_ratio <= 1.0):
        raise SystemExit("--homology-long-bar-ratio deve estar em (0, 1]. Use 0.75 ou 0.80.")
    if not (0.0 <= args.homology_density_quantile < 1.0):
        raise SystemExit("--homology-density-quantile deve estar em [0, 1).")

    names = selected_ablations(args.only)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[ABLATIONS] %s", ", ".join(names))
    logger.info("[OUTPUT] %s", output_dir)

    if args.prepare_data:
        prepare_data(args)

    for name in names:
        variant_dir = output_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 72)
        logger.info("[%s]", name)
        logger.info("=" * 72)

        if not args.skip_train:
            train_variant(name, args)
        if not args.skip_extract:
            extract_variant(name, args, variant_dir)
        if not args.skip_foliation:
            foliation_variant(name, args, variant_dir)

    if args.dry_run:
        logger.info("[DONE] dry-run concluido")
        return

    rows = [load_summary(name, output_dir / name) for name in names]
    write_summary(rows, output_dir)
    logger.info("[DONE] resumo: %s", output_dir / "summary.md")


if __name__ == "__main__":
    main()
