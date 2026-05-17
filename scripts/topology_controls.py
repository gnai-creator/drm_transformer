"""Roda controles topologicos para DRM Transformer.

Este script organiza os controles que viraram recorrentes no experimento:

- random init;
- best.pt e final.pt do baseline;
- checkpoints intermediarios step_*.pt;
- ablacoes existentes;
- seeds adicionais, com treino opcional.

Ele reaproveita scripts/extract_drm_vectors.py e scripts/voronoi_foliation_drm.py
para manter a avaliacao identica ao pipeline principal.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drm_transformer import DRMTransformer, DRMTransformerConfig
from drm_transformer.training.reproducibility import set_seed


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASELINE_DIR = Path("checkpoints/baseline_3.5m")
ABLATION_DIR = Path("checkpoints/ablations")
ABLATIONS = [
    "no_gravity",
    "no_gamma",
    "no_variable_dim",
    "no_torus",
    "annealed_torus",
    "generic_geometry",
]


@dataclass
class Control:
    name: str
    checkpoint: Path
    kind: str
    seed: int | None = None


def run(cmd: list[str], dry_run: bool = False) -> None:
    logger.info("[CMD] %s", " ".join(cmd))
    if dry_run:
        return
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_random_init_checkpoint(
    config_path: Path,
    output_dir: Path,
    seed: int,
    dry_run: bool = False,
) -> Path:
    out = output_dir / "checkpoints" / f"random_init_seed_{seed}.pt"
    if dry_run:
        return out
    if out.exists():
        return out

    cfg = load_yaml(config_path)
    model_cfg = DRMTransformerConfig(**{
        k: v for k, v in cfg.items()
        if hasattr(DRMTransformerConfig, k)
    })
    set_seed(seed)
    model = DRMTransformer(model_cfg)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg,
            "global_step": 0,
            "control": "random_init",
            "seed": seed,
        },
        out,
    )
    logger.info("[RANDOM_INIT] %s", out)
    return out


def train_seed_if_needed(args: argparse.Namespace, seed: int) -> Path:
    save_dir = Path(args.seed_checkpoint_root) / f"seed_{seed}"
    checkpoint = save_dir / args.checkpoint_name
    if checkpoint.exists() and not args.force_train:
        logger.info("[TRAIN][SKIP] seed=%d ja existe: %s", seed, checkpoint)
        return checkpoint

    cmd = [
        sys.executable,
        "scripts/train_distributed.py",
        "--config",
        str(args.config),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--override",
        f"save_dir={save_dir}",
    ]
    if args.deterministic:
        cmd.append("--deterministic")
    if args.data_dir_override:
        cmd.extend(["--data-dir", args.data_dir_override])
    if args.eval_data_dir:
        cmd.extend(["--eval-data-dir", args.eval_data_dir])

    run(cmd, dry_run=args.dry_run)
    return checkpoint


def discover_controls(args: argparse.Namespace) -> list[Control]:
    controls: list[Control] = []

    if args.include_random_init:
        ckpt = save_random_init_checkpoint(
            args.config,
            args.output_dir,
            args.random_seed,
            dry_run=args.dry_run,
        )
        controls.append(Control("random_init", ckpt, "random_init", args.random_seed))

    if args.include_baseline:
        for name in ["best", "final"]:
            ckpt = BASELINE_DIR / f"{name}.pt"
            if ckpt.exists():
                controls.append(Control(f"baseline_{name}", ckpt, "baseline"))
            else:
                logger.warning("[SKIP] checkpoint ausente: %s", ckpt)

    if args.include_steps:
        for ckpt in sorted(BASELINE_DIR.glob("step_*.pt")):
            controls.append(Control(f"baseline_{ckpt.stem}", ckpt, "checkpoint_sweep"))

    if args.include_ablations:
        for name in ABLATIONS:
            ckpt = ABLATION_DIR / name / args.checkpoint_name
            if ckpt.exists():
                controls.append(Control(f"ablation_{name}", ckpt, "ablation"))
            else:
                logger.warning("[SKIP] ablacao ausente: %s", ckpt)

    for seed in parse_int_list(args.train_seeds):
        ckpt = train_seed_if_needed(args, seed)
        controls.append(Control(f"seed_{seed}", ckpt, "seed", seed))

    for seed in parse_int_list(args.existing_seeds):
        ckpt = Path(args.seed_checkpoint_root) / f"seed_{seed}" / args.checkpoint_name
        if ckpt.exists():
            controls.append(Control(f"seed_{seed}", ckpt, "seed", seed))
        else:
            logger.warning("[SKIP] seed ausente: %s", ckpt)

    if args.only:
        wanted = set(item.strip() for item in args.only.split(",") if item.strip())
        controls = [c for c in controls if c.name in wanted]

    if not controls:
        raise SystemExit("Nenhum controle encontrado para avaliar.")
    return controls


def parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def control_dir(output_dir: Path, control: Control) -> Path:
    safe = control.name.replace("/", "_").replace("\\", "_")
    return output_dir / "runs" / safe


def extract_vectors(args: argparse.Namespace, control: Control, out_dir: Path) -> None:
    coords = out_dir / "drm_coords.npy"
    if coords.exists() and not args.force_extract:
        logger.info("[EXTRACT][SKIP] %s", control.name)
        return

    cmd = [
        sys.executable,
        "scripts/extract_drm_vectors.py",
        "--checkpoint",
        str(control.checkpoint),
        "--data-dir",
        args.extract_data_dir,
        "--output-dir",
        str(out_dir),
        "--max-tokens",
        str(args.max_tokens),
        "--max-seqs",
        str(args.max_seqs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--label",
        "drm",
    ]
    run(cmd, dry_run=args.dry_run)


def run_foliation(args: argparse.Namespace, control: Control, out_dir: Path) -> None:
    result = out_dir / "foliation_results.json"
    if result.exists() and not args.force_foliation:
        logger.info("[FOLIATION][SKIP] %s", control.name)
        return

    cmd = [
        sys.executable,
        "scripts/voronoi_foliation_drm.py",
        "--coords",
        str(out_dir / "drm_coords.npy"),
        "--G-diag",
        str(out_dir / "drm_G_diag.npy"),
        "--gamma",
        str(out_dir / "drm_gamma.npy"),
        "--output-dir",
        str(out_dir),
        "--n-seeds",
        str(args.n_seeds),
        "--n-restarts",
        str(args.n_restarts),
        "--homology-points",
        str(args.homology_points),
        "--homology-restarts",
        str(args.homology_restarts),
        "--homology-long-bar-ratio",
        str(args.homology_long_bar_ratio),
        "--homology-projection",
        args.homology_projection,
        "--homology-density-quantile",
        str(args.homology_density_quantile),
    ]
    if args.use_gamma_distance:
        cmd.append("--use-gamma-distance")
    run(cmd, dry_run=args.dry_run)


def read_result(control: Control, out_dir: Path) -> dict:
    path = out_dir / "foliation_results.json"
    if not path.exists():
        return {
            "name": control.name,
            "kind": control.kind,
            "status": "missing",
            "checkpoint": str(control.checkpoint),
        }

    with open(path, encoding="utf-8") as f:
        result = json.load(f)

    hom = result.get("homology", {})
    hdata = hom.get("homology", {})
    h1 = hdata.get("H1", {})
    h2 = hdata.get("H2", {})
    h1_counts = h1.get("long_bars_by_restart", [])
    h2_counts = h2.get("long_bars_by_restart", [])
    near_t2 = near_t2_fraction(h1_counts, h2_counts)

    return {
        "name": control.name,
        "kind": control.kind,
        "seed": control.seed,
        "status": "ok",
        "checkpoint": str(control.checkpoint),
        "topology": hom.get("topology", "?"),
        "h1_long": h1.get("long_bars"),
        "h2_long": h2.get("long_bars"),
        "h1_by_restart": h1_counts,
        "h2_by_restart": h2_counts,
        "t2_stable_fraction": hom.get("t2_stable_fraction"),
        "near_t2_fraction": near_t2,
        "foliation_score": result.get("foliation_score"),
        "ari": result.get("stability", {}).get("mean_ari"),
        "ari_std": result.get("stability", {}).get("std_ari"),
        "mean_eff_dim": result.get("ltsa", {}).get("mean_eff_dim"),
        "coherent_fraction": result.get("coherence", {}).get("coherent_fraction"),
        "homology_projection": hom.get("projection"),
        "density_quantile": hom.get("density_quantile"),
        "pca_explained_variance": hom.get("pca_explained_variance"),
        "path": str(path),
    }


def near_t2_fraction(h1_counts: list[int], h2_counts: list[int]) -> float:
    if not h1_counts or not h2_counts:
        return 0.0
    n = min(len(h1_counts), len(h2_counts))
    good = 0
    for h1, h2 in zip(h1_counts[:n], h2_counts[:n]):
        if h2 == 1 and h1 in (1, 2, 3):
            good += 1
    return good / n


def write_summary(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "topology_controls.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    lines = [
        "# Topology Controls",
        "",
        f"Gerado em: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "| Controle | Tipo | Topologia | H1 | H2 | T2 stable | Near T2 | F | ARI | EffDim | Coerencia |",
        "|----------|------|-----------|----|----|-----------|---------|---|-----|--------|-----------|",
    ]

    def fmt(value, digits: int = 3) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    for row in rows:
        lines.append(
            f"| {row['name']} | {row.get('kind', '-')} | {row.get('topology', '-')} "
            f"| {fmt(row.get('h1_long'), 0)} | {fmt(row.get('h2_long'), 0)} "
            f"| {fmt(row.get('t2_stable_fraction'))} | {fmt(row.get('near_t2_fraction'))} "
            f"| {fmt(row.get('foliation_score'))} | {fmt(row.get('ari'))} "
            f"| {fmt(row.get('mean_eff_dim'))} | {fmt(row.get('coherent_fraction'))} |"
        )

    lines.extend([
        "",
        "Criterio estrito: `H1=2`, `H2=1`, `T2 stable >= 0.60`.",
        "Near T2 conta restarts com `H2=1` e `H1` em `{1,2,3}` para diagnostico, nao para validacao final.",
        "",
    ])
    with open(output_dir / "topology_controls.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Controles topologicos DRM")
    parser.add_argument("--config", type=Path, default=Path("configs/baselines/small_3.5M.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval-results/topology_controls"))
    parser.add_argument("--extract-data-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--checkpoint-name", default="final.pt")
    parser.add_argument("--include-random-init", action="store_true", default=True)
    parser.add_argument("--no-random-init", dest="include_random_init", action="store_false")
    parser.add_argument("--include-baseline", action="store_true", default=True)
    parser.add_argument("--no-baseline", dest="include_baseline", action="store_false")
    parser.add_argument("--include-ablations", action="store_true", default=True)
    parser.add_argument("--no-ablations", dest="include_ablations", action="store_false")
    parser.add_argument("--include-steps", action="store_true")
    parser.add_argument("--only", default="", help="Nomes de controles separados por virgula")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--existing-seeds", default="",
                        help="Seeds ja treinadas em --seed-checkpoint-root/seed_<seed>")
    parser.add_argument("--train-seeds", default="",
                        help="Treina seeds e depois avalia: ex. 42,123,2025")
    parser.add_argument("--seed-checkpoint-root", default="checkpoints/topology_controls")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--data-dir-override", default="")
    parser.add_argument("--eval-data-dir", default="")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--force-foliation", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=100_000)
    parser.add_argument("--max-seqs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-seeds", type=int, default=80)
    parser.add_argument("--n-restarts", type=int, default=10)
    parser.add_argument("--homology-points", type=int, default=1200)
    parser.add_argument("--homology-restarts", type=int, default=10)
    parser.add_argument("--homology-long-bar-ratio", type=float, default=0.80)
    parser.add_argument("--homology-projection", choices=["none", "pca2", "pca3"], default="pca3")
    parser.add_argument("--homology-density-quantile", type=float, default=0.10)
    parser.add_argument("--use-gamma-distance", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (0.0 < args.homology_long_bar_ratio <= 1.0):
        raise SystemExit("--homology-long-bar-ratio deve estar em (0, 1].")
    if not (0.0 <= args.homology_density_quantile < 1.0):
        raise SystemExit("--homology-density-quantile deve estar em [0, 1).")

    controls = discover_controls(args)
    logger.info("[CONTROLS] %s", ", ".join(c.name for c in controls))

    rows = []
    for control in controls:
        out_dir = control_dir(args.output_dir, control)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 72)
        logger.info("[%s] %s", control.kind, control.name)
        logger.info("=" * 72)
        extract_vectors(args, control, out_dir)
        run_foliation(args, control, out_dir)
        if not args.dry_run:
            rows.append(read_result(control, out_dir))

    if args.dry_run:
        logger.info("[DONE] dry-run concluido")
        return

    write_summary(rows, args.output_dir)
    logger.info("[DONE] %s", args.output_dir / "topology_controls.md")


if __name__ == "__main__":
    main()
