"""
Script de treinamento distribuido do DRM Transformer.

Uso:
    # Single GPU
    python scripts/train_distributed.py --config configs/scaling/15m.yaml --data-dir data/

    # Single GPU com resume
    python scripts/train_distributed.py --config configs/scaling/350m.yaml --resume auto

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 scripts/train_distributed.py --config configs/scaling/1.3b.yaml

    # Multi-node (2 nodes x 8 GPUs)
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
        --master_addr=<IP> --master_port=29500 \
        scripts/train_distributed.py --config configs/scaling/13b.yaml

    # Fine-tune de backbone
    python scripts/train_distributed.py --config configs/scaling/350m.yaml \
        --resume checkpoints/backbone.pt --finetune
"""

import sys
import json
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import torch
from drm_transformer.config import DRMTransformerConfig
from drm_transformer.model import DRMTransformer
from drm_transformer.training.distributed import (
    setup_distributed,
    cleanup_distributed,
    wrap_model_ddp,
)
from drm_transformer.training.trainer import DRMTrainer
from drm_transformer.training.data import create_dataloader
from drm_transformer.training.reproducibility import (
    set_seed,
    set_deterministic,
    build_run_manifest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_config(path: str) -> dict:
    """Carrega config YAML e retorna como dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def _find_latest_checkpoint(save_dir: str) -> str:
    """Encontra checkpoint mais recente no diretorio."""
    save_path = Path(save_dir)
    if not save_path.exists():
        return ""
    checkpoints = list(save_path.glob("step_*.pt"))
    if not checkpoints:
        return ""

    def _step_num(p):
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    checkpoints.sort(key=_step_num)
    return str(checkpoints[-1])


def _detect_vocab_size(data_dir: str) -> int:
    """Detecta vocab_size do metadata dos dados."""
    meta_path = Path(data_dir) / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("vocab_size", 0)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Treinamento DRM Transformer")
    parser.add_argument("--config", required=True, help="Caminho do YAML")
    parser.add_argument("--resume", default="", help="Checkpoint ('auto' = mais recente)")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune: carrega pesos, reseta step")
    parser.add_argument("--data-dir", default="", help="Override diretorio de dados")
    parser.add_argument("--eval-data-dir", default="", help="Dados de avaliacao")
    parser.add_argument("--override", nargs="*", help="Overrides: key=value")
    parser.add_argument("--seed", type=int, default=42, help="Seed global (default: 42)")
    parser.add_argument("--deterministic", action="store_true", help="Ativa flags deterministicas")
    args = parser.parse_args()

    config = _load_config(args.config)

    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.override:
        for ov in args.override:
            key, val = ov.split("=", 1)
            if key in config:
                orig = config[key]
                if isinstance(orig, bool):
                    config[key] = val.lower() in ("true", "1", "yes")
                elif isinstance(orig, int):
                    config[key] = int(val)
                elif isinstance(orig, float):
                    config[key] = float(val)
                else:
                    config[key] = val

    # Reprodutibilidade: seed + determinismo
    seed = args.seed
    set_seed(seed)
    if args.deterministic:
        set_deterministic(warn_only=True)
    config["seed"] = seed

    data_dir = config.get("data_dir", "data/")
    data_vocab = _detect_vocab_size(data_dir)
    if data_vocab > 0 and data_vocab != config.get("vocab_size", 50257):
        logger.warning(
            "[CONFIG] vocab_size config=%d, dados=%d. Ajustando.",
            config.get("vocab_size", 50257), data_vocab,
        )
        config["vocab_size"] = data_vocab

    dist_info = setup_distributed(config)
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    is_main = dist_info["is_main"]
    device = dist_info["device"]
    config["_is_main"] = is_main

    model_config = DRMTransformerConfig(
        vocab_size=config.get("vocab_size", 50257),
        max_seq_len=config.get("max_seq_len", 1024),
        d_model=config.get("d_model", 768),
        n_layers=config.get("n_layers", 12),
        n_heads=config.get("n_heads", 12),
        d_ff=config.get("d_ff", 3072),
        dropout=config.get("dropout", 0.1),
        bias=config.get("bias", False),
        d_manifold=config.get("d_manifold", 16),
        metric_hidden=config.get("metric_hidden", 64),
        metric_rank=config.get("metric_rank", 4),   # <-- adicionar aqui
        n_quad=config.get("n_quad", 0),
        gamma_enabled=config.get("gamma_enabled", True),
        gamma_c=config.get("gamma_c", 4.0),
        gamma_alpha=config.get("gamma_alpha", 0.0),
        gravity_enabled=config.get("gravity_enabled", True),
        gravity_strength=config.get("gravity_strength", 0.1),
        gravity_n_rff=config.get("gravity_n_rff", 64),
        n_anchors=config.get("n_anchors", 6),
        temperature_init=config.get("temperature_init", 1.0),
        temperature_min=config.get("temperature_min", 0.5),
        variable_dim=config.get("variable_dim", True),
    )

    if is_main:
        logger.info("[CONFIG] %s", args.config)
        logger.info(
            "[CONFIG] d_model=%d, n_layers=%d, n_heads=%d, d_manifold=%d",
            model_config.d_model, model_config.n_layers,
            model_config.n_heads, model_config.d_manifold,
        )
        logger.info(
            "[CONFIG] gamma=%s, gravity=%s (%.3f), variable_dim=%s",
            model_config.gamma_enabled, model_config.gravity_enabled,
            model_config.gravity_strength, model_config.variable_dim,
        )
        logger.info(
            "[CONFIG] batch=%d x accum=%d x world=%d",
            config.get("batch_size", 16),
            config.get("gradient_accumulation_steps", 1),
            world_size,
        )

    model = DRMTransformer(model_config)

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("[MODEL] Parametros: %s", f"{total_params:,}")
        param_bytes = total_params * 2
        optim_bytes = total_params * 12
        total_gb = (param_bytes + optim_bytes) / (1024 ** 3)
        logger.info("[MODEL] Memoria estimada (model+optim): %.1f GB", total_gb)

    model = wrap_model_ddp(model, config, device)

    # Run manifest (apenas no processo principal)
    if is_main:
        save_dir = config.get("save_dir", "checkpoints")
        manifest = build_run_manifest(
            config=config,
            seed=seed,
            config_path=args.config,
            save_dir=save_dir,
        )

    train_loader = create_dataloader(
        data_dir=data_dir,
        seq_len=config.get("max_seq_len", 1024),
        batch_size=config.get("batch_size", 16),
        max_tokens=config.get("total_tokens", 0),
        rank=rank,
        world_size=world_size,
    )

    eval_loader = None
    if args.eval_data_dir:
        eval_loader = create_dataloader(
            data_dir=args.eval_data_dir,
            seq_len=config.get("max_seq_len", 1024),
            batch_size=config.get("batch_size", 16),
            rank=rank,
            world_size=world_size,
        )

    trainer = DRMTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )

    resume_path = args.resume
    if resume_path.lower() == "auto":
        save_dir = config.get("save_dir", "checkpoints")
        resume_path = _find_latest_checkpoint(save_dir)
        if resume_path and is_main:
            logger.info("[RESUME] Auto-detectado: %s", resume_path)
        elif is_main:
            logger.info("[RESUME] Nenhum checkpoint, iniciando do zero.")

    if resume_path:
        if args.finetune:
            state = torch.load(
                resume_path, map_location=device, weights_only=False,
            )
            model_state = state.get("model", state)
            cleaned = {
                k.replace("module.", "").replace("_orig_mod.", ""): v
                for k, v in model_state.items()
            }
            trainer.raw_model.load_state_dict(cleaned, strict=False)
            if is_main:
                logger.info("[FINETUNE] Pesos carregados (step resetado)")
        else:
            trainer.load_checkpoint(resume_path)
            if is_main:
                logger.info("[RESUME] Step %d", trainer.global_step)

    history = trainer.train()

    if is_main:
        logger.info("[DONE] Treinamento completo.")
        logger.info("  Tempo: %.0fs", history["total_time"])
        logger.info("  Steps: %d", history["steps"])
        logger.info("  Skip-grads: %d", history["skip_grads"])

    cleanup_distributed()


if __name__ == "__main__":
    main()
